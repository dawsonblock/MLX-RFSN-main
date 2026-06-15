"""Metal kernel implementation for dense attention over reconstructed KV.

WARNING: This is NOT a true packed attention kernel. It decodes all compressed
blocks to dense tensors, concatenates them, and runs dense attention via a
custom Metal kernel. This is a transitional implementation that provides GPU
acceleration for the attention computation while the full packed Metal kernel
is being developed.

This module will be replaced by a true packed Metal kernel that consumes packed
codes and scales directly inside the shader.
"""
from __future__ import annotations

import time
from typing import Any

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore


# ---------------------------------------------------------------------------
# Metal kernel source code
# ---------------------------------------------------------------------------

# Main attention kernel: processes one query token at a time
# Uses template parameters for compile-time constants (num_q_heads, etc.)
_ATTENTION_KERNEL = """
// Thread layout: each thread processes one (query_head, query_token) pair
uint qh = thread_position_in_grid.x;  // query head index
uint qt = thread_position_in_grid.y;  // query token index

if (qh >= NUM_Q_HEADS || qt >= NUM_Q_TOKENS) return;

// Compute KV head index for GQA
uint kv_head = qh / Q_PER_KV;

// Query offset: [num_q_heads, num_q_tokens, head_dim]
uint q_offset = (qh * NUM_Q_TOKENS + qt) * HEAD_DIM;

// Output offset
uint out_offset = (qh * NUM_Q_TOKENS + qt) * HEAD_DIM;

// Read scale from 1-element float array
float scale_val = scale_arr[0];

// Read query start position from 1-element int array
int query_start = query_start_arr[0];

// Online softmax state
float running_max = -INFINITY;
float running_sum = 0.0;

// First pass: compute QK scores and online softmax
for (uint kv_t = 0; kv_t < NUM_KV_TOKENS; kv_t++) {
    // Causal mask: query at position (query_start + qt) can attend to kv at position kv_t if query_pos >= kv_pos
    int query_pos = query_start + int(qt);
    if (CAUSAL != 0 && int(kv_t) > query_pos) continue;

    // Compute dot product Q @ K for this token
    float score = 0.0;
    uint k_offset = (kv_head * NUM_KV_TOKENS + kv_t) * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {
        score += queries[q_offset + d] * keys[k_offset + d];
    }
    score *= scale_val;

    // Update online softmax
    float new_max = max(running_max, score);
    float old_scale = exp(running_max - new_max);
    // Guard against NaN when running_max is -inf
    if (running_max == -INFINITY) old_scale = 0.0;
    running_sum = running_sum * old_scale + exp(score - new_max);
    running_max = new_max;
}

// Second pass: accumulate weighted values
for (uint d = 0; d < HEAD_DIM; d++) {
    output[out_offset + d] = 0.0;
}

for (uint kv_t = 0; kv_t < NUM_KV_TOKENS; kv_t++) {
    // Causal mask
    int query_pos2 = query_start + int(qt);
    if (CAUSAL != 0 && int(kv_t) > query_pos2) continue;

    // Compute score again
    float score = 0.0;
    uint k_offset = (kv_head * NUM_KV_TOKENS + kv_t) * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {
        score += queries[q_offset + d] * keys[k_offset + d];
    }
    score *= scale_val;

    // Compute weight
    float weight = exp(score - running_max) / running_sum;

    // Accumulate weighted value
    uint v_offset = (kv_head * NUM_KV_TOKENS + kv_t) * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] += weight * values[v_offset + d];
    }
}
"""


class StrictPackedExecutionError(RuntimeError):
    """Raised when strict packed execution fails and fallback is disabled."""


def _probe_metal_capability() -> dict[str, Any]:
    """Probe whether Metal kernels actually work on this device.

    Returns a dict with:
        - available: bool
        - reason: str | None
    """
    if not HAS_MLX:
        return {"available": False, "reason": "MLX not installed"}

    # Check if we're on Apple Silicon with GPU
    import platform
    try:
        device = mx.default_device()
        # On macOS Apple Silicon, MLX GPU is Metal
        # device.type is an enum, not a string
        is_gpu = str(device.type) == "DeviceType.gpu"
        is_apple_silicon = platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"
        if not is_gpu or not is_apple_silicon:
            return {"available": False, "reason": f"Device is not Apple Metal GPU: {device}, type={device.type}, system={platform.system()}, machine={platform.machine()}"}
    except Exception as exc:
        return {"available": False, "reason": f"Cannot query device: {exc}"}

    # Try a minimal compile + dispatch
    try:
        test_source = """
            uint idx = thread_position_in_grid.x;
            if (idx >= 4) return;
            out[idx] = inp[idx] * 2.0;
        """
        test_kernel = mx.fast.metal_kernel(
            name="probe",
            input_names=["inp"],
            output_names=["out"],
            source=test_source,
        )
        test_input = mx.array([1.0, 2.0, 3.0, 4.0])
        test_outputs = test_kernel(
            inputs=[test_input],
            grid=(4, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(4,)],
            output_dtypes=[mx.float32],
        )
        mx.eval(test_outputs[0])
        expected = mx.array([2.0, 4.0, 6.0, 8.0])
        if not mx.allclose(test_outputs[0], expected):
            return {"available": False, "reason": "Metal probe kernel output mismatch"}
    except Exception as exc:
        return {"available": False, "reason": f"Metal probe failed: {exc}"}

    return {"available": True, "reason": None}


# Global capability cache
_METAL_CAPABILITY: dict[str, Any] | None = None


def metal_available() -> bool:
    """Return True if Metal kernels are confirmed to work on this device."""
    global _METAL_CAPABILITY
    if _METAL_CAPABILITY is None:
        _METAL_CAPABILITY = _probe_metal_capability()
    return _METAL_CAPABILITY["available"]


def metal_dense_attention_over_reconstructed_kv(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    causal: bool = True,
    query_start_pos: int = 0,
) -> mx.array:
    """Compute attention using a custom Metal kernel over dense K/V tensors.

    WARNING: This is NOT packed attention. The K/V inputs must already be
    dense float32 tensors. See module docstring for details.

    Args:
        queries: [B, Hq, Lq, D] float32
        keys: [B, Hkv, Lkv, D] float32 (already decoded from packed)
        values: [B, Hkv, Lkv, D] float32 (already decoded from packed)
        scale: Attention scale factor
        causal: Whether to apply causal mask
        query_start_pos: Global position of the first query token in the sequence.

    Returns:
        output: [B, Hq, Lq, D] float32
    """
    if not HAS_MLX:
        raise RuntimeError("MLX is required")

    B, Hq, Lq, D = queries.shape
    _, Hkv, Lkv, _ = keys.shape

    assert B == 1, "Only batch_size=1 is supported"
    assert Hq % Hkv == 0, "GQA ratio must be integer"

    q_per_kv = Hq // Hkv

    # Flatten batch dimension (assume B=1)
    queries_flat = queries.reshape(Hq, Lq, D).astype(mx.float32)
    keys_flat = keys.reshape(Hkv, Lkv, D).astype(mx.float32)
    values_flat = values.reshape(Hkv, Lkv, D).astype(mx.float32)

    # Pass scale and query_start_pos as 1-element arrays (template doesn't support float)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    query_start_arr = mx.array([int(query_start_pos)], dtype=mx.int32)

    kernel = mx.fast.metal_kernel(
        name="dense_attention",
        input_names=["queries", "keys", "values", "scale_arr", "query_start_arr"],
        output_names=["output"],
        source=_ATTENTION_KERNEL,
    )

    outputs = kernel(
        inputs=[queries_flat, keys_flat, values_flat, scale_arr, query_start_arr],
        template=[
            ("NUM_Q_HEADS", int(Hq)),
            ("NUM_Q_TOKENS", int(Lq)),
            ("NUM_KV_TOKENS", int(Lkv)),
            ("HEAD_DIM", int(D)),
            ("Q_PER_KV", int(q_per_kv)),
            ("CAUSAL", int(1 if causal else 0)),
        ],
        grid=(Hq, Lq, 1),
        threadgroup=(16, 16, 1),
        output_shapes=[(Hq, Lq, D)],
        output_dtypes=[mx.float32],
    )

    return outputs[0].reshape(B, Hq, Lq, D)


def _decode_blocks_for_metal(
    layer_cache: Any,
) -> tuple[mx.array | None, mx.array | None]:
    """Decode all blocks, staging, and dense residual into dense K/V tensors.

    WARNING: This performs full-history materialization. All compressed blocks
    are decoded to dense tensors and concatenated. This is expensive and defeats
    the memory savings of packed storage.

    Returns:
        (keys, values) as BHTD float32 arrays, or (None, None) if empty.
    """
    if not HAS_MLX:
        return None, None

    key_blocks = list(layer_cache.iter_key_blocks())
    value_blocks = list(layer_cache.iter_value_blocks())

    decoded_keys: list[Any] = []
    decoded_values: list[Any] = []

    # Decode sealed blocks
    for kb, vb in zip(key_blocks, value_blocks):
        k_decoded = layer_cache.key_codec.decode_bhtd(kb)
        v_decoded = layer_cache.value_codec.decode_bhtd(vb)
        decoded_keys.append(k_decoded)
        decoded_values.append(v_decoded)

    # Add staging if present
    stage_k, stage_v, stage_n = layer_cache.get_staging()
    if stage_n > 0 and stage_k is not None:
        decoded_keys.append(stage_k)
        decoded_values.append(stage_v)

    # Add dense residual if present
    dense_k, dense_v = layer_cache.get_dense_residual()
    if dense_k is not None:
        decoded_keys.append(dense_k)
        decoded_values.append(dense_v)

    if not decoded_keys:
        return None, None

    # Concatenate along token axis (axis=2 for BHTD)
    all_keys = mx.concatenate(decoded_keys, axis=2).astype(mx.float32)
    all_values = mx.concatenate(decoded_values, axis=2).astype(mx.float32)

    return all_keys, all_values


def attend_metal(
    queries: mx.array,
    layer_cache: Any,
    *,
    scale: float | None = None,
    mask: Any | None = None,
    query_start_pos: int | None = None,
    causal: bool = False,
    strict: bool = False,
) -> tuple[mx.array, Any]:
    """Metal-accelerated dense attention over reconstructed KV (NOT packed).

    This function decodes all compressed blocks to dense tensors and runs
    attention via a custom Metal kernel. It is NOT a true packed attention
    implementation. See module docstring for details.

    Args:
        queries: [B, Hq, Lq, D] float32
        layer_cache: QuantizedLayerCache with packed blocks
        scale: Attention scale factor
        mask: Attention mask (not used by Metal path)
        query_start_pos: Global position of first query token
        causal: Whether to apply causal mask
        strict: If True, raise on any failure instead of falling back

    Returns:
        (output, scratch) tuple matching the reference interface.

    Raises:
        StrictPackedExecutionError: If strict=True and Metal execution fails.
    """
    from rfsn_v10.cache.contracts import AttentionScratch
    from rfsn_v10.cache.mlx_packed_attention_reference import attend

    if not HAS_MLX or not metal_available():
        if strict:
            raise StrictPackedExecutionError(
                "Metal not available and strict mode prevents fallback"
            )
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)

    B, Hq, Lq, D = queries.shape
    s = scale if scale is not None else (D ** -0.5)

    # Decode all cache contents into dense tensors
    # WARNING: This is full-history materialization
    all_keys, all_values = _decode_blocks_for_metal(layer_cache)

    if all_keys is None or all_values is None:
        # Empty cache — fall back to reference
        if strict:
            raise StrictPackedExecutionError(
                "Empty cache in strict mode prevents fallback"
            )
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)

    # Record full-history materialization
    key_blocks = list(layer_cache.iter_key_blocks())
    value_blocks = list(layer_cache.iter_value_blocks())
    if hasattr(layer_cache, "session") and layer_cache.session is not None:
        layer_cache.session.runtime_counters.record_full_history_materialization()
        # Also record block reads since we decoded them
        from rfsn_v10.cache.contracts import _array_itemsize
        layer_cache.session.runtime_counters.record_block_read(len(key_blocks))
        for kb, vb in zip(key_blocks, value_blocks):
            if kb.packed_codes is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(kb.packed_codes.size) * _array_itemsize(kb.packed_codes))
            if vb.packed_codes is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(vb.packed_codes.size) * _array_itemsize(vb.packed_codes))
            if kb.scales is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(kb.scales.size) * _array_itemsize(kb.scales))
            if vb.scales is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(vb.scales.size) * _array_itemsize(vb.scales))

    # If a string mask was passed (e.g. "causal"), override the causal arg
    if isinstance(mask, str) and mask.lower() == "causal":
        causal = True

    try:
        # Determine query_start_pos if not provided
        q_start = query_start_pos if query_start_pos is not None else (all_keys.shape[2] - queries.shape[2])

        # Call Metal kernel over dense reconstructed KV
        output = metal_dense_attention_over_reconstructed_kv(
            queries, all_keys, all_values, s, causal=causal, query_start_pos=q_start
        )

        # Track scratch memory from decoded dense tensors
        scratch = AttentionScratch(
            max_reconstructed_block_tokens=int(all_keys.shape[2]),
        )

        return output, scratch

    except Exception as exc:
        if strict:
            raise StrictPackedExecutionError(
                f"Metal kernel failed in strict mode: {exc}"
            ) from exc
        # Fallback to reference on any error
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)


def benchmark_metal_vs_reference(
    B: int = 1,
    Hq: int = 8,
    Lq: int = 1,
    D: int = 64,
    Hkv: int = 2,
    Lkv: int = 128,
    num_runs: int = 10,
) -> dict[str, float]:
    """Benchmark Metal kernel against MLX reference implementation.

    Returns:
        Dictionary with timing results in ms.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX is required for benchmarking")

    queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
    keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    scale = D ** -0.5

    # Warmup
    for _ in range(3):
        _ = metal_dense_attention_over_reconstructed_kv(
            queries, keys, values, scale, causal=True, query_start_pos=Lkv - Lq
        )
        mx.eval(_)

    # Benchmark Metal
    metal_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = metal_dense_attention_over_reconstructed_kv(
            queries, keys, values, scale, causal=True, query_start_pos=Lkv - Lq
        )
        mx.eval(out)
        t1 = time.perf_counter()
        metal_times.append((t1 - t0) * 1000)

    # Benchmark reference (matmul-based)
    ref_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        # Simple causal attention using MLX ops with GQA repeat
        keys_rep = mx.repeat(keys, Hq // Hkv, axis=1)
        values_rep = mx.repeat(values, Hq // Hkv, axis=1)
        scores = mx.matmul(queries, keys_rep.transpose(0, 1, 3, 2)) * scale
        q_pos = mx.arange(Lq)[:, None]
        kv_pos = mx.arange(Lkv)[None, :]
        causal_mask = q_pos >= kv_pos
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (B, Hq, Lq, Lkv))
        scores = mx.where(causal_mask, scores, mx.array(-float("inf")))
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, values_rep)
        mx.eval(out)
        t1 = time.perf_counter()
        ref_times.append((t1 - t0) * 1000)

    return {
        "metal_mean_ms": sum(metal_times) / len(metal_times),
        "metal_min_ms": min(metal_times),
        "metal_max_ms": max(metal_times),
        "ref_mean_ms": sum(ref_times) / len(ref_times),
        "ref_min_ms": min(ref_times),
        "ref_max_ms": max(ref_times),
        "speedup": sum(ref_times) / sum(metal_times),
    }
