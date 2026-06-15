"""Metal kernel implementation for fused packed attention using mx.fast.metal_kernel.

This module provides a GPU-accelerated implementation of packed attention
that uses MLX's custom Metal kernel support.
"""
from __future__ import annotations

import math
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


def metal_packed_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    causal: bool = True,
    query_start_pos: int = 0,
) -> mx.array:
    """Compute attention using a custom Metal kernel.

    Args:
        queries: [B, Hq, Lq, D] float32
        keys: [B, Hkv, Lkv, D] float32 (decoded from packed)
        values: [B, Hkv, Lkv, D] float32 (decoded from packed)
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
        name="packed_attention",
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
) -> tuple[mx.array, Any]:
    """Metal-accelerated packed attention (drop-in replacement for attend).

    This function mirrors the interface of mlx_packed_attention_reference.attend()
    but uses a custom Metal kernel for GPU acceleration.

    Returns:
        (output, scratch) tuple matching the reference interface.
    """
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.contracts import AttentionScratch

    if not HAS_MLX:
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)

    B, Hq, Lq, D = queries.shape
    s = scale if scale is not None else (D ** -0.5)

    # Decode all cache contents into dense tensors
    all_keys, all_values = _decode_blocks_for_metal(layer_cache)

    if all_keys is None or all_values is None:
        # Empty cache — fall back to reference
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)

    # If a string mask was passed (e.g. "causal"), override the causal arg
    if isinstance(mask, str) and mask.lower() == "causal":
        causal = True

    try:
        # Determine query_start_pos if not provided
        q_start = query_start_pos if query_start_pos is not None else (all_keys.shape[2] - queries.shape[2])

        # Call Metal kernel
        output = metal_packed_attention(
            queries, all_keys, all_values, s, causal=causal, query_start_pos=q_start
        )

        # Record proof counters for compressed-execution gate
        key_blocks = list(layer_cache.iter_key_blocks())
        value_blocks = list(layer_cache.iter_value_blocks())
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            layer_cache.session.runtime_counters.record_block_read(len(key_blocks))
            for kb, vb in zip(key_blocks, value_blocks):
                if kb.packed_codes is not None:
                    layer_cache.session.runtime_counters.record_packed_read(int(kb.packed_codes.size) * 4)
                if vb.packed_codes is not None:
                    layer_cache.session.runtime_counters.record_packed_read(int(vb.packed_codes.size) * 4)
                if kb.scales is not None:
                    layer_cache.session.runtime_counters.record_packed_read(int(kb.scales.size) * 4)
                if vb.scales is not None:
                    layer_cache.session.runtime_counters.record_packed_read(int(vb.scales.size) * 4)

        # Track scratch memory from decoded dense tensors
        scratch = AttentionScratch(
            max_reconstructed_block_tokens=int(all_keys.shape[2]),
        )

        return output, scratch

    except Exception as exc:
        # Fallback to reference on any error
        import warnings
        warnings.warn(f"Metal kernel failed, falling back to reference: {exc}")
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
        _ = metal_packed_attention(queries, keys, values, scale, causal=True, query_start_pos=Lkv - Lq)
        mx.eval(_)

    # Benchmark Metal
    metal_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = metal_packed_attention(queries, keys, values, scale, causal=True, query_start_pos=Lkv - Lq)
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
        scores = mx.where(causal_mask, scores, mx.array(-float('inf')))
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
