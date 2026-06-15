"""Metal kernel implementation for fused packed attention using mx.fast.metal_kernel.

This module provides a GPU-accelerated implementation of packed attention
that uses MLX's custom Metal kernel support.
"""
from __future__ import annotations

import math
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

# Decode 8-bit packed values from uint32 array
# Each uint32 contains 4 packed 8-bit values
_DECODE_U8_KERNEL = """
uint idx = thread_position_in_grid.x;
if (idx >= N) return;

// Load uint32 and extract byte
uint packed_val = packed_codes[idx / 4];
uint byte_idx = idx % 4;
uint8_t byte = (packed_val >> (byte_idx * 8)) & 0xFF;

// Decode: signed integer [-127, 127] -> float [-1.0, 1.0]
float val = (float(byte) - 127.5) / 127.5;
out[idx] = val * scale;
"""

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

// Online softmax state
float running_max = -INFINITY;
float running_sum = 0.0;

// First pass: compute QK scores and online softmax
for (uint kv_t = 0; kv_t < NUM_KV_TOKENS; kv_t++) {
    // Causal mask
    if (CAUSAL != 0 && kv_t > qt) continue;
    
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
    if (CAUSAL != 0 && kv_t > qt) continue;
    
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


def _decode_packed_u8(packed_codes: mx.array, scales: mx.array, shape: tuple) -> mx.array:
    """Decode uint32 packed 8-bit codes to float array.
    
    Args:
        packed_codes: uint32 array where each element contains 4 packed bytes
        scales: float32 scales for dequantization [n_groups]
        shape: Target shape (n_heads, n_tokens, head_dim)
    
    Returns:
        Decoded float32 array of shape
    """
    if not HAS_MLX:
        raise RuntimeError("MLX is required")
    
    # Flatten packed codes to individual bytes
    n_elements = math.prod(shape)
    
    kernel = mx.fast.metal_kernel(
        name="decode_u8",
        input_names=["packed_codes", "scale"],
        output_names=["out"],
        source=_DECODE_U8_KERNEL,
    )
    
    outputs = kernel(
        inputs=[packed_codes, scales],
        grid=(n_elements, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_elements,)],
        output_dtypes=[mx.float32],
    )
    
    return outputs[0].reshape(shape)


def metal_packed_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    causal: bool = True,
) -> mx.array:
    """Compute attention using Metal kernel.
    
    Args:
        queries: [B, Hq, Lq, D] float32
        keys: [B, Hkv, Lkv, D] float32 (decoded from packed)
        values: [B, Hkv, Lkv, D] float32 (decoded from packed)
        scale: Attention scale factor
        causal: Whether to apply causal mask
    
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
    
    # Pass scale as 1-element float array (template doesn't support float)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    
    kernel = mx.fast.metal_kernel(
        name="packed_attention",
        input_names=["queries", "keys", "values", "scale_arr"],
        output_names=["output"],
        source=_ATTENTION_KERNEL,
    )
    
    outputs = kernel(
        inputs=[queries_flat, keys_flat, values_flat, scale_arr],
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
    
    Current limitations:
    - Only supports 8-bit quantization (K8/V8)
    - Does not apply WHT/signs (assumes pre-decoded or simplified codec)
    - Batch size must be 1
    """
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.contracts import AttentionScratch
    
    if not HAS_MLX:
        # Fallback to reference implementation
        return attend(queries, layer_cache, scale=scale, mask=mask, 
                     query_start_pos=query_start_pos, causal=causal)
    
    B, Hq, Lq, D = queries.shape
    s = scale if scale is not None else (D ** -0.5)
    
    # Get blocks from layer cache
    key_blocks = list(layer_cache.iter_key_blocks())
    value_blocks = list(layer_cache.iter_value_blocks())
    
    if not key_blocks:
        # No sealed blocks - use reference implementation
        return attend(queries, layer_cache, scale=scale, mask=mask,
                     query_start_pos=query_start_pos, causal=causal)
    
    # For now, always use reference implementation since the decode path
    # requires complex block metadata (layer_id, stream_id, codec_signature)
    # that may not be present in all blocks.
    # TODO: Implement proper block decode for Metal path
    return attend(queries, layer_cache, scale=scale, mask=mask,
                 query_start_pos=query_start_pos, causal=causal)
