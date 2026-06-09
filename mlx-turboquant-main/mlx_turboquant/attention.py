"""
TurboQuant attention: compute attention with compressed KV cache.
"""

import mlx.core as mx

from .turbo_quant import TurboQuantCompressor, CompressedKeys, CompressedValues


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    compressed_keys: CompressedKeys,
    compressed_values: CompressedValues,
    compressor: TurboQuantCompressor,
    scale: float,
    mask=None,
) -> mx.array:
    """Compute attention with TurboQuant-compressed KV cache.

    Args:
        queries: (B, n_q_heads, L_q, D)
        compressed_keys: CompressedKeys from cache
        compressed_values: CompressedValues from cache
        compressor: TurboQuantCompressor instance
        scale: 1/sqrt(D)
        mask: Optional attention mask

    Returns:
        output: (B, n_q_heads, L_q, D)
    """
    B, n_q_heads, L_q, D = queries.shape
    n_kv_heads = compressed_keys.indices.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    # Handle GQA: group queries by KV head
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L_q, D))

    # Compute attention scores with QJL correction
    # Need to handle the GQA reshape for score computation
    if n_repeats > 1:
        # Expand compressed keys for grouped computation
        scores = _gqa_attention_scores(
            queries, compressed_keys, compressor, scale, n_repeats
        )
    else:
        scores = compressor.attention_scores(queries, compressed_keys, scale)

    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L_kv = compressed_keys.indices.shape[2]
            q_indices = mx.arange(L_kv - L_q, L_kv)
            k_indices = mx.arange(L_kv)
            mask = q_indices[:, None] >= k_indices[None]
        if hasattr(mask, 'dtype'):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores += mask

    # Softmax
    scores = mx.softmax(scores, axis=-1, precise=True)

    # Reconstruct values and compute weighted sum
    values = compressor.reconstruct_values(compressed_values)

    if n_repeats > 1:
        values = mx.expand_dims(values, axis=2)  # (B, n_kv_heads, 1, L_kv, D)
        # scores: (B, n_kv_heads, n_repeats, L_q, L_kv)
        # values: (B, n_kv_heads, 1, L_kv, D)
        out = scores @ values  # (B, n_kv_heads, n_repeats, L_q, D)
        out = mx.reshape(out, (B, n_q_heads, L_q, D))
    else:
        out = scores @ values  # (B, n_heads, L_q, D)

    return out


def _gqa_attention_scores(
    queries: mx.array,
    compressed_keys: CompressedKeys,
    compressor: TurboQuantCompressor,
    scale: float,
    n_repeats: int,
) -> mx.array:
    """Compute attention scores with GQA grouping.

    queries: (B, n_kv_heads, n_repeats, L_q, D)
    Returns: (B, n_kv_heads, n_repeats, L_q, L_kv)
    """
    B, n_kv_heads, n_rep, L_q, D = queries.shape

    # Reconstruct keys
    reconstructed_keys = compressor.key_pq.dequantize(
        compressed_keys.indices, compressed_keys.norms
    )

    queries_scaled = queries * scale

    # Base scores: (B, n_kv_heads, n_repeats, L_q, D) @ (B, n_kv_heads, D, L_kv)
    keys_t = mx.swapaxes(reconstructed_keys, -2, -1)
    keys_t = mx.expand_dims(keys_t, axis=2)  # (B, n_kv_heads, 1, D, L_kv)
    base_scores = queries_scaled @ keys_t  # broadcasts over n_repeats

    if not compressor.use_qjl:
        return base_scores

    # QJL correction for GQA
    qjl = compressor.qjl

    # Project queries
    projected_queries = queries_scaled @ qjl.projection.T  # (B, kv, rep, L_q, proj_dim)

    # Convert signs
    sign_values = mx.where(compressed_keys.signs, 1.0, -1.0)
    sign_t = mx.swapaxes(sign_values, -2, -1)
    sign_t = mx.expand_dims(sign_t, axis=2)  # (B, kv, 1, proj_dim, L_kv)

    correction = projected_queries @ sign_t  # (B, kv, rep, L_q, L_kv)

    residual_norms_t = mx.swapaxes(compressed_keys.residual_norms, -2, -1)
    residual_norms_t = mx.expand_dims(residual_norms_t, axis=2)
    correction = correction * residual_norms_t * qjl.correction_scale

    return base_scores + correction
