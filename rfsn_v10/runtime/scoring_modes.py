"""RFSN v10 — Prepared vs packed scoring modes.

Provides five quantization-aware attention scoring paths:

* ``fp16``               — baseline without quantization.
* ``reconstructed``      — fully unpack / dequant the entire KV cache
                           before scoring.
* ``prepared``           — reuse pre-computed dequantized (or transformed)
                           cache blocks.
* ``packed_block``       — unpack / dequant only the selected blocks.
* ``score_corrected``    — future QJL or residual score correction (stub).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    from rfsn_v10.compat import mx

from rfsn_v10.attention_reference import causal_attention_dense


def score_attention_fp16(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float | None = None,
) -> mx.array:
    """Standard FP16 baseline attention (no quantization).

    Routes through :func:`rfsn_v10.attention_reference.causal_attention_dense`
    so that causal masking is always applied for multi-token prefill.

    Args:
        queries: [B, H, T_q, D]
        keys:    [B, H, T_k, D]
        values:  [B, H, T_k, D]
        scale:   Optional manual scale; defaults to ``1 / sqrt(D)``.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(queries.shape[-1])
    return causal_attention_dense(queries, keys, values, scale=scale, backend="mlx")


def score_attention_reconstructed(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    dequant_fn: Callable[[Any, Any], tuple[mx.array, mx.array]],
    scale: float | None = None,
) -> mx.array:
    """Full dequantization then score.

    Args:
        queries:      [B, H, T_q, D]
        keys_packet:  opaque packed key container.
        values_packet: opaque packed value container.
        dequant_fn:   callable that takes ``(keys_packet, values_packet)``
                      and returns a tuple ``(keys, values)`` of shape
                      ``[B, H, T_k, D]``.
        scale:        Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    keys, values = dequant_fn(keys_packet, values_packet)
    return score_attention_fp16(queries, keys, values, scale=scale)


def score_attention_prepared(
    queries: mx.array,
    prepared_k: mx.array,
    prepared_v: mx.array,
    scale: float | None = None,
) -> mx.array:
    """Score using pre-computed dequantized or transformed cache blocks.

    Args:
        queries:    [B, H, T_q, D]
        prepared_k: [B, H, T_k, D]  (already dequantized / transformed)
        prepared_v: [B, H, T_k, D]  (already dequantized / transformed)
        scale:      Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    return score_attention_fp16(queries, prepared_k, prepared_v, scale=scale)


def score_attention_packed_block(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    block_indices: mx.array | list[int],
    block_dequant_fn: Callable[
        [Any, Any, mx.array | list[int]],
        tuple[mx.array, mx.array],
    ],
    scale: float | None = None,
) -> mx.array:
    """Unpack / dequant only the selected blocks needed for attention.

    Args:
        queries:          [B, H, T_q, D]
        keys_packet:      opaque packed key container.
        values_packet:    opaque packed value container.
        block_indices:    1-D array or list of block indices to retrieve.
        block_dequant_fn: callable that takes
                          ``(keys_packet, values_packet, block_indices)`` and
                          returns ``(keys, values)`` of shape
                          ``[B, H, len(block_indices)*Bsz, D]``.
        scale:            Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    keys, values = block_dequant_fn(keys_packet, values_packet, block_indices)
    return score_attention_fp16(queries, keys, values, scale=scale)


def score_attention_quantized_metal(
    queries: mx.array,
    packed_k: mx.array,
    packed_v: mx.array,
    scales_k: mx.array,
    scales_v: mx.array,
    n_keys: int,
    bits: int,
    group_size: int = 64,
    scale: float | None = None,
) -> mx.array:
    """Fused quantized attention via Metal kernel (decode-step only).

    Args:
        queries:   [B, H, 1, D] or [H, D] single query token.
        packed_k:  [H, K_words] uint32 packed key codes.
        packed_v:  [H, V_words] uint32 packed value codes.
        scales_k:  [H, K_groups] float32 key scales.
        scales_v:  [H, V_groups] float32 value scales.
        n_keys:    Number of key positions (T_k).
        bits:      Bit width (2-8).
        group_size: Quant group size.
        scale:     Attention scale.

    Returns:
        [B, H, 1, D] attention output.
    """
    from ..kernels import quantized_attention_decode_metal

    # Support [B, H, 1, D] by squeezing / unsqueezing
    orig_shape = queries.shape
    if queries.ndim == 4:
        queries = queries[:, :, 0, :]  # [B, H, D]
        # Flatten batch+heads for the kernel
        bsz, n_h, d_head = queries.shape
        queries = queries.reshape(bsz * n_h, d_head)
    elif queries.ndim == 3:
        queries = queries[:, 0, :]  # [B, D]
        bsz, d_head = queries.shape
        queries = queries.reshape(bsz, d_head)
    else:
        bsz = 1

    out = quantized_attention_decode_metal(
        queries=queries,
        packed_k=packed_k,
        packed_v=packed_v,
        scales_k=scales_k,
        scales_v=scales_v,
        n_keys=n_keys,
        bits=bits,
        group_size=group_size,
        scale=scale,
    )

    if len(orig_shape) == 4:
        bsz = orig_shape[0]
        n_h = orig_shape[1]
        d_head = orig_shape[3]
        out = out.reshape(bsz, n_h, 1, d_head)
    elif len(orig_shape) == 3:
        bsz = orig_shape[0]
        d_head = orig_shape[2]
        out = out.reshape(bsz, 1, d_head)
    return out


def score_attention_score_corrected(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    correction_fn: Callable[..., mx.array] | None = None,
    scale: float = 1.0,
    **kwargs: Any,
) -> mx.array:
    """QJL or residual score correction path.

    Args:
        queries:       [B, H, T_q, D]
        keys_packet:   opaque packed key container.
        values_packet: dequantized values of shape ``[B, H, T_k, D]``.
        correction_fn: callable that takes
                       ``(queries, keys_packet, **kwargs)`` and returns
                       corrected scores of shape ``[B, H, T_q, T_k]``.
        scale:         scalar applied to scores before softmax.
        **kwargs:      forwarded to *correction_fn*.

    Raises:
        ValueError: If *correction_fn* is not provided.
        TypeError: If *values_packet* is not an mlx array.
    """
    if correction_fn is None:
        raise ValueError(
            "score_attention_score_corrected requires correction_fn"
        )
    if not isinstance(values_packet, mx.array):
        raise TypeError(
            f"values_packet must be an mx.array, got {type(values_packet).__name__}"
        )
    scores = correction_fn(queries, keys_packet, **kwargs) * scale
    # Stable softmax
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights @ values_packet
