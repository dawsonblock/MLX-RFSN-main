"""MLX reference packed-attention engine.

Consolidates the blockwise attention path into one function:
  attend(queries, layer_cache, scale, mask, query_start_pos)

This is the canonical MLX reference that the NumPy oracle must match,
and that Metal kernels must eventually match.
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.compat import mx

from .contracts import AttentionScratch
from .incremental_layer_cache import QuantizedLayerCache


def attend(
    queries: Any,
    layer_cache: QuantizedLayerCache,
    *,
    scale: float | None = None,
    mask: Any | None = None,
    query_start_pos: int | None = None,
    causal: bool = False,
) -> tuple[Any, AttentionScratch]:
    """Compute attention output directly from quantized blocks (MLX reference).

    Parameters
    ----------
    queries
        Shape ``(B, Hq, Lq, D)``.
    layer_cache
        The per-layer quantized cache.
    scale
        Attention scale.  Defaults to ``D ** -0.5``.
    mask
        Optional additive mask.
    query_start_pos
        Global sequence position of the first query token.
        Defaults to ``layer_cache.total_token_count()``.
    causal
        If True, apply a causal mask when *mask* is None.

    Returns
    -------
    output
        Shape ``(B, Hq, Lq, D)``.
    scratch
        Scratch-memory accounting.
    """
    B, Hq, Lq, D = queries.shape
    s = scale if scale is not None else (D ** -0.5)

    # Infer KV heads from cache blocks
    key_blocks = list(layer_cache.iter_key_blocks())
    value_blocks = list(layer_cache.iter_value_blocks())

    if key_blocks:
        n_kv_heads = key_blocks[0].n_kv_heads
    else:
        stage_k, _, _ = layer_cache.get_staging()
        if stage_k is not None:
            n_kv_heads = stage_k.shape[1]
        else:
            dense_k, _ = layer_cache.get_dense_residual()
            if dense_k is not None:
                n_kv_heads = dense_k.shape[1]
            else:
                raise ValueError("cache is empty")

    if Hq % n_kv_heads != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({n_kv_heads})")

    repeats = Hq // n_kv_heads

    if query_start_pos is None:
        query_start_pos = layer_cache.total_token_count()

    max_block_tokens = 0
    position_offset = 0

    # Online softmax state
    # Initialise to -1e9 (finite) so that exp(running_max - new_max) never
    # hits the NaN-producing -inf - (-inf) case, even for fully-masked blocks.
    running_max = mx.full((B, Hq, Lq, 1), -1e9, dtype=mx.float32)
    running_sum = mx.zeros((B, Hq, Lq, 1), dtype=mx.float32)
    out = mx.zeros((B, Hq, Lq, D), dtype=mx.float32)

    def _process_region(k_bhtd: Any, v_bhtd: Any, region_tokens: int) -> None:
        nonlocal running_max, running_sum, out, position_offset, max_block_tokens
        max_block_tokens = max(max_block_tokens, region_tokens)

        # GQA repeat if needed
        if k_bhtd.shape[1] != Hq:
            k_bhtd = mx.repeat(k_bhtd, repeats, axis=1)
            v_bhtd = mx.repeat(v_bhtd, repeats, axis=1)

        # Scores
        scores = mx.matmul(
            queries.astype(mx.float32),
            k_bhtd.astype(mx.float32).transpose(0, 1, 3, 2),
        ) * s

        # Mask
        if mask is not None and not isinstance(mask, str):
            scores = scores + mask[..., position_offset:position_offset + region_tokens]
        elif causal or (isinstance(mask, str) and mask.lower() == "causal"):
            # Causal mask: query at global position q can attend to kv at position kv if q >= kv
            q_positions = mx.arange(query_start_pos, query_start_pos + Lq)[:, None]
            kv_positions = mx.arange(position_offset, position_offset + region_tokens)[None, :]
            causal_mask = (q_positions >= kv_positions).astype(mx.float32)
            causal_mask = mx.broadcast_to(
                causal_mask[None, None, :, :], (B, Hq, Lq, region_tokens)
            )
            scores = mx.where(causal_mask, scores, mx.array(-mx.inf, dtype=scores.dtype))
        elif isinstance(mask, str):
            raise ValueError(f"unrecognized mask string: {mask!r}")

        # Online softmax with direct exp(scores - new_max) to avoid NaN
        # when block_max is -inf (fully-masked block).
        block_max = mx.max(scores, axis=-1, keepdims=True)
        new_max = mx.maximum(running_max, block_max)
        old_scale = mx.exp(running_max - new_max)

        running_sum = running_sum * old_scale
        out = out * old_scale

        block_exp = mx.exp(scores.astype(mx.float32) - new_max)
        running_sum = running_sum + mx.sum(block_exp, axis=-1, keepdims=True)
        out = out + mx.matmul(block_exp, v_bhtd.astype(mx.float32))

        running_max = new_max
        position_offset += region_tokens

    # Sealed blocks (use decode_bhtd directly)
    for kb, vb in zip(key_blocks, value_blocks):
        k_dense = layer_cache.key_codec.decode_bhtd(kb)
        v_dense = layer_cache.value_codec.decode_bhtd(vb)
        _process_region(k_dense, v_dense, kb.token_count)

    # Staging
    stage_k, stage_v, stage_n = layer_cache.get_staging()
    if stage_k is not None:
        _process_region(stage_k, stage_v, stage_n)

    # Dense residual
    dense_k, dense_v = layer_cache.get_dense_residual()
    if dense_k is not None:
        _process_region(dense_k, dense_v, dense_k.shape[2])

    # Guard against fully-masked rows where running_sum == 0
    output = mx.where(running_sum == 0, mx.zeros_like(out), out / running_sum)
    output = output.astype(queries.dtype)

    scratch = AttentionScratch(
        max_reconstructed_block_tokens=max_block_tokens,
        score_vector_bytes=0,
        output_accumulator_bytes=int(out.size) * 4,
    )
    return output, scratch
