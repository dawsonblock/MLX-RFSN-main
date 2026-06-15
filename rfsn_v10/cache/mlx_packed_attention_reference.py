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

    # Online softmax state with explicit valid-mass tracking.
    # Initialising running_max to -inf and tracking has_mass avoids the
    # NaN-producing -inf - (-inf) case on fully-masked blocks.
    running_max = mx.full((B, Hq, Lq, 1), -mx.inf, dtype=mx.float32)
    running_sum = mx.zeros((B, Hq, Lq, 1), dtype=mx.float32)
    out = mx.zeros((B, Hq, Lq, D), dtype=mx.float32)
    has_mass = mx.zeros((B, Hq, Lq, 1), dtype=mx.bool_)

    # Fix #6: Track scratch memory for initial allocations
    if hasattr(layer_cache, "session") and layer_cache.session is not None:
        if hasattr(out, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_allocation(out.nbytes)
        if hasattr(running_max, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_allocation(running_max.nbytes)
        if hasattr(running_sum, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_allocation(running_sum.nbytes)
        if hasattr(has_mass, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_allocation(has_mass.nbytes)

    def _process_region(k_bhtd: Any, v_bhtd: Any, region_tokens: int) -> None:
        nonlocal running_max, running_sum, out, has_mass, position_offset, max_block_tokens
        max_block_tokens = max(max_block_tokens, region_tokens)

        # GQA repeat if needed
        if k_bhtd.shape[1] != Hq:
            k_bhtd = mx.repeat(k_bhtd, repeats, axis=1)
            v_bhtd = mx.repeat(v_bhtd, repeats, axis=1)
            # Fix #6: Track scratch memory for GQA expansion
            if hasattr(layer_cache, "session") and layer_cache.session is not None:
                if hasattr(k_bhtd, 'nbytes'):
                    layer_cache.session.runtime_counters.record_scratch_allocation(k_bhtd.nbytes)
                if hasattr(v_bhtd, 'nbytes'):
                    layer_cache.session.runtime_counters.record_scratch_allocation(v_bhtd.nbytes)

        # Scores
        scores = mx.matmul(
            queries.astype(mx.float32),
            k_bhtd.astype(mx.float32).transpose(0, 1, 3, 2),
        ) * s
        # Fix #6: Track scratch memory for scores
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(scores, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(scores.nbytes)

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

        # Online softmax — explicit valid-mass tracking for fully-masked blocks.
        block_max = mx.max(scores, axis=-1, keepdims=True)
        new_max = mx.maximum(running_max, block_max)
        old_scale = mx.exp(running_max - new_max)
        # Guard exp(-inf - (-inf)) => NaN on rows with no valid mass yet.
        old_scale = mx.where(mx.isfinite(old_scale), old_scale, 1.0)
        # Fix #6: Track scratch memory for softmax intermediates
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(block_max, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(block_max.nbytes)
            if hasattr(new_max, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(new_max.nbytes)
            if hasattr(old_scale, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(old_scale.nbytes)

        running_sum = running_sum * old_scale
        out = out * old_scale

        block_exp = mx.exp(scores.astype(mx.float32) - new_max)
        # Zero out exp(NaN) from masked positions when new_max is also -inf.
        block_exp = mx.where(mx.isfinite(block_exp), block_exp, 0.0)
        # Fix #6: Track scratch memory for block_exp
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(block_exp, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(block_exp.nbytes)

        # Fix #6: Free scratch memory for scores and intermediates
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(scores, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(scores.nbytes)
            if hasattr(block_max, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(block_max.nbytes)
            if hasattr(new_max, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(new_max.nbytes)
            if hasattr(old_scale, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(old_scale.nbytes)
            if hasattr(block_exp, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(block_exp.nbytes)
        running_sum = running_sum + mx.sum(block_exp, axis=-1, keepdims=True)
        out = out + mx.matmul(block_exp, v_bhtd.astype(mx.float32))

        running_max = new_max
        has_mass = mx.logical_or(has_mass, mx.any(mx.isfinite(scores), axis=-1, keepdims=True))
        position_offset += region_tokens

    # Sealed blocks (use decode_bhtd directly)
    # Fix #2: Use typed methods instead of string-based increment
    # Fix #4: Record actual block creation, block reads, packed calls and bytes
    if hasattr(layer_cache, "session") and layer_cache.session is not None:
        layer_cache.session.runtime_counters.record_block_read(len(key_blocks))
        # Track bytes read for keys and values including scales
        from rfsn_v10.cache.contracts import _array_itemsize
        for kb, vb in zip(key_blocks, value_blocks):
            # Code arrays
            if kb.packed_codes is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(kb.packed_codes.size) * _array_itemsize(kb.packed_codes))
            if vb.packed_codes is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(vb.packed_codes.size) * _array_itemsize(vb.packed_codes))
            # Scale arrays
            if kb.scales is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(kb.scales.size) * _array_itemsize(kb.scales))
            if vb.scales is not None:
                layer_cache.session.runtime_counters.record_packed_read(int(vb.scales.size) * _array_itemsize(vb.scales))

    for kb, vb in zip(key_blocks, value_blocks):
        k_dense = layer_cache.key_codec.decode_bhtd(kb)
        v_dense = layer_cache.value_codec.decode_bhtd(vb)
        # Fix #4: Track decoded block bytes using actual tensor sizes
        # Fix #6: Track scratch memory for decoded tensors
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(k_dense, 'nbytes'):
                layer_cache.session.runtime_counters.record_decoded_block(k_dense.nbytes)
                layer_cache.session.runtime_counters.record_scratch_allocation(k_dense.nbytes)
            if hasattr(v_dense, 'nbytes'):
                layer_cache.session.runtime_counters.record_decoded_block(v_dense.nbytes)
                layer_cache.session.runtime_counters.record_scratch_allocation(v_dense.nbytes)
        _process_region(k_dense, v_dense, kb.token_count)
        # Fix #6: Free scratch memory after processing
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(k_dense, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(k_dense.nbytes)
            if hasattr(v_dense, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(v_dense.nbytes)

    # Staging
    stage_k, stage_v, stage_n = layer_cache.get_staging()
    if stage_k is not None:
        # Fix #6: Track scratch memory for staging
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(stage_k, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(stage_k.nbytes)
            if hasattr(stage_v, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(stage_v.nbytes)
        _process_region(stage_k, stage_v, stage_n)
        # Fix #6: Free scratch memory after processing
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(stage_k, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(stage_k.nbytes)
            if hasattr(stage_v, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(stage_v.nbytes)

    # Dense residual
    dense_k, dense_v = layer_cache.get_dense_residual()
    if dense_k is not None:
        # Fix #6: Track scratch memory for dense residual
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(dense_k, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(dense_k.nbytes)
            if hasattr(dense_v, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_allocation(dense_v.nbytes)
        _process_region(dense_k, dense_v, dense_k.shape[2])
        # Fix #6: Free scratch memory after processing
        if hasattr(layer_cache, "session") and layer_cache.session is not None:
            if hasattr(dense_k, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(dense_k.nbytes)
            if hasattr(dense_v, 'nbytes'):
                layer_cache.session.runtime_counters.record_scratch_free(dense_v.nbytes)

    # Fully-masked rows return defined zero output.
    # Guard against running_sum == 0 due to numerical underflow.
    output = mx.where(
        has_mass & (running_sum > 0),
        out / running_sum,
        mx.zeros_like(out)
    )
    output = output.astype(queries.dtype)

    # Fix #6: Free initial scratch allocations
    if hasattr(layer_cache, "session") and layer_cache.session is not None:
        if hasattr(out, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_free(out.nbytes)
        if hasattr(running_max, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_free(running_max.nbytes)
        if hasattr(running_sum, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_free(running_sum.nbytes)
        if hasattr(has_mass, 'nbytes'):
            layer_cache.session.runtime_counters.record_scratch_free(has_mass.nbytes)

    from rfsn_v10.cache.contracts import _array_itemsize
    scratch = AttentionScratch(
        max_reconstructed_block_tokens=max_block_tokens,
        score_vector_bytes=0,
        output_accumulator_bytes=int(out.size) * _array_itemsize(out),
    )

    # P0 #5: Instrument runtime counters - track scratch bytes
    if hasattr(layer_cache, "session") and layer_cache.session is not None:
        layer_cache.session.runtime_counters.scratch_bytes_current = scratch.output_accumulator_bytes
        if scratch.output_accumulator_bytes > layer_cache.session.runtime_counters.scratch_bytes_peak:
            layer_cache.session.runtime_counters.scratch_bytes_peak = scratch.output_accumulator_bytes

    return output, scratch
