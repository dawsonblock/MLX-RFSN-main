"""NumPy reference oracle for packed blockwise attention.

Implements direct packed attention without reconstructing the full dense K/V
history.  Uses online softmax so the full score vector is never materialised.

This is the executable specification for correct packed attention.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rfsn_v10.cache.contracts import PackedBlockV4, Preconditioner

from .numpy_codec_oracle import NumpyCartesianCodec, _numpy_wht64, _numpy_hash_signs


def _decode_block_bhtd(
    block: PackedBlockV4, codec: NumpyCartesianCodec
) -> np.ndarray:
    """Decode one packed block to dense BHTD (NumPy)."""
    return codec.decode_bhtd(block)


def numpy_packed_attention(
    queries: np.ndarray,
    key_blocks: list[PackedBlockV4],
    value_blocks: list[PackedBlockV4],
    key_codec: NumpyCartesianCodec,
    value_codec: NumpyCartesianCodec,
    *,
    scale: float,
    query_start_pos: int = 0,
    causal: bool = False,
    additive_mask: np.ndarray | None = None,
    stage_k: np.ndarray | None = None,
    stage_v: np.ndarray | None = None,
    dense_k: np.ndarray | None = None,
    dense_v: np.ndarray | None = None,
) -> np.ndarray:
    """Compute attention directly from packed K/V blocks (NumPy reference).

    Parameters
    ----------
    queries
        Shape ``(B, Hq, Lq, D)``.  Already transformed if WHT is used.
    key_blocks, value_blocks
        Lists of ``PackedBlockV4`` sealed blocks.
    key_codec, value_codec
        NumPy codecs for decoding blocks.
    scale
        Attention scale (typically ``head_dim ** -0.5``).
    query_start_pos
        Global sequence position of the first query token.
    causal
        If True, apply a causal mask when *additive_mask* is None.
    additive_mask
        Optional additive mask of shape ``(Lq, total_kv_tokens)``.
    stage_k, stage_v
        Optional staging tensors of shape ``(B, Hkv, Tstage, D)``.
    dense_k, dense_v
        Optional dense residual tensors of shape ``(B, Hkv, Tdense, D)``.

    Returns
    -------
    output
        Attention output of shape ``(B, Hq, Lq, D)``.
    """
    B, Hq, Lq, D = queries.shape

    # GQA validation
    if key_blocks:
        Hkv = key_blocks[0].n_kv_heads
    elif stage_k is not None:
        Hkv = stage_k.shape[1]
    elif dense_k is not None:
        Hkv = dense_k.shape[1]
    else:
        raise ValueError("No K/V blocks, staging, or dense residual provided")

    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({Hkv})")

    repeats = Hq // Hkv

    # Online softmax state
    running_max = np.full((B, Hq, Lq, 1), -np.inf, dtype=np.float32)
    running_sum = np.zeros((B, Hq, Lq, 1), dtype=np.float32)
    out = np.zeros((B, Hq, Lq, D), dtype=np.float32)

    token_offset = 0

    def _process_block(k_block_bhtd: np.ndarray, v_block_bhtd: np.ndarray) -> None:
        nonlocal running_max, running_sum, out, token_offset
        block_T = k_block_bhtd.shape[2]

        # GQA repeat
        if k_block_bhtd.shape[1] != Hq:
            k_block_bhtd = np.repeat(k_block_bhtd, repeats, axis=1)
            v_block_bhtd = np.repeat(v_block_bhtd, repeats, axis=1)

        # Scores
        scores = np.matmul(
            queries.astype(np.float32),
            k_block_bhtd.astype(np.float32).transpose(0, 1, 3, 2),
        ) * scale  # (B, Hq, Lq, block_T)

        # Causal / additive mask
        if additive_mask is not None:
            mask_slice = additive_mask[..., token_offset:token_offset + block_T]
            scores = scores + mask_slice
        elif causal:
            q_positions = np.arange(query_start_pos, query_start_pos + Lq)[:, None]
            kv_positions = np.arange(token_offset, token_offset + block_T)[None, :]
            causal_mask = q_positions >= kv_positions
            causal_mask = np.broadcast_to(
                causal_mask[None, None, :, :], (B, Hq, Lq, block_T)
            )
            scores = np.where(causal_mask, scores, -np.inf)

        # Online softmax update for this block
        block_max = np.max(scores, axis=-1, keepdims=True)
        new_max = np.maximum(running_max, block_max)

        old_scale = np.exp(running_max - new_max)

        running_sum = running_sum * old_scale
        out = out * old_scale

        block_exp = np.exp(scores.astype(np.float32) - new_max)
        running_sum = running_sum + np.sum(block_exp, axis=-1, keepdims=True)
        out = out + np.matmul(block_exp, v_block_bhtd.astype(np.float32))

        running_max = new_max
        token_offset += block_T

    # Process sealed blocks
    for kb, vb in zip(key_blocks, value_blocks):
        k_dense = _decode_block_bhtd(kb, key_codec)
        v_dense = _decode_block_bhtd(vb, value_codec)
        _process_block(k_dense, v_dense)

    # Process staging
    if stage_k is not None:
        _process_block(stage_k, stage_v)

    # Process dense residual
    if dense_k is not None:
        _process_block(dense_k, dense_v)

    # Normalize — guard against fully-masked rows where running_sum == 0
    output = np.where(running_sum == 0, 0.0, out / running_sum)
    return output.astype(queries.dtype)


def numpy_dense_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scale: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standard dense attention (NumPy reference).

    Parameters
    ----------
    queries
        Shape ``(B, Hq, Lq, D)``.
    keys, values
        Shape ``(B, Hkv, T, D)``.
    scale
        Attention scale.
    mask
        Optional additive mask of shape ``(Lq, T)`` or broadcastable.

    Returns
    -------
    output
        Shape ``(B, Hq, Lq, D)``.
    """
    B, Hq, Lq, D = queries.shape
    _, Hkv, T, _ = keys.shape

    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({Hkv})")

    repeats = Hq // Hkv
    k_expanded = np.repeat(keys, repeats, axis=1)
    v_expanded = np.repeat(values, repeats, axis=1)

    scores = np.matmul(
        queries.astype(np.float32),
        k_expanded.astype(np.float32).transpose(0, 1, 3, 2),
    ) * scale

    if mask is not None:
        scores = scores + mask

    # Causal mask by default if no explicit mask and keys are longer than queries
    # (caller is expected to provide mask if needed)

    max_score = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores.astype(np.float32) - max_score)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / sum_exp

    output = np.matmul(weights, v_expanded.astype(np.float32))
    return output.astype(queries.dtype)
