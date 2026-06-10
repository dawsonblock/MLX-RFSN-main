"""Exact dense tiled attention kernel for RFSN v11.

Builds in stages (no skipping):
  Step 22: exact dense tiled attention — single head, no batch, no compression
  Step 23: add causal mask, multi-head, compressed K/V dequant
  Step 24: add residual split + SnapKV selected-block path

Design principle: IO-aware tiling to reduce memory reads/writes.
Each stage validates against MLX dense reference before the next stage.

Reference: FlashAttention (Dao et al. 2022) — exact attention should be
IO-aware and tiled before building aggressive sparse kernels.

This is a pure Python/MLX implementation. No custom Metal kernel yet.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Step 22: Exact dense tiled attention — single head, no batch
# ---------------------------------------------------------------------------

def tiled_attention_single_head(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    tile_size: int = 64,
    causal: bool = False,
) -> "mx.array":
    """Compute attention Q @ K^T @ V using tiled accumulation.

    Parameters
    ----------
    q, k, v : (seq_len, head_dim) float arrays
    tile_size : int
        Tile size for the sequence dimension.
    causal : bool
        If True, mask out future positions (only used in Step 23).

    Returns
    -------
    output : (seq_len, head_dim)
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required for tiled_attention")

    seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = mx.zeros_like(q)

    n_tiles_q = (seq_len + tile_size - 1) // tile_size
    n_tiles_kv = (seq_len + tile_size - 1) // tile_size

    for i in range(n_tiles_q):
        q_start = i * tile_size
        q_end = min((i + 1) * tile_size, seq_len)
        q_tile = q[q_start:q_end]  # (T_tile, D)

        # Accumulate over KV tiles
        for j in range(n_tiles_kv):
            k_start = j * tile_size
            k_end = min((j + 1) * tile_size, seq_len)
            k_tile = k[k_start:k_end]  # (T_tile, D)
            v_tile = v[k_start:k_end]  # (T_tile, D)

            # Attention scores for this Q-tile × KV-tile
            scores = q_tile @ k_tile.T * scale  # (T_q, T_kv)

            if causal:
                # Create causal mask for this tile pair
                row_idx = mx.arange(q_start, q_end)[:, None]
                col_idx = mx.arange(k_start, k_end)[None, :]
                causal_mask = col_idx <= row_idx  # True = allowed
                scores = mx.where(
                    causal_mask,
                    scores,
                    mx.array(-1e9, dtype=scores.dtype),
                )

            # Softmax over KV dimension
            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            exp_scores = mx.exp(scores)
            sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
            weights = exp_scores / (sum_exp + 1e-12)

            # Accumulate weighted values
            output[q_start:q_end] += weights @ v_tile

    return output


# ---------------------------------------------------------------------------
# Step 23: Multi-head + compressed K/V dequant
# ---------------------------------------------------------------------------

def tiled_attention_multi_head(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    tile_size: int = 64,
    causal: bool = True,
    dequant_fn: Optional[Any] = None,
) -> "mx.array":
    """Multi-head tiled attention with optional dequantization.

    Parameters
    ----------
    q, k, v : (heads, seq_len, head_dim) float arrays
    dequant_fn : callable, optional
        If provided, called on k and v before each tile to decompress
        from a compressed representation.

    Returns
    -------
    output : (heads, seq_len, head_dim)
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required")

    H, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = mx.zeros_like(q)

    n_tiles_q = (seq_len + tile_size - 1) // tile_size
    n_tiles_kv = (seq_len + tile_size - 1) // tile_size

    for h in range(H):
        q_h = q[h]
        k_h = k[h]
        v_h = v[h]

        for i in range(n_tiles_q):
            q_start = i * tile_size
            q_end = min((i + 1) * tile_size, seq_len)
            q_tile = q_h[q_start:q_end]

            acc = mx.zeros((q_end - q_start, head_dim), dtype=q.dtype)
            max_score = mx.full((q_end - q_start, 1), -1e9, dtype=q.dtype)
            sum_exp = mx.zeros((q_end - q_start, 1), dtype=q.dtype)

            for j in range(n_tiles_kv):
                k_start = j * tile_size
                k_end = min((j + 1) * tile_size, seq_len)

                k_tile = k_h[k_start:k_end]
                v_tile = v_h[k_start:k_end]

                if dequant_fn is not None:
                    k_tile = dequant_fn(k_tile, h)
                    v_tile = dequant_fn(v_tile, h)

                scores = q_tile @ k_tile.T * scale  # (T_q, T_kv)

                if causal:
                    row_idx = mx.arange(q_start, q_end)[:, None]
                    col_idx = mx.arange(k_start, k_end)[None, :]
                    causal_mask = col_idx <= row_idx
                    scores = mx.where(
                        causal_mask,
                        scores,
                        mx.array(-1e9, dtype=scores.dtype),
                    )

                # Online softmax (numerically stable)
                tile_max = mx.max(scores, axis=-1, keepdims=True)
                new_max = mx.maximum(max_score, tile_max)

                # Correct accumulated sum for new max
                corr = mx.exp(max_score - new_max)
                sum_exp = sum_exp * corr

                exp_scores = mx.exp(scores - new_max)
                sum_exp = sum_exp + mx.sum(exp_scores, axis=-1, keepdims=True)
                acc = acc * corr + exp_scores @ v_tile

                max_score = new_max

            output[h, q_start:q_end] = acc / (sum_exp + 1e-12)

    return output


# ---------------------------------------------------------------------------
# Step 24: Residual split + SnapKV selected-block path
# ---------------------------------------------------------------------------

def tiled_attention_with_residual(
    q: "mx.array",
    k_compressed: "mx.array",
    v_compressed: "mx.array",
    k_residual: Optional["mx.array"] = None,
    v_residual: Optional["mx.array"] = None,
    selected_blocks: Optional[list[int]] = None,
    block_size: int = 64,
    tile_size: int = 64,
    causal: bool = True,
    dequant_fn: Optional[Any] = None,
) -> "mx.array":
    """Tiled attention with residual window
    and optional SnapKV block selection.

    Parameters
    ----------
    q : (heads, seq_len, head_dim)
    k_compressed, v_compressed : compressed history
    k_residual, v_residual : optional FP16 residual window
        (heads, res_len, head_dim)
    selected_blocks : list[int], optional
        Block indices selected by SnapKV.
        Only these blocks are used from history.
    block_size : int
        Size of each block (default 64).
    dequant_fn : callable, optional
        Decompress function for compressed K/V.

    Returns
    -------
    output : (heads, seq_len, head_dim)
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required")

    H, seq_len, head_dim = q.shape

    # Determine which KV positions to use
    if selected_blocks is not None:
        # Only use selected blocks from compressed history
        use_positions: list[int] = []
        for b in sorted(selected_blocks):
            start = b * block_size
            end = (b + 1) * block_size
            use_positions.extend(range(start, end))
    else:
        # Use all compressed history
        use_positions = list(range(k_compressed.shape[1]))

    # Build full KV sequence: compressed positions + residual window
    if k_residual is not None:
        k_full = mx.concatenate(
            [k_compressed[:, use_positions], k_residual], axis=1
        )
        v_full = mx.concatenate(
            [v_compressed[:, use_positions], v_residual], axis=1
        )
    else:
        k_full = k_compressed[:, use_positions]
        v_full = v_compressed[:, use_positions]

    return tiled_attention_multi_head(
        q, k_full, v_full,
        tile_size=tile_size,
        causal=causal,
        dequant_fn=dequant_fn,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_against_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    fn: Any,
    causal: bool = False,
) -> tuple[float, float]:
    """Validate a tiled attention function against dense reference.

    Returns (cosine_similarity, max_abs_error).
    """
    if not _MLX_AVAILABLE:
        return 0.0, float("inf")

    # Dense reference
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.T * scale
    if causal:
        mask = np.tril(np.ones_like(scores), k=0)
        scores = np.where(mask, scores, -1e9)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
    ref_output = weights @ v

    # Tiled output
    q_mx = mx.array(q)
    k_mx = mx.array(k)
    v_mx = mx.array(v)
    tiled_output = np.array(fn(q_mx, k_mx, v_mx, causal=causal))

    flat_ref = ref_output.flatten()
    flat_tiled = tiled_output.flatten()

    cosine = float(
        np.dot(flat_ref, flat_tiled)
        / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_tiled) + 1e-12)
    )
    max_err = float(np.max(np.abs(ref_output - tiled_output)))
    return cosine, max_err
