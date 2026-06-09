"""RFSN v10 — Canonical causal attention reference.

This is the single authoritative implementation of causal masked attention.

Rules:
1. One-token decode (T_q == 1) may skip the causal mask because all KV
   tokens are strictly in the past.  This is mathematically safe.
2. Multi-token prefill (T_q > 1) MUST apply the causal mask.  Skipping
   it allows future-token leakage and silently corrupts generation.
3. Dense fallback MUST call this function, not mx.fast.scaled_dot_product_attention
   directly.
4. Audit comparisons MUST use this function as the reference.
5. Sparse-vs-dense drift tests MUST compare against this.

Usage::

    from rfsn_v10.attention_reference import causal_attention_dense

    out = causal_attention_dense(q, k, v, backend="mlx")
"""
from __future__ import annotations

import math

from .compat import MLX_AVAILABLE, mx


def causal_attention_dense(
    q,
    k,
    v,
    *,
    scale: float | None = None,
    backend: str = "mlx",
):
    """Compute causal scaled dot-product attention.

    Always applies a causal mask when T_q > 1.  For T_q == 1 the mask is
    skipped because a single query can only attend to past tokens.

    Args:
        q: Query tensor [B, H, T_q, D].
        k: Key tensor   [B, H, T_k, D].
        v: Value tensor [B, H, T_k, D].
        scale: Attention scale.  Defaults to ``1 / sqrt(D)``.
        backend: ``"mlx"`` (default) or ``"numpy"``.

    Returns:
        Output tensor [B, H, T_q, D] with the same dtype as *q*.

    Raises:
        RuntimeError: If ``backend="mlx"`` but MLX is not installed.
        ValueError: If tensor shapes are invalid.
    """
    if backend == "mlx":
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX backend requested but mlx is not installed.  "
                "Install mlx or set backend='numpy' for portable operation."
            )
        return _causal_attention_mlx(q, k, v, scale=scale)
    elif backend == "numpy":
        return _causal_attention_numpy(q, k, v, scale=scale)
    else:
        raise ValueError(f"Unknown backend: {backend!r}.  Choose 'mlx' or 'numpy'.")


# ---------------------------------------------------------------------------
# MLX implementation
# ---------------------------------------------------------------------------

def _causal_attention_mlx(q, k, v, *, scale: float | None = None):
    """MLX causal attention.  T_q == 1 uses the fast path; T_q > 1 applies
    a manual causal mask to prevent future-token leakage."""
    B, H, T_q, D = q.shape
    T_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if T_q == 1:
        # Decode path: single query attends only to past KV — no mask needed.
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    # Prefill path: apply causal mask manually.
    # Query at position i (offset = T_k - T_q) may attend to keys j <= i + offset.
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, H, T_q, T_k]
    q_idx = mx.arange(T_q, dtype=mx.int32).reshape(1, 1, T_q, 1)
    k_idx = mx.arange(T_k, dtype=mx.int32).reshape(1, 1, 1, T_k)
    offset = T_k - T_q
    mask = (k_idx <= (q_idx + offset)).astype(scores.dtype)
    scores = scores * mask + (1.0 - mask) * mx.array(-1e9, dtype=scores.dtype)
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


# ---------------------------------------------------------------------------
# NumPy implementation (for CPU validation without MLX)
# ---------------------------------------------------------------------------

def _causal_attention_numpy(q, k, v, *, scale: float | None = None):
    """Pure NumPy causal attention for portability tests and kernel validation."""
    import numpy as np

    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    B, H, T_q, D = q.shape
    T_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # scores: [B, H, T_q, T_k]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if T_q > 1:
        # Apply causal mask
        q_idx = np.arange(T_q).reshape(1, 1, T_q, 1)
        k_idx = np.arange(T_k).reshape(1, 1, 1, T_k)
        offset = T_k - T_q
        mask = k_idx <= (q_idx + offset)
        scores = np.where(mask, scores, -1e9)

    # softmax over T_k
    scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)

    return weights @ v
