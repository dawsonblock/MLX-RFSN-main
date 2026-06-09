"""Drift detector: sparse vs dense attention KL divergence < 5e-5.

Task 1.3 of the repair plan.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")  # noqa: E402

from rfsn_v10.attention import AdaptiveBlockSparseAttention


def _reference_dense_causal(queries, keys, values):
    """Reference causal dense attention using raw MLX ops."""
    B, H, T_q, D = queries.shape
    T_k = keys.shape[2]
    scale = 1.0 / (D ** 0.5)
    scores = queries @ keys.transpose(0, 1, 3, 2) * scale
    q_pos = mx.arange(T_q, dtype=mx.int32).reshape(1, 1, T_q, 1)
    k_pos = mx.arange(T_k, dtype=mx.int32).reshape(1, 1, 1, T_k)
    offset = T_k - T_q
    causal = (k_pos <= (q_pos + offset)).astype(scores.dtype)
    scores = scores * causal + (1.0 - causal) * mx.array(
        -1e9, dtype=scores.dtype
    )
    weights = mx.softmax(scores, axis=-1)
    return weights @ values


def _kl_divergence(p, q, eps=1e-10):
    """KL(P || Q) averaged over batch and heads."""
    p = mx.maximum(p, eps)
    q = mx.maximum(q, eps)
    kl = mx.sum(p * (mx.log(p) - mx.log(q)), axis=-1)
    return float(mx.mean(kl).item())


def test_prefill_causal_mask_matches_reference():
    """Prefill (T_q > 1) dense fallback must match reference causal attn."""
    q = mx.random.normal((1, 4, 64, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    out, active, mode = AdaptiveBlockSparseAttention.execute(
        q, k, v, top_k_ratio=0.25
    )
    mx.eval(out)

    assert mode == "dense_prefill"
    ref = _reference_dense_causal(q, k, v)
    mx.eval(ref)

    diff = float(mx.mean(mx.abs(out - ref)).item())
    assert diff < 1e-5, f"prefill dense fallback drift {diff} >= 1e-5"


def test_decode_sparse_vs_dense_kl():
    """KL divergence between sparse and dense decode < 5e-5.

    Uses top_k_ratio=0.99 so only 1 % of blocks are dropped;
    context is large enough (6400 tokens / 100 blocks) that the
    sparse path is still exercised.
    """
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 6400, 64))
    v = mx.random.normal((1, 4, 6400, 64))

    out_sparse, active, mode = AdaptiveBlockSparseAttention.execute(
        q, k, v, top_k_ratio=0.99
    )
    mx.eval(out_sparse)
    assert mode == "sparse_compacted"

    out_dense, _, _ = AdaptiveBlockSparseAttention._dense_masked(
        q, k, v, scale=1.0 / (64 ** 0.5), block_size=64, mode="ref"
    )
    mx.eval(out_dense)

    p = mx.softmax(out_sparse, axis=-1)
    q_dist = mx.softmax(out_dense, axis=-1)
    kl = _kl_divergence(p, q_dist)
    assert kl < 5e-5, f"sparse vs dense KL {kl} >= 5e-5"


def test_decode_8k_context_sparse_vs_dense_kl():
    """KL divergence on 8k context with near-complete block retention."""
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 8192, 64))
    v = mx.random.normal((1, 4, 8192, 64))

    out_sparse, active, mode = AdaptiveBlockSparseAttention.execute(
        q, k, v, top_k_ratio=0.99
    )
    mx.eval(out_sparse)
    assert mode == "sparse_compacted"

    out_dense, _, _ = AdaptiveBlockSparseAttention._dense_masked(
        q, k, v, scale=1.0 / (64 ** 0.5), block_size=64, mode="ref"
    )
    mx.eval(out_dense)

    p = mx.softmax(out_sparse, axis=-1)
    q_dist = mx.softmax(out_dense, axis=-1)
    kl = _kl_divergence(p, q_dist)
    assert kl < 5e-5, f"8k sparse vs dense KL {kl} >= 5e-5"
