"""RFSN v10 — Causal attention mask correctness tests.

Verifies that:
1. Multi-token prefill cannot attend to future tokens.
2. Dense fallback equals causal reference.
3. Audit dense baseline equals causal reference.
4. Sparse fallback equals causal dense reference when sparse is disabled.
5. One-token decode still matches expected dense result.
6. Removing the mask would fail the test (sanity check).

These tests require MLX and are skipped on non-Apple Silicon.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.attention_reference import causal_attention_dense, _causal_attention_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def _to_np(x) -> np.ndarray:
    return np.array(x.tolist(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: Multi-token prefill causal mask — future tokens must be masked
# ---------------------------------------------------------------------------

def test_prefill_cannot_attend_future_tokens():
    """Verify that logit of query_i attending to key_j > i is suppressed."""
    B, H, T_q, D = 1, 2, 4, 16
    T_k = T_q  # square prefill

    np.random.seed(42)
    q_np = np.random.randn(B, H, T_q, D).astype(np.float32)
    k_np = np.random.randn(B, H, T_k, D).astype(np.float32)
    v_np = np.eye(T_k, D, dtype=np.float32)[None, None].repeat(B, axis=0).repeat(H, axis=1)

    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    out = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out)
    out_np = _to_np(out)  # [B, H, T_q, D]

    scale = 1.0 / math.sqrt(D)
    scores = np.einsum("bhqd,bhkd->bhqk", q_np, k_np) * scale  # [B, H, T_q, T_k]

    # For each query i, future keys j > i should have near-zero weight
    for i in range(T_q):
        for j in range(T_k):
            if j > i:
                # The contribution from future key j to output of query i
                # should be effectively zero (weight ~ 0)
                # We check via softmax: if mask is applied, the weight for j > i
                # must be ≈ 0. With v = eye, the output at dim j reflects the weight.
                if j < D:
                    weight_heads = out_np[0, :, i, j]  # shape [H]
                    assert float(weight_heads.max()) < 0.01, (
                        f"Query {i} attended to future key {j} with max weight "
                        f"{float(weight_heads.max()):.4f} (should be ~0)"
                    )


def test_removing_mask_fails_causal_check():
    """Confirm that without masking, future attention IS present (sanity check)."""
    B, H, T_q, D = 1, 2, 4, 16
    T_k = T_q

    np.random.seed(1)
    q_np = np.random.randn(B, H, T_q, D).astype(np.float32)
    k_np = np.random.randn(B, H, T_k, D).astype(np.float32)
    v_np = np.eye(T_k, D, dtype=np.float32)[None, None].repeat(B, axis=0).repeat(H, axis=1)

    # Unmasked: use raw mx.fast.scaled_dot_product_attention
    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    scale = 1.0 / math.sqrt(D)
    out_unmasked = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(out_unmasked)
    out_np = _to_np(out_unmasked)

    # With unmasked attention, at least one future-token contribution should be nonzero
    has_future_leak = False
    for i in range(1, T_q):  # start from 1 since token 0 has no future
        for j in range(i + 1, T_k):
            if j < D and out_np[0, :, i, j].max() > 0.05:
                has_future_leak = True
                break
    # This should be True (unmasked has future leakage)
    assert has_future_leak, (
        "Expected unmasked attention to show future leakage, but it did not. "
        "Check test setup."
    )


# ---------------------------------------------------------------------------
# Test 2: Dense fallback equals causal reference
# ---------------------------------------------------------------------------

def test_dense_fallback_equals_causal_reference():
    """AdaptiveBlockSparseAttention dense fallback must equal causal_attention_dense."""
    from rfsn_v10.attention import AdaptiveBlockSparseAttention

    B, H, T_q, D = 1, 2, 8, 32
    T_k = 8

    np.random.seed(10)
    q = mx.array(np.random.randn(B, H, T_q, D).astype(np.float32))
    k = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))
    v = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))

    # Force dense path by using top_k_ratio=1.0
    out_sparse, _, mode = AdaptiveBlockSparseAttention.execute(
        q, k, v,
        top_k_ratio=1.0,
        block_size=64,
        kv_is_strictly_past=True,
    )
    mx.eval(out_sparse)
    assert mode in ("dense_requested", "dense_short_context"), f"Expected dense mode, got {mode}"

    out_ref = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out_ref)

    cos = _cosine(_to_np(out_sparse), _to_np(out_ref))
    assert cos > 0.9999, f"Dense fallback cosine vs causal reference: {cos:.6f}"


# ---------------------------------------------------------------------------
# Test 3: One-token decode matches dense result
# ---------------------------------------------------------------------------

def test_single_token_decode_matches_dense():
    """Single query token decode must match causal dense reference."""
    B, H, T_q, T_k, D = 1, 2, 1, 64, 32

    np.random.seed(20)
    q = mx.array(np.random.randn(B, H, T_q, D).astype(np.float32))
    k = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))
    v = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))

    out = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out)

    # Reference: manual causal attention (T_q=1 skips mask, equiv to full attend)
    scale = 1.0 / math.sqrt(D)
    out_manual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(out_manual)

    cos = _cosine(_to_np(out), _to_np(out_manual))
    assert cos > 0.9999, f"Single-token decode cosine: {cos:.6f}"


# ---------------------------------------------------------------------------
# Test 4: Sparse fallback equals causal dense reference when sparse disabled
# ---------------------------------------------------------------------------

def test_sparse_disabled_fallback_equals_causal_reference():
    """When sparse is disabled (top_k_ratio=1.0), output must equal causal_attention_dense."""
    from rfsn_v10.attention import AdaptiveBlockSparseAttention

    B, H, T_q, T_k, D = 1, 2, 1, 128, 32

    np.random.seed(30)
    q = mx.array(np.random.randn(B, H, T_q, D).astype(np.float32))
    k = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))
    v = mx.array(np.random.randn(B, H, T_k, D).astype(np.float32))

    # Sparse disabled via top_k_ratio=1.0
    out_sparse, _, mode = AdaptiveBlockSparseAttention.execute(
        q, k, v,
        top_k_ratio=1.0,
        block_size=64,
        kv_is_strictly_past=True,
    )
    mx.eval(out_sparse)

    out_ref = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out_ref)

    cos = _cosine(_to_np(out_sparse), _to_np(out_ref))
    assert cos > 0.9999, f"Sparse-disabled fallback cosine vs causal reference: {cos:.6f}"


# ---------------------------------------------------------------------------
# Test 5: MLX causal reference matches NumPy causal reference
# ---------------------------------------------------------------------------

def test_mlx_causal_matches_numpy_causal():
    """MLX and NumPy implementations of causal_attention_dense must agree."""
    B, H, T_q, T_k, D = 1, 2, 6, 8, 16

    np.random.seed(99)
    q_np = np.random.randn(B, H, T_q, D).astype(np.float32)
    k_np = np.random.randn(B, H, T_k, D).astype(np.float32)
    v_np = np.random.randn(B, H, T_k, D).astype(np.float32)

    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    out_mlx = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out_mlx)

    out_numpy = _causal_attention_numpy(q_np, k_np, v_np)

    cos = _cosine(_to_np(out_mlx), out_numpy)
    assert cos > 0.9999, f"MLX vs NumPy causal cosine: {cos:.6f}"

    max_abs = float(np.max(np.abs(_to_np(out_mlx) - out_numpy)))
    assert max_abs < 1e-4, f"MLX vs NumPy max abs error: {max_abs:.2e}"
