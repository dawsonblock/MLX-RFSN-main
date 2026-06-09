#!/usr/bin/env python3
"""
RFSN v10 — Scoring mode acceptance tests.

Covers mode validation, fallback selection, and config parsing
without requiring full MLX execution where possible.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
from rfsn_v10.runtime.scoring_modes import (  # noqa: E402, I001
    score_attention_fp16,
    score_attention_prepared,
    score_attention_reconstructed,
    score_attention_packed_block,
    score_attention_score_corrected,
)


# ------------------------------------------------------------------
# Mode validation / smoke tests
# ------------------------------------------------------------------

class TestScoringModesSmoke:
    def test_fp16_runs_with_random_tensors(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_fp16(q, k, v)
        assert out.shape == q.shape

    def test_prepared_runs_with_random_tensors(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_prepared(q, k, v)
        assert out.shape == q.shape

    def test_reconstructed_runs_with_random_tensors(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_reconstructed(
            q, "packet", "packet",
            dequant_fn=lambda _kp, _vp: (k, v),
        )
        assert out.shape == q.shape

    def test_packed_block_runs_with_random_tensors(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_packed_block(
            q, "packet", "packet",
            block_indices=[0, 1],
            block_dequant_fn=lambda _kp, _vp, _bi: (k, v),
        )
        assert out.shape == q.shape

    def test_score_corrected_raises_without_fn(self):
        q = mx.random.normal((1, 4, 1, 64))
        with pytest.raises(ValueError, match="requires correction_fn"):
            score_attention_score_corrected(q, None, None)

    def test_score_corrected_runs_with_fn(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_score_corrected(
            q, k, v,
            correction_fn=lambda _q, _k, **_kw: (
                q.astype(mx.float32) @ k.astype(mx.float32).transpose(
                    0, 1, 3, 2
                )
            ) / 8.0,
        )
        assert out.shape == q.shape

    def test_fp16_scale_override(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_fp16(q, k, v, scale=0.5)
        assert out.shape == q.shape

    def test_packed_block_with_array_indices(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        indices = mx.array([0, 1])
        out = score_attention_packed_block(
            q, "packet", "packet",
            block_indices=indices,
            block_dequant_fn=lambda _kp, _vp, _bi: (k, v),
        )
        assert out.shape == q.shape

    def test_reconstructed_with_scale(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out = score_attention_reconstructed(
            q, "p", "p",
            dequant_fn=lambda _kp, _vp: (k, v),
            scale=0.5,
        )
        assert out.shape == q.shape


# ------------------------------------------------------------------
# Numerical sanity: fp16 baseline should be deterministic
# ------------------------------------------------------------------

class TestScoringModesDeterminism:
    def test_fp16_is_deterministic(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out1 = score_attention_fp16(q, k, v)
        out2 = score_attention_fp16(q, k, v)
        assert mx.allclose(out1, out2, atol=1e-5).item()

    def test_prepared_matches_fp16_for_same_inputs(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out_fp16 = score_attention_fp16(q, k, v)
        out_prep = score_attention_prepared(q, k, v)
        assert mx.allclose(out_fp16, out_prep, atol=1e-5).item()

    def test_reconstructed_matches_fp16_when_identity_dequant(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        out_fp16 = score_attention_fp16(q, k, v)
        out_rec = score_attention_reconstructed(
            q, "p", "p", dequant_fn=lambda _kp, _vp: (k, v)
        )
        assert mx.allclose(out_fp16, out_rec, atol=1e-5).item()
