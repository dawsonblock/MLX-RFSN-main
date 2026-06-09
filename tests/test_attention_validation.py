#!/usr/bin/env python3
"""RFSN v10 — Additional attention validation and edge-case tests.

Covers input validation, dense fallback modes, block merging logic,
and dtype/batch/head dimension mismatches.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.attention import AdaptiveBlockSparseAttention


class TestValidateInputs:
    def test_rejects_batch_mismatch(self):
        q = mx.random.normal((2, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        with pytest.raises(ValueError, match="batch mismatch"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=0.7,
            )

    def test_rejects_head_mismatch(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 8, 512, 64))
        v = mx.random.normal((1, 8, 512, 64))
        with pytest.raises(ValueError, match="head mismatch"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=0.7,
            )

    def test_rejects_dim_mismatch(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 32))
        v = mx.random.normal((1, 4, 512, 32))
        with pytest.raises(ValueError, match="head_dim mismatch"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=0.7,
            )

    def test_rejects_non_positive_t(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 0, 64))
        v = mx.random.normal((1, 4, 0, 64))
        with pytest.raises(ValueError, match="positive"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=0.7,
            )

    def test_rejects_non_positive_block_size(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        with pytest.raises(ValueError, match="block_size"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=0,
                consensus_mix=0.7,
            )

    def test_rejects_top_k_out_of_range(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        with pytest.raises(ValueError, match="top_k_ratio"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=-0.1, block_size=64,
                consensus_mix=0.7,
            )

    def test_rejects_infinite_consensus_mix(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        with pytest.raises(ValueError, match="finite"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=float("inf"),
            )

    def test_rejects_dtype_mismatch(self):
        q = mx.random.normal((1, 4, 1, 64), dtype=mx.float16)
        k = mx.random.normal((1, 4, 512, 64), dtype=mx.float32)
        v = mx.random.normal((1, 4, 512, 64), dtype=mx.float32)
        with pytest.raises(ValueError, match="dtype mismatch"):
            AdaptiveBlockSparseAttention._validate_inputs(
                q, k, v, top_k_ratio=0.25, block_size=64,
                consensus_mix=0.7,
            )

    def test_returns_dimensions(self):
        q = mx.random.normal((2, 8, 1, 64))
        k = mx.random.normal((2, 8, 512, 64))
        v = mx.random.normal((2, 8, 512, 64))
        result = AdaptiveBlockSparseAttention._validate_inputs(
            q, k, v, top_k_ratio=0.25, block_size=64,
            consensus_mix=0.7,
        )
        assert result == (2, 8, 1, 512, 64)


class TestDenseFallbackModes:
    def test_dense_short_context(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 32, 64))
        v = mx.random.normal((1, 4, 32, 64))
        out, active, mode = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=0.25, block_size=64,
        )
        mx.eval(out)
        assert mode == "dense_short_context"
        assert out.shape == q.shape

    def test_dense_requested_full_top_k(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        out, active, mode = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=1.0, block_size=64,
        )
        mx.eval(out)
        assert mode == "dense_requested"
        assert out.shape == q.shape

    def test_dense_not_strictly_past(self):
        q = mx.random.normal((1, 4, 1, 64))
        k = mx.random.normal((1, 4, 512, 64))
        v = mx.random.normal((1, 4, 512, 64))
        out, active, mode = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=0.25, block_size=64,
            kv_is_strictly_past=False,
        )
        mx.eval(out)
        assert mode == "dense_not_strictly_past"


class TestMergeReservedAndScoredBlocks:
    def test_budget_overflow_allowed(self):
        selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
            num_blocks=4, k_active=2,
            score_selected=[2, 3],
            reserved_sink_blocks=1, reserved_recent_blocks=1,
            allow_budget_overflow=True,
        )
        assert len(selected) >= 2
        assert 0 in selected  # sink
        assert 3 in selected  # recent

    def test_budget_overflow_disallowed(self):
        selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
            num_blocks=4, k_active=2,
            score_selected=[2, 3],
            reserved_sink_blocks=1, reserved_recent_blocks=1,
            allow_budget_overflow=False,
        )
        assert len(selected) == 2

    def test_no_duplicate_blocks(self):
        selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
            num_blocks=4, k_active=3,
            score_selected=[0, 1, 2],
            reserved_sink_blocks=1, reserved_recent_blocks=0,
            allow_budget_overflow=False,
        )
        # Block 0 is both sink and scored; should not duplicate
        assert len(selected) == len(set(selected))


class TestCeilDiv:
    def test_basic(self):
        assert AdaptiveBlockSparseAttention._ceil_div(10, 3) == 4
        assert AdaptiveBlockSparseAttention._ceil_div(9, 3) == 3

    def test_rejects_zero_divisor(self):
        with pytest.raises(ValueError, match="divisor"):
            AdaptiveBlockSparseAttention._ceil_div(10, 0)


class TestDtypeNbytes:
    def test_float16(self):
        assert AdaptiveBlockSparseAttention._dtype_nbytes(mx.float16) == 2

    def test_float32(self):
        assert AdaptiveBlockSparseAttention._dtype_nbytes(mx.float32) == 4

    def test_bfloat16(self):
        # bfloat16 may not exist in all MLX versions
        try:
            assert AdaptiveBlockSparseAttention._dtype_nbytes(mx.bfloat16) == 2
        except AttributeError:
            pytest.skip("bfloat16 not available")

    def test_unknown_defaults_to_4(self):
        assert AdaptiveBlockSparseAttention._dtype_nbytes(mx.int32) == 4
