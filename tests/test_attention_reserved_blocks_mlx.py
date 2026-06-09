#!/usr/bin/env python3
"""Reserved block behavior tests for sparse attention."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.attention import AdaptiveBlockSparseAttention


def test_sink_block_always_retained() -> None:
    selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
        num_blocks=16,
        k_active=6,
        score_selected=[5, 6, 7, 8, 9, 10],
        reserved_sink_blocks=1,
        reserved_recent_blocks=0,
        allow_budget_overflow=False,
    )
    assert 0 in selected


def test_recent_blocks_always_retained() -> None:
    selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
        num_blocks=16,
        k_active=6,
        score_selected=[1, 2, 3, 4, 5, 6],
        reserved_sink_blocks=0,
        reserved_recent_blocks=2,
        allow_budget_overflow=False,
    )
    assert 15 in selected
    assert 14 in selected


def test_blocks_remain_sorted_and_deduped() -> None:
    selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
        num_blocks=16,
        k_active=8,
        score_selected=[7, 6, 6, 2, 1, 0],
        reserved_sink_blocks=1,
        reserved_recent_blocks=2,
        allow_budget_overflow=False,
    )
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected))


def test_active_blocks_respect_budget_without_overflow() -> None:
    selected = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
        num_blocks=16,
        k_active=4,
        score_selected=[7, 8, 9, 10, 11],
        reserved_sink_blocks=1,
        reserved_recent_blocks=2,
        allow_budget_overflow=False,
    )
    assert len(selected) <= 4


def test_prefill_still_falls_back_dense() -> None:
    q = mx.random.normal((1, 4, 8, 64))
    k = mx.random.normal((1, 4, 256, 64))
    v = mx.random.normal((1, 4, 256, 64))

    _, _, mode = AdaptiveBlockSparseAttention.execute(
        q,
        k,
        v,
        top_k_ratio=0.5,
        block_size=64,
    )
    assert mode == "dense_prefill"


def test_kv_not_strictly_past_falls_back_dense() -> None:
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 256, 64))
    v = mx.random.normal((1, 4, 256, 64))

    _, _, mode = AdaptiveBlockSparseAttention.execute(
        q,
        k,
        v,
        top_k_ratio=0.5,
        block_size=64,
        kv_is_strictly_past=False,
    )
    assert mode == "dense_not_strictly_past"
