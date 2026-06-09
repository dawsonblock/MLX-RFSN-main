#!/usr/bin/env python3
"""MemoryGuard + runtime integration tests for Main11."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.memory_guard import MemoryGuard
from rfsn_v10 import RFSNRuntime


def test_soft_pressure_calls_eviction_callback_with_target_bytes() -> None:
    calls: list[int] = []

    def _evict(target: int) -> int:
        calls.append(target)
        return target

    guard = MemoryGuard(soft_limit_gb=1e-9, hard_limit_gb=1.0, eviction_callback=_evict)
    freed = guard.enforce_safety(estimated_cache_bytes=4096)

    assert calls
    assert calls[0] >= 4096
    assert freed == calls[0]


def test_runtime_hard_pressure_disables_sparse_path(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path), k_bits=8, v_bits=3)
    guard = MemoryGuard(soft_limit_gb=1e-9, hard_limit_gb=1e-9)
    runtime = RFSNRuntime(
        kv_manager=manager,
        model_id="m",
        block_size=64,
        top_k_ratio=0.5,
        enable_sparse_decode=True,
        memory_guard=guard,
    )

    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    _, info = runtime.execute_decode_step(
        skill_pattern="pressure",
        layer_id="l0",
        batch_id="b0",
        queries=q,
        keys=k,
        values=v,
    )

    assert info["sparse_enabled"] is False


def test_runtime_hard_pressure_skips_compressed_store(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path), k_bits=8, v_bits=3)
    guard = MemoryGuard(soft_limit_gb=1e-9, hard_limit_gb=1e-9)
    runtime = RFSNRuntime(
        kv_manager=manager,
        model_id="m",
        block_size=64,
        top_k_ratio=0.5,
        enable_sparse_decode=True,
        memory_guard=guard,
    )

    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    runtime.execute_decode_step(
        skill_pattern="pressure_store",
        layer_id="l0",
        batch_id="b0",
        queries=q,
        keys=k,
        values=v,
    )

    assert "pressure_store" not in manager.active_caches
