"""Tests for PolarCache."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.cache import PolarCache
from rfsn_v11.polar_fused.config import PolarFusedConfig


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_and_grow() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
        block_size=4,
    )
    # Append 3 tokens (below block size)
    for _ in range(3):
        ki = mx.zeros((1, 2, 1, 64), dtype=mx.uint8)
        kn = mx.ones((1, 2, 1), dtype=mx.float32)
        vi = mx.zeros((1, 2, 1, 64), dtype=mx.uint8)
        vn = mx.ones((1, 2, 1), dtype=mx.float32)
        cache.append(ki, kn, vi, vn)

    assert cache.state is not None
    assert cache.state.offset == 3
    assert cache.state.capacity >= 4


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_grow_beyond_block() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    # Append 10 tokens to trigger growth
    for _ in range(10):
        ki = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        kn = mx.ones((1, 1, 1), dtype=mx.float32)
        vi = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        vn = mx.ones((1, 1, 1), dtype=mx.float32)
        cache.append(ki, kn, vi, vn)

    assert cache.state.offset == 10
    assert cache.state.capacity >= 10


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_trim() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    for _ in range(8):
        ki = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        kn = mx.ones((1, 1, 1), dtype=mx.float32)
        vi = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        vn = mx.ones((1, 1, 1), dtype=mx.float32)
        cache.append(ki, kn, vi, vn)

    cache.trim(5)
    assert cache.state.offset == 5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_trim_to_zero() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    ki = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
    kn = mx.ones((1, 1, 1), dtype=mx.float32)
    vi = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
    vn = mx.ones((1, 1, 1), dtype=mx.float32)
    cache.append(ki, kn, vi, vn)
    cache.trim(0)
    assert cache.state.offset == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_memory_accounting() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    # Empty cache
    assert cache.memory_bytes() == 0

    for _ in range(4):
        ki = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        kn = mx.ones((1, 1, 1), dtype=mx.float32)
        vi = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        vn = mx.ones((1, 1, 1), dtype=mx.float32)
        cache.append(ki, kn, vi, vn)

    # Memory should be positive
    assert cache.memory_bytes() > 0
    # Capacity >= memory
    assert cache.capacity_bytes() >= cache.memory_bytes()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_compression_ratio() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    for _ in range(10):
        ki = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        kn = mx.ones((1, 1, 1), dtype=mx.float32)
        vi = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        vn = mx.ones((1, 1, 1), dtype=mx.float32)
        cache.append(ki, kn, vi, vn)

    ratio = cache.compression_ratio()
    # With unpacked uint8 indices, compression won't be great,
    # but it should be defined and > 0
    assert ratio > 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metadata() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
        block_size=4,
    )
    meta = cache.metadata()
    assert meta["batch_size"] == 1
    assert meta["num_kv_heads"] == 2
    assert meta["head_dim"] == 64
    assert meta["offset"] == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_validation_empty() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cache = PolarCache(
        config=cfg,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=4,
    )
    cache.validate()  # should not raise


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="Batch size must be 1"):
        PolarCache(
            config=PolarFusedConfig.polar_safe(),
            batch_size=2,
            num_kv_heads=1,
            head_dim=64,
            block_size=4,
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_invalid_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be"):
        PolarCache(
            config=PolarFusedConfig.polar_safe(),
            batch_size=1,
            num_kv_heads=1,
            head_dim=32,
            block_size=4,
        )
