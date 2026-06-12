"""Tests for LazyPolarCache."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.lazy_convert import CacheState, LazyPolarCache
from rfsn_v11.polar_fused.config import PolarFusedConfig
from rfsn_v11.polar_fused.quantize import PolarQuantizer


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_empty_state() -> None:
    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=cfg.head_dim, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=cfg.head_dim, rotation_seed=cfg.value_rotation_seed)

    cache = LazyPolarCache(cfg, batch_size=1, num_kv_heads=2, head_dim=cfg.head_dim,
                           key_quantizer=key_q, value_quantizer=value_q)
    assert cache.state == CacheState.EMPTY
    assert cache.token_count == 0
    assert cache.memory_bytes() == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_fp16_warmup_below_threshold() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cfg = PolarFusedConfig(**{**cfg.__dict__, "lazy_quantization_tokens": 10})
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=cfg.head_dim, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=cfg.head_dim, rotation_seed=cfg.value_rotation_seed)

    cache = LazyPolarCache(cfg, batch_size=1, num_kv_heads=2, head_dim=cfg.head_dim,
                           key_quantizer=key_q, value_quantizer=value_q)

    # Append 5 tokens (below threshold of 10)
    for _ in range(5):
        k = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        v = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        cache.append(k, v)

    assert cache.state == CacheState.FP16_WARMUP
    assert cache.token_count == 5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_conversion_at_threshold() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cfg = PolarFusedConfig(**{**cfg.__dict__, "lazy_quantization_tokens": 4})
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=cfg.head_dim, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=cfg.head_dim, rotation_seed=cfg.value_rotation_seed)

    cache = LazyPolarCache(cfg, batch_size=1, num_kv_heads=2, head_dim=cfg.head_dim,
                           key_quantizer=key_q, value_quantizer=value_q)

    # Append 6 tokens (above threshold of 4)
    for _ in range(6):
        k = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        v = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        cache.append(k, v)

    assert cache.state == CacheState.POLAR_PACKED
    assert cache.token_count == 6
    assert cache._polar_cache is not None
    assert cache._polar_cache.state.offset == 6


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_trim_fp16() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cfg = PolarFusedConfig(**{**cfg.__dict__, "lazy_quantization_tokens": 10})
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=cfg.head_dim, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=cfg.head_dim, rotation_seed=cfg.value_rotation_seed)

    cache = LazyPolarCache(cfg, batch_size=1, num_kv_heads=2, head_dim=cfg.head_dim,
                           key_quantizer=key_q, value_quantizer=value_q)

    for _ in range(5):
        k = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        v = mx.random.normal(shape=(1, 2, 1, cfg.head_dim))
        cache.append(k, v)

    cache.trim(3)
    assert cache.token_count == 3
    assert cache._fp16_keys.shape[2] == 3


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metadata() -> None:
    cfg = PolarFusedConfig.polar_safe()
    cfg = PolarFusedConfig(**{**cfg.__dict__, "lazy_quantization_tokens": 10})
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=cfg.head_dim, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=cfg.head_dim, rotation_seed=cfg.value_rotation_seed)

    cache = LazyPolarCache(cfg, batch_size=1, num_kv_heads=2, head_dim=cfg.head_dim,
                           key_quantizer=key_q, value_quantizer=value_q)

    meta = cache.metadata()
    assert meta["state"] == "EMPTY"
    assert meta["token_count"] == 0
    assert meta["threshold"] == 10
