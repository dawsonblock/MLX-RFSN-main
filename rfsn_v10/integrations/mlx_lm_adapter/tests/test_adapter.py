"""Tests for the explicit MLX-LM adapter."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def test_rfsn_quantized_kv_cache_interface() -> None:
    """RfsnQuantizedKVCache must satisfy the MLX-LM cache interface."""
    if not HAS_MLX:
        pytest.skip("MLX not available on this platform")
    try:
        from rfsn_v10.cache.cartesian_codec import CartesianCodec
        from rfsn_v10.cache.session import GenerationCacheSession
        from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

        k_codec = CartesianCodec(bits=8, group_size=64)
        v_codec = CartesianCodec(bits=5, group_size=64)
        session = GenerationCacheSession("test", 2, k_codec, v_codec)

        cache = RfsnQuantizedKVCache(
            layer_cache=session.get_layer_cache(0),
            session=session,
        )

        # Required interface attributes/methods
        assert hasattr(cache, "update_and_fetch")
        assert hasattr(cache, "offset")
        assert hasattr(cache, "state")
        assert hasattr(cache, "is_trimmable")
        assert hasattr(cache, "trim")
        assert cache.is_trimmable() is False
    except RuntimeError as e:
        if "MLX" in str(e):
            pytest.skip("MLX version mismatch on this platform")
        raise


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_update_and_fetch_appends_and_reconstructs() -> None:
    """update_and_fetch appends to quantized cache and returns dense."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test", 2, k_codec, v_codec)

    cache = RfsnQuantizedKVCache(
        layer_cache=session.get_layer_cache(0),
        session=session,
    )

    B, Hkv, T, D = 1, 2, 10, 64
    keys = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)

    full_k, full_v = cache.update_and_fetch(keys, values)

    assert full_k.shape == (B, Hkv, T, D)
    assert full_v.shape == (B, Hkv, T, D)
    assert cache.offset == T

    # Proof counters (legacy dict)
    assert session.get_counter("new_tokens_received") == T
    assert session.get_counter("new_tokens_encoded") == T
    assert session.get_counter("fallback_attention_calls") == 1

    # Unified RuntimeCounters
    # tokens_appended is only incremented by new_tokens_received to avoid double-counting
    assert session.runtime_counters.tokens_appended == T
    assert session.runtime_counters.dense_fallback_calls == 1


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_update_and_fetch_accumulates() -> None:
    """Multiple update_and_fetch calls accumulate tokens."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test", 2, k_codec, v_codec)

    cache = RfsnQuantizedKVCache(
        layer_cache=session.get_layer_cache(0),
        session=session,
    )

    B, Hkv, T, D = 1, 2, 10, 64
    for _ in range(3):
        keys = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)
        values = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)
        full_k, full_v = cache.update_and_fetch(keys, values)

    assert full_k.shape == (B, Hkv, 30, D)
    assert cache.offset == 30
    assert session.get_counter("new_tokens_received") == 30
    assert session.get_counter("fallback_attention_calls") == 3

    # Unified RuntimeCounters
    # tokens_appended is only incremented by new_tokens_received to avoid double-counting
    assert session.runtime_counters.tokens_appended == 30
    assert session.runtime_counters.dense_fallback_calls == 3


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_state_returns_empty_tuple() -> None:
    """state must return something mx.eval can handle."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test", 2, k_codec, v_codec)
    cache = RfsnQuantizedKVCache(
        layer_cache=session.get_layer_cache(0),
        session=session,
    )

    # mx.eval must not raise
    mx.eval([cache.state])


def test_model_inspection() -> None:
    """Model inspection extracts layer count and head dim."""
    from rfsn_v10.integrations.mlx_lm_adapter.model_inspection import inspect_model

    # Mock model with args
    class MockArgs:
        num_attention_heads = 8
        num_key_value_heads = 2
        hidden_size = 512
        head_dim = 64
        rope_theta = 10000.0

    class MockModel:
        model_type = "llama"
        args = MockArgs()
        layers = [object() for _ in range(12)]

    info = inspect_model(MockModel())
    assert info["num_layers"] == 12
    assert info["num_heads"] == 8
    assert info["num_kv_heads"] == 2
    assert info["head_dim"] == 64
