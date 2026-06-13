"""Tests for the attention wrapper and direct packed cache."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_direct_packed_cache_interface() -> None:
    """RfsnDirectPackedKVCache must satisfy the MLX-LM cache interface."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
    )

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    cache = RfsnDirectPackedKVCache(
        layer_id=0,
        key_codec=k_codec,
        value_codec=v_codec,
        staging_capacity=64,
    )

    assert hasattr(cache, "update_and_fetch")
    assert hasattr(cache, "offset")
    assert hasattr(cache, "state")
    assert hasattr(cache, "is_trimmable")
    assert hasattr(cache, "trim")

    # update_and_fetch appends and updates offset
    keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    cache.update_and_fetch(keys, values)
    assert cache.offset == 10
    assert cache.layer_cache.total_token_count() == 10


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_direct_packed_cache_trim_raises() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = RfsnDirectPackedKVCache(layer_id=0, key_codec=k_codec, value_codec=v_codec)

    with pytest.raises(NotImplementedError):
        cache.trim(5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_direct_packed_cache_state_injection_raises() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = RfsnDirectPackedKVCache(layer_id=0, key_codec=k_codec, value_codec=v_codec)

    with pytest.raises(NotImplementedError):
        cache.state = (1, 2)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wrap_and_unwrap_model_attention() -> None:
    """Wrap and unwrap must restore the original __call__."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        wrap_model_attention,
        unwrap_model_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    # Create a minimal fake model
    class FakeAttn:
        def __call__(self, x, mask=None, cache=None):
            return x

    class FakeLayer:
        def __init__(self) -> None:
            self.self_attn = FakeAttn()

    class FakeModel:
        def __init__(self) -> None:
            self.layers = [FakeLayer(), FakeLayer()]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(2)
    ]

    original_func = model.layers[0].self_attn.__call__.__func__ if hasattr(model.layers[0].self_attn.__call__, '__func__') else model.layers[0].self_attn.__call__
    wrap_model_attention(model, caches)
    wrapped_func = model.layers[0].self_attn.__call__.__func__ if hasattr(model.layers[0].self_attn.__call__, '__func__') else model.layers[0].self_attn.__call__
    assert wrapped_func is not original_func

    unwrap_model_attention(model)
    restored_func = model.layers[0].self_attn.__call__.__func__ if hasattr(model.layers[0].self_attn.__call__, '__func__') else model.layers[0].self_attn.__call__
    assert restored_func is original_func


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wrap_model_mismatched_layer_count_raises() -> None:
    """Mismatch between model layers and caches must raise."""
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        wrap_model_attention,
    )
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)

    class FakeModel:
        def __init__(self) -> None:
            self.layers = [object() for _ in range(3)]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="3 layers but 2 caches"):
        wrap_model_attention(model, caches)
