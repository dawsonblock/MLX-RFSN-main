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
    """Wrap must intercept normal ``attn(...)`` calls; unwrap must restore original."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        _PackedAttentionWrapper,
        wrap_model_attention,
        unwrap_model_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    class FakeLinear:
        def __call__(self, x):
            return x

    class FakeRope:
        def __call__(self, x, offset=0):
            return x

    class FakeAttn:
        def __init__(self):
            self.n_heads = 2
            self.n_kv_heads = 2
            self.scale = 0.125
            self.q_proj = FakeLinear()
            self.k_proj = FakeLinear()
            self.v_proj = FakeLinear()
            self.o_proj = FakeLinear()
            self.rope = FakeRope()
            self.call_count = 0

        def __call__(self, x, mask=None, cache=None):
            self.call_count += 1
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

    original_attn = model.layers[0].self_attn
    original_call_count = original_attn.call_count

    wrap_model_attention(model, caches)

    # Wrap replaces the instance with a _PackedAttentionWrapper
    assert isinstance(model.layers[0].self_attn, _PackedAttentionWrapper)
    assert model.layers[0].self_attn._original is original_attn

    # Calling with normal ``attn(...)`` syntax must route through the wrapper,
    # NOT through the original FakeAttn.__call__.
    x = mx.random.normal(shape=(1, 2, 128)).astype(mx.float32)
    result = model.layers[0].self_attn(x, mask=None, cache=caches[0])

    # Original FakeAttn.__call__ should never have been invoked.
    assert original_attn.call_count == original_call_count

    # The cache should have received the 2 tokens from the call.
    assert caches[0].offset == 2
    assert caches[0].layer_cache.total_token_count() == 2

    # Result should be an MLX array (not the unmodified x)
    assert hasattr(result, "shape")

    unwrap_model_attention(model)

    # After unwrap, the original attention is restored.
    assert model.layers[0].self_attn is original_attn

    # Calling the restored original should now increment call_count.
    model.layers[0].self_attn(x, mask=None, cache=caches[0])
    assert original_attn.call_count == original_call_count + 1


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
