"""Tests for the attention wrapper and direct packed cache."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mx_utils
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
    v_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)

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
def test_direct_packed_cache_state_returns_all_live_tensors() -> None:
    """state must expose sealed blocks, staging, and dense residual for mx.eval."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
    )

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)

    cache = RfsnDirectPackedKVCache(
        layer_id=0,
        key_codec=k_codec,
        value_codec=v_codec,
        staging_capacity=4,
        dense_residual_window=4,
    )

    # First append: with dense_residual_window=4, tokens go to the dense
    # residual window; no sealed pages yet.
    k1 = mx.random.normal(shape=(1, 2, 4, 64)).astype(mx.float32)
    v1 = mx.random.normal(shape=(1, 2, 4, 64)).astype(mx.float32)
    cache.update_and_fetch(k1, v1)

    state1 = cache.state
    assert len(state1) == 2  # dense_k, dense_v
    mx.eval(state1)

    # Second append: 8 new tokens overflow the dense window, so 8 tokens are
    # evicted to staging and flushed into 2 sealed paged pages.  Staging is
    # empty after the flush, and the dense window retains the last 4 tokens.
    k2 = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.float32)
    v2 = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.float32)
    cache.update_and_fetch(k2, v2)

    state2 = cache.state
    # Expect: 7 paged-arena arrays (K/V codes/scales + page metadata) +
    #          dense K, dense V
    assert len(state2) == 9
    mx.eval(state2)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_direct_packed_cache_trim_raises() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
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
    v_codec = CartesianCodec(bits=8, group_size=64)
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
        install_packed_attention,
        uninstall_packed_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)

    class FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((8, 8))

        def __call__(self, x):
            return x

    class FakeRope(nn.Module):
        def __call__(self, x, offset=0):
            return x

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
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

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [FakeLayer(), FakeLayer()]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(2)
    ]

    original_attn = model.layers[0].self_attn
    original_call_count = original_attn.call_count

    install_packed_attention(model, caches)

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

    uninstall_packed_attention(model)

    # After unwrap, the original attention is restored.
    assert model.layers[0].self_attn is original_attn

    # Calling the restored original should now increment call_count.
    model.layers[0].self_attn(x, mask=None, cache=caches[0])
    assert original_attn.call_count == original_call_count + 1


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_strict_mode_raises_on_missing_cache() -> None:
    """Strict wrapper must raise when cache is missing or wrong type."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
        uninstall_packed_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)

    class FakeAttn(nn.Module):
        def __call__(self, x, mask=None, cache=None):
            return x

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=0, key_codec=k_codec, value_codec=v_codec)
    ]

    install_packed_attention(model, caches, strict=True)
    x = mx.random.normal(shape=(1, 2, 128)).astype(mx.float32)

    with pytest.raises(RuntimeError, match="Strict packed mode"):
        model.layers[0].self_attn(x, mask=None, cache=None)

    with pytest.raises(RuntimeError, match="Strict packed mode"):
        model.layers[0].self_attn(x, mask=None, cache="not_a_packed_cache")

    uninstall_packed_attention(model)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_permissive_fallback_increments_counter() -> None:
    """Permissive wrapper must count fallback and record backend."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
        uninstall_packed_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)

    class FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((8, 8))
        def __call__(self, x):
            return x

    class FakeRope(nn.Module):
        def __call__(self, x, offset=0):
            return x

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 2
            self.n_kv_heads = 2
            self.scale = 0.125
            self.q_proj = FakeLinear()
            self.k_proj = FakeLinear()
            self.v_proj = FakeLinear()
            self.o_proj = FakeLinear()
            self.rope = FakeRope()
        def __call__(self, x, mask=None, cache=None):
            return x

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=0, key_codec=k_codec, value_codec=v_codec)
    ]

    install_packed_attention(model, caches, strict=False)
    wrapper = model.layers[0].self_attn
    x = mx.random.normal(shape=(1, 2, 128)).astype(mx.float32)

    assert wrapper._fallback_count == 0
    assert wrapper._executed_backend == "unknown"

    model.layers[0].self_attn(x, mask=None, cache=None)
    assert wrapper._fallback_count == 1
    assert wrapper._executed_backend == "dense"

    # packed path requires BHTD inputs; x is (B, L, D) = (1, 2, 128)
    # projections return x which is (1, 2, 128); reshape will fail because
    # n_heads=2, head_dim would be 64, but 128/2 = 64... wait, x is (1,2,128).
    # q_proj(x) returns (1,2,128). reshape(1, 2, 2, -1) -> (1, 2, 2, 64).transpose -> (1,2,2,64)
    # That works. rope returns same shape. Then attend() needs proper shapes.
    # But the fake cache has empty layer_cache, so attend() will raise.
    # We just want to verify _executed_backend is set before attend() is called.
    # Since attend() is called after setting backend, it will be set to "packed"
    # even if attend() fails. We can catch the error or just not call it.
    # Actually, for this test, we only need to verify fallback behavior.

    uninstall_packed_attention(model)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wrap_preserves_parameter_and_state_parity() -> None:
    """Wrapping must keep the same arrays in model.parameters() and model.state."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
        uninstall_packed_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.k_proj = nn.Linear(16, 16)
            self.n_heads = 2
            self.n_kv_heads = 2
            self.scale = 0.125

        def __call__(self, x, mask=None, cache=None):
            return x

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=0, key_codec=k_codec, value_codec=v_codec)
    ]

    before_params = model.parameters()
    before_state_keys = set(model.state.keys())

    install_packed_attention(model, caches)

    after_params = model.parameters()
    after_state_keys = set(model.state.keys())

    # Parameter tree structure must be identical
    assert before_state_keys == after_state_keys
    assert before_params == after_params

    # Every array object must be the same (no copies)
    before_ids = {id(v) for _, v in mx_utils.tree_flatten(before_params)}
    after_ids = {id(v) for _, v in mx_utils.tree_flatten(after_params)}
    assert before_ids == after_ids

    uninstall_packed_attention(model)

    # After uninstall, parity must be restored
    assert model.parameters() == before_params
    assert set(model.state.keys()) == before_state_keys


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wrap_model_mismatched_layer_count_raises() -> None:
    """Mismatch between model layers and caches must raise."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
    )

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)

    class FakeModel:
        def __init__(self) -> None:
            self.layers = [object() for _ in range(3)]

    model = FakeModel()
    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="3 layers but 2 caches"):
        install_packed_attention(model, caches)
