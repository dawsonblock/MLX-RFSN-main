"""Tests for QuantizedLayerCache — append-only, never recompresses."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_once_and_stats() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    cache.append(keys, values)

    assert cache.total_token_count() == 10
    assert cache.encoded_token_count == 0  # not flushed yet
    assert cache.stats().staged_tokens == 10


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_flushes_at_capacity() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    # Append 40 tokens → staging reaches 40 (>= 32).
    # Fixed-size flush encodes one 32-token block and keeps 8 in staging.
    for _ in range(4):
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        cache.append(keys, values)

    assert cache.encoded_token_count == 32
    assert cache.stats().staged_tokens == 8
    assert cache.stats().sealed_blocks == 1
    assert cache.requantized_token_count == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_1024_token_append_no_requantize() -> None:
    """Phase 3 exit condition: 1024 tokens, 0 requantized."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    for _ in range(1024):
        keys = mx.random.normal(shape=(1, 2, 1, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 1, 64)).astype(mx.float32)
        cache.append(keys, values)

    assert cache.total_token_count() == 1024
    assert cache.requantized_token_count == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_memory_grows_linearly() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    # Append 512 tokens in batches of 64
    for _ in range(8):
        keys = mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float32)
        cache.append(keys, values)

    payload = cache.payload_bytes()
    assert payload > 0
    # After flushing, staging should be empty
    assert cache.stats().staged_tokens == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_dense_residual_bounded() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64, dense_residual_window=16)

    for _ in range(10):
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        cache.append(keys, values)

    dense_k, dense_v = cache.get_dense_residual()
    assert dense_k is not None
    assert dense_k.shape[2] <= 16, f"Dense residual window exceeded: {dense_k.shape[2]}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reset_clears_all_state() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    keys = mx.random.normal(shape=(1, 2, 100, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 100, 64)).astype(mx.float32)
    cache.append(keys, values)

    cache.reset()
    assert cache.total_token_count() == 0
    assert cache.payload_bytes() == 0
