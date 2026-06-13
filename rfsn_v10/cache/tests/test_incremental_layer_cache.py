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


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_logical_start_increments_correctly_across_flushes() -> None:
    """Block logical_start must be monotonic and correctly offset across flushes."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    # First append: 48 tokens → one 32-token block, 16 staged
    cache.append(
        mx.random.normal(shape=(1, 2, 48, 64)).astype(mx.float32),
        mx.random.normal(shape=(1, 2, 48, 64)).astype(mx.float32),
    )
    assert cache.encoded_token_count == 32
    assert cache.stats().staged_tokens == 16

    # Second append: 32 tokens → staged(16) + new(32) = 48 → another 32-token block, 16 staged
    cache.append(
        mx.random.normal(shape=(1, 2, 32, 64)).astype(mx.float32),
        mx.random.normal(shape=(1, 2, 32, 64)).astype(mx.float32),
    )
    assert cache.encoded_token_count == 64
    assert cache.stats().staged_tokens == 16

    # Verify logical_start values
    logical_starts = [b.logical_start for b in cache.iter_key_blocks()]
    assert logical_starts == [0, 32], f"Unexpected logical_starts: {logical_starts}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_trim_disabled_raises() -> None:
    """trim() must raise NotImplementedError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    cache.append(keys, values)

    with pytest.raises(NotImplementedError):
        cache.trim(5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_geometry_frozen_after_first_append() -> None:
    """Appending mismatched geometry must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    cache.append(
        mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32),
        mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32),
    )

    with pytest.raises(ValueError):
        cache.append(
            mx.random.normal(shape=(1, 4, 10, 64)).astype(mx.float32),  # Hkv mismatch
            mx.random.normal(shape=(1, 4, 10, 64)).astype(mx.float32),
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_shape_mismatch_raises() -> None:
    """Mismatched keys/values shapes must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    with pytest.raises(ValueError, match="shape"):
        cache.append(
            mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32),
            mx.random.normal(shape=(1, 2, 10, 32)).astype(mx.float32),
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_rank_not_4_raises() -> None:
    """Rank != 4 must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    with pytest.raises(ValueError, match="keys must have shape"):
        cache.append(
            mx.random.normal(shape=(1, 2, 10)).astype(mx.float32),
            mx.random.normal(shape=(1, 2, 10)).astype(mx.float32),
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_zero_tokens_raises() -> None:
    """Zero new tokens must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    with pytest.raises(ValueError, match="new token count must be positive"):
        cache.append(
            mx.zeros((1, 2, 0, 64), dtype=mx.float32),
            mx.zeros((1, 2, 0, 64), dtype=mx.float32),
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_block_positions_are_monotonic_and_contiguous() -> None:
    """After multiple flushes, block positions must be contiguous."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.contracts import validate_block_positions
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    for _ in range(6):
        cache.append(
            mx.random.normal(shape=(1, 2, 16, 64)).astype(mx.float32),
            mx.random.normal(shape=(1, 2, 16, 64)).astype(mx.float32),
        )

    key_blocks = list(cache.iter_key_blocks())
    validate_block_positions(key_blocks)
    logical_starts = [b.logical_start for b in key_blocks]
    assert logical_starts == [0, 32, 64], f"Unexpected positions: {logical_starts}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_destroy_prevents_reuse() -> None:
    """destroy() must make all public methods raise RuntimeError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
    cache.append(keys, values)

    cache.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        cache.append(keys, values)
    with pytest.raises(RuntimeError, match="destroyed"):
        cache.total_token_count()
    with pytest.raises(RuntimeError, match="destroyed"):
        cache.payload_bytes()
    with pytest.raises(RuntimeError, match="destroyed"):
        cache.stats()
    with pytest.raises(RuntimeError, match="destroyed"):
        cache.trim(5)
    with pytest.raises(RuntimeError, match="destroyed"):
        list(cache.iter_key_blocks())


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reset_clears_geometry() -> None:
    """reset() must allow reuse with different geometry."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    cache.append(
        mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32),
        mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32),
    )
    cache.reset()
    # Should not raise geometry mismatch
    cache.append(
        mx.random.normal(shape=(1, 4, 10, 64)).astype(mx.float32),
        mx.random.normal(shape=(1, 4, 10, 64)).astype(mx.float32),
    )
    assert cache.total_token_count() == 10


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_unsupported_dtype_raises() -> None:
    """Unsupported dtypes must raise TypeError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    with pytest.raises(TypeError, match="unsupported key dtype"):
        cache.append(
            mx.zeros((1, 2, 10, 64), dtype=mx.int32),
            mx.zeros((1, 2, 10, 64), dtype=mx.float32),
        )
    with pytest.raises(TypeError, match="unsupported value dtype"):
        cache.append(
            mx.zeros((1, 2, 10, 64), dtype=mx.float32),
            mx.zeros((1, 2, 10, 64), dtype=mx.int32),
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_nan_inf_raises() -> None:
    """NaN or Inf values must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    bad_keys = mx.zeros((1, 2, 10, 64), dtype=mx.float32)
    bad_keys[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        cache.append(bad_keys, mx.zeros((1, 2, 10, 64), dtype=mx.float32))

    bad_values = mx.zeros((1, 2, 10, 64), dtype=mx.float32)
    bad_values[0, 0, 0, 0] = float("inf")

    with pytest.raises(ValueError, match="Inf"):
        cache.append(mx.zeros((1, 2, 10, 64), dtype=mx.float32), bad_values)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_incompatible_head_dim_raises() -> None:
    """head_dim not divisible by group_size must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    with pytest.raises(ValueError, match="incompatible"):
        cache.append(
            mx.zeros((1, 2, 10, 32), dtype=mx.float32),
            mx.zeros((1, 2, 10, 32), dtype=mx.float32),
        )
