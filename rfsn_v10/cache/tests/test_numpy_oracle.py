"""Tests for the NumPy cache oracle.

These validate cache invariants without requiring MLX, so they can run on
any CI runner.  When MLX is available, an additional cross-check compares
the oracle against the real QuantizedLayerCache.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_identity_tensor(B: int, Hkv: int, T: int, D: int, layer_id: int = 0) -> np.ndarray:  # noqa: N803
    """Create a tensor where each head/token has a unique broadcast value."""
    base = layer_id * 1000
    vals = []
    for _b in range(B):
        for h in range(Hkv):
            for t in range(T):
                val = float(base + h * 100 + t)
                for _d in range(D):
                    vals.append(val)
    arr = np.array(vals, dtype=np.float32).reshape(B, Hkv, T, D)
    return arr


def test_append_once_and_stats() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=64)
    keys = np.random.randn(1, 2, 10, 64).astype(np.float32)
    values = np.random.randn(1, 2, 10, 64).astype(np.float32)
    cache.append(keys, values)

    assert cache.total_token_count() == 10
    assert cache.encoded_token_count == 0
    assert cache._stage_token_count == 10


def test_append_flushes_at_capacity() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32)
    for _ in range(4):
        keys = np.random.randn(1, 2, 10, 64).astype(np.float32)
        values = np.random.randn(1, 2, 10, 64).astype(np.float32)
        cache.append(keys, values)

    assert cache.encoded_token_count == 32
    assert cache._stage_token_count == 8
    assert len(cache._key_blocks) == 1


def test_fixed_size_blocks_after_large_prefill() -> None:
    """A 200-token prefill must yield fixed-size blocks, not one giant block."""
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=64)
    keys = np.random.randn(1, 2, 200, 64).astype(np.float32)
    values = np.random.randn(1, 2, 200, 64).astype(np.float32)
    cache.append(keys, values)

    assert cache.total_token_count() == 200
    assert cache.encoded_token_count == 192  # 3 * 64
    assert cache._stage_token_count == 8      # remainder
    assert len(cache._key_blocks) == 3
    for block in cache._key_blocks:
        assert block.shape[2] == 64


def test_multiple_appends_and_flush_preserves_identity() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32)
    B, Hkv, D = 1, 4, 64
    all_keys = []

    for i in range(5):
        keys = _make_identity_tensor(B, Hkv, 10, D, layer_id=i)
        cache.append(keys, keys)
        all_keys.append(keys)

    assert cache.total_token_count() == 50
    assert cache.encoded_token_count == 32
    assert cache._stage_token_count == 18
    assert len(cache._key_blocks) == 1

    full_k, _ = cache.reconstruct_dense()
    expected = np.concatenate(all_keys, axis=2)
    assert full_k.shape == (B, Hkv, 50, D)
    np.testing.assert_allclose(full_k, expected, atol=1e-5)


def test_trim_across_regions() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32, dense_residual_window=0)
    B, Hkv, D = 1, 2, 64

    keys1 = _make_identity_tensor(B, Hkv, 32, D, layer_id=0)
    cache.append(keys1, keys1)

    keys2 = _make_identity_tensor(B, Hkv, 8, D, layer_id=1)
    cache.append(keys2, keys2)

    assert cache.total_token_count() == 40

    cache.trim(32)
    assert cache.total_token_count() == 32
    assert cache.encoded_token_count == 32
    assert cache._stage_token_count == 0

    full_k, _ = cache.reconstruct_dense()
    np.testing.assert_allclose(full_k, keys1, atol=1e-5)

    cache.trim(0)
    assert cache.total_token_count() == 0


def test_partial_staging_trim() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=64, dense_residual_window=0)
    B, Hkv, D = 1, 2, 64

    keys1 = _make_identity_tensor(B, Hkv, 64, D, layer_id=0)
    cache.append(keys1, keys1)

    keys2 = _make_identity_tensor(B, Hkv, 12, D, layer_id=1)
    cache.append(keys2, keys2)

    assert cache.total_token_count() == 76
    assert cache.encoded_token_count == 64
    assert cache._stage_token_count == 12

    cache.trim(68)
    assert cache.total_token_count() == 68
    assert cache.encoded_token_count == 64
    assert cache._stage_token_count == 4

    stage_k, _stage_v, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_k.shape == (B, Hkv, 4, D)
    np.testing.assert_allclose(stage_k, keys2[:, :, :4, :], atol=1e-5)


def test_dense_residual_mutual_exclusion() -> None:
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=64, dense_residual_window=8)
    B, Hkv, D = 1, 2, 64

    keys = _make_identity_tensor(B, Hkv, 12, D, layer_id=0)
    cache.append(keys, keys)

    assert cache.total_token_count() == 12
    dense_k, _ = cache.get_dense_residual()
    assert dense_k is not None
    assert dense_k.shape == (B, Hkv, 8, D)
    np.testing.assert_allclose(dense_k, keys[:, :, 4:, :], atol=1e-5)

    stage_k, _, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_k.shape == (B, Hkv, 4, D)
    np.testing.assert_allclose(stage_k, keys[:, :, :4, :], atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-check against the real MLX cache when available
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_and_oracle_produce_same_structure() -> None:
    """Append the same data to both caches and verify identical token counts."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    mlx_cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32, dense_residual_window=0)
    np_cache = NumpyLayerCache(staging_capacity=32, dense_residual_window=0)

    for i in range(5):
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        mlx_cache.append(keys, values)
        np_cache.append(np.array(keys), np.array(values))

    assert mlx_cache.total_token_count() == np_cache.total_token_count()
    assert mlx_cache.encoded_token_count == np_cache.encoded_token_count
    assert mlx_cache.stats().staged_tokens == np_cache._stage_token_count
    assert mlx_cache.stats().sealed_blocks == len(np_cache._key_blocks)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_and_oracle_reconstruction_match() -> None:
    """Reconstructed dense shapes and token counts must match."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    mlx_cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32, dense_residual_window=0)
    np_cache = NumpyLayerCache(staging_capacity=32, dense_residual_window=0)

    for i in range(5):
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        mlx_cache.append(keys, values)
        np_cache.append(np.array(keys), np.array(values))

    # Reconstruct dense from MLX cache
    mlx_parts_k = []
    mlx_parts_v = []
    for kb in mlx_cache.iter_key_blocks():
        k_flat = k_codec.decode(kb)
        mlx_parts_k.append(k_flat.reshape(1, 2, kb.token_count, 64))
    for vb in mlx_cache.iter_value_blocks():
        v_flat = v_codec.decode(vb)
        mlx_parts_v.append(v_flat.reshape(1, 2, vb.token_count, 64))
    stage_k, stage_v, _ = mlx_cache.get_staging()
    if stage_k is not None:
        mlx_parts_k.append(stage_k)
        mlx_parts_v.append(stage_v)
    mlx_dense_k = mx.concatenate(mlx_parts_k, axis=2)

    np_dense_k, _ = np_cache.reconstruct_dense()

    assert tuple(mlx_dense_k.shape) == tuple(np_dense_k.shape)
    assert int(mlx_dense_k.shape[2]) == np_cache.total_token_count()


def test_fixed_size_block_packing() -> None:
    """All sealed blocks must be exactly staging_capacity tokens."""
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32)
    keys = np.random.randn(1, 2, 100, 64).astype(np.float32)
    cache.append(keys, keys)

    for block in cache._key_blocks:
        assert block.shape[2] == 32, f"Block size {block.shape[2]} != 32"
    assert cache._stage_token_count == 4


def test_numpy_attention_shape_and_finite() -> None:
    """numpy_attention must return the correct shape and finite values."""
    from rfsn_v10.cache.numpy_oracle import numpy_attention

    B, Hq, Hkv, Lq, T, D = 1, 4, 2, 1, 64, 64
    queries = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    keys = np.random.randn(B, Hkv, T, D).astype(np.float32)
    values = np.random.randn(B, Hkv, T, D).astype(np.float32)

    output = numpy_attention(queries, keys, values)
    assert output.shape == (B, Hq, Lq, D)
    assert np.all(np.isfinite(output))


def test_numpy_attention_matches_simple_computation() -> None:
    """numpy_attention must match a simple manual attention computation."""
    from rfsn_v10.cache.numpy_oracle import numpy_attention

    B, Hq, Hkv, Lq, T, D = 1, 2, 2, 1, 8, 4
    queries = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    keys = np.random.randn(B, Hkv, T, D).astype(np.float32)
    values = np.random.randn(B, Hkv, T, D).astype(np.float32)
    scale = D ** -0.5

    output = numpy_attention(queries, keys, values, scale=scale)

    # Manual computation
    scores = np.einsum("bhqd,bhtd->bhqt", queries, keys) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = np.einsum("bhqt,bhtd->bhqd", weights, values)

    np.testing.assert_allclose(output, expected, atol=1e-5)


def test_numpy_attention_matches_manual_computation() -> None:
    """NumpyLayerCache.numpy_attention must match a manual single-block attention."""
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=64)
    B, Hq, Hkv, T, D = 1, 8, 2, 16, 64
    queries = np.random.randn(B, Hq, 1, D).astype(np.float32)
    keys = np.random.randn(B, Hkv, T, D).astype(np.float32)
    values = np.random.randn(B, Hkv, T, D).astype(np.float32)
    cache.append(keys, values)

    out = cache.numpy_attention(queries, scale=D ** -0.5)
    assert out.shape == (B, Hq, 1, D)
    assert np.all(np.isfinite(out))

    # Compare against dense oracle
    k_exp = np.repeat(keys, Hq // Hkv, axis=1)
    v_exp = np.repeat(values, Hq // Hkv, axis=1)
    scores = np.matmul(queries, k_exp.transpose(0, 1, 3, 2)) * (D ** -0.5)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores.astype(np.float32) - max_scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = np.matmul(weights, v_exp)
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_numpy_attention_multi_block_matches_dense() -> None:
    """numpy_attention across multiple blocks must match dense reference."""
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32)
    B, Hq, Hkv, D = 1, 8, 2, 64
    scale = D ** -0.5

    all_keys = []
    all_values = []
    for _ in range(3):
        keys = np.random.randn(B, Hkv, 20, D).astype(np.float32)
        values = np.random.randn(B, Hkv, 20, D).astype(np.float32)
        cache.append(keys, values)
        all_keys.append(keys)
        all_values.append(values)

    queries = np.random.randn(B, Hq, 1, D).astype(np.float32)
    out = cache.numpy_attention(queries, scale=scale)

    # Dense reference
    full_k = np.concatenate(all_keys, axis=2)
    full_v = np.concatenate(all_values, axis=2)
    k_exp = np.repeat(full_k, Hq // Hkv, axis=1)
    v_exp = np.repeat(full_v, Hq // Hkv, axis=1)
    scores = np.matmul(queries, k_exp.transpose(0, 1, 3, 2)) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores.astype(np.float32) - max_scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = np.matmul(weights, v_exp)
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_numpy_oracle_sealed_blocks_are_exactly_staging_capacity() -> None:
    """After flush, every sealed block must be exactly staging_capacity tokens."""
    from rfsn_v10.cache.numpy_oracle import NumpyLayerCache

    cache = NumpyLayerCache(staging_capacity=32)
    for _ in range(6):
        cache.append(
            np.random.randn(1, 2, 16, 64).astype(np.float32),
            np.random.randn(1, 2, 16, 64).astype(np.float32),
        )

    for kb in cache.iter_key_blocks():
        assert kb.shape[2] == 32, f"Expected 32, got {kb.shape[2]}"
