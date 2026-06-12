"""Tests for BlockwiseReferenceAttention — bounded-memory blockwise attention."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_blockwise_matches_dense_attention() -> None:
    """Blockwise attention output should match dense attention for same K/V."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.reference_attention import BlockwiseReferenceAttention

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    B, Hq, Hkv, T, D = 1, 4, 2, 64, 64
    keys = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, T, D)).astype(mx.float32)
    cache.append(keys, values)
    cache._flush_staging()

    queries = mx.random.normal(shape=(B, Hq, 1, D)).astype(mx.float32)
    scale = D ** -0.5

    attn = BlockwiseReferenceAttention(k_codec, v_codec, scale=scale)
    output, scratch = attn.attend(queries, cache)

    # Dense oracle
    k_exp = mx.repeat(keys, Hq // Hkv, axis=1)
    v_exp = mx.repeat(values, Hq // Hkv, axis=1)
    scores = mx.matmul(queries, k_exp.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)
    oracle = mx.matmul(weights, v_exp)

    diff = mx.max(mx.abs(output - oracle)).item()
    assert diff < 0.5, f"Blockwise diverged from dense: max_diff={diff}"

    # Scratch memory should be bounded (max block size <= 64 in this test)
    assert scratch.max_reconstructed_block_tokens <= 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_blockwise_with_staging_and_dense_residual() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.reference_attention import BlockwiseReferenceAttention

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32, dense_residual_window=16)

    B, Hq, Hkv, D = 1, 4, 2, 64
    # Add 48 tokens: 32 sealed + 16 staged
    keys = mx.random.normal(shape=(B, Hkv, 48, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 48, D)).astype(mx.float32)
    cache.append(keys, values)

    queries = mx.random.normal(shape=(B, Hq, 1, D)).astype(mx.float32)
    scale = D ** -0.5

    attn = BlockwiseReferenceAttention(k_codec, v_codec, scale=scale)
    output, scratch = attn.attend(queries, cache)

    assert output.shape == (B, Hq, 1, D)
    assert mx.all(mx.isfinite(output)).item()
    assert scratch.max_reconstructed_block_tokens <= 48
