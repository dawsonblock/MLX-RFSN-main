"""Tests for the consolidated MLX reference packed-attention engine.

Validates against:
  * NumPy packed-attention oracle
  * Existing BlockwiseReferenceAttention
  * Dense attention baseline
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_reference_matches_numpy_oracle_single_block() -> None:
    """MLX packed reference must match NumPy packed oracle for one block."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.numpy_attention_oracle import numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    np.random.seed(42)
    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2
    T = 64

    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)
    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    # Build cache with one block
    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)
    cache.append(mx.array(k), mx.array(v))

    # MLX reference
    mlx_out, _ = attend(mx.array(q), cache, scale=D ** -0.5, query_start_pos=Lq)

    # NumPy oracle
    np_k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    np_v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    k_blocks = list(cache.iter_key_blocks())
    v_blocks = list(cache.iter_value_blocks())

    np_out = numpy_packed_attention(
        q, k_blocks, v_blocks, np_k_codec, np_v_codec,
        scale=D ** -0.5, query_start_pos=Lq,
    )

    np.testing.assert_allclose(np.array(mlx_out), np_out, atol=0.05, rtol=0.01)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_reference_matches_numpy_oracle_multiple_blocks() -> None:
    """MLX packed reference must match NumPy packed oracle for multiple blocks."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.numpy_attention_oracle import numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    np.random.seed(43)
    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)

    # Append two 64-token blocks
    for i in range(2):
        k = np.random.randn(B, Hkv, 64, D).astype(np.float32)
        v = np.random.randn(B, Hkv, 64, D).astype(np.float32)
        cache.append(mx.array(k), mx.array(v))

    mlx_out, _ = attend(mx.array(q), cache, scale=D ** -0.5, query_start_pos=Lq)

    np_k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    np_v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    k_blocks = list(cache.iter_key_blocks())
    v_blocks = list(cache.iter_value_blocks())

    np_out = numpy_packed_attention(
        q, k_blocks, v_blocks, np_k_codec, np_v_codec,
        scale=D ** -0.5, query_start_pos=Lq,
    )

    np.testing.assert_allclose(np.array(mlx_out), np_out, atol=0.05, rtol=0.01)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_reference_matches_blockwise_reference() -> None:
    """New consolidated reference must match existing BlockwiseReferenceAttention."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.reference_attention import BlockwiseReferenceAttention

    np.random.seed(44)
    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2

    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    q = mx.array(np.random.randn(B, Hq, Lq, D).astype(np.float32))

    for i in range(2):
        k = mx.array(np.random.randn(B, Hkv, 64, D).astype(np.float32))
        v = mx.array(np.random.randn(B, Hkv, 64, D).astype(np.float32))
        cache.append(k, v)

    # Add some staging
    k_stage = mx.array(np.random.randn(B, Hkv, 10, D).astype(np.float32))
    v_stage = mx.array(np.random.randn(B, Hkv, 10, D).astype(np.float32))
    cache.append(k_stage, v_stage)

    new_out, _ = attend(q, cache, scale=D ** -0.5, query_start_pos=Lq)

    old_attn = BlockwiseReferenceAttention(k_codec, v_codec, scale=D ** -0.5)
    old_out, _ = old_attn.attend(q, cache)

    np.testing.assert_allclose(np.array(new_out), np.array(old_out), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_reference_invalid_gqa() -> None:
    """Invalid GQA ratio must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    k = mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float32)
    v = mx.random.normal(shape=(1, 2, 64, 64)).astype(mx.float32)
    cache.append(k, v)

    q = mx.random.normal(shape=(1, 3, 4, 64)).astype(mx.float32)  # Hq=3, not divisible by Hkv=2

    with pytest.raises(ValueError, match="must be divisible"):
        attend(q, cache, scale=1.0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_mlx_reference_empty_cache_raises() -> None:
    """Empty cache must raise ValueError."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    q = mx.random.normal(shape=(1, 2, 4, 64)).astype(mx.float32)

    with pytest.raises(ValueError, match="cache is empty"):
        attend(q, cache, scale=1.0)
