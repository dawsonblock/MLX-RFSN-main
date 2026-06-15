"""Tests for the NumPy packed-codec oracle.

Validates that NumPy and MLX produce identical V4 blocks and round-trip data.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def test_numpy_codec_k8_exact_shape() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    x = np.random.randn(1, 2, 64, 64).astype(np.float32)
    block = codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="K")

    assert block.bits == 8
    assert block.codes_per_word == 4
    assert block.words_per_vector == 16
    assert block.packed_codes.shape == (1, 2, 64, 16)
    assert block.scales.shape == (1, 2, 64, 1)
    block.validate()


def test_numpy_codec_v5_exact_shape() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    x = np.random.randn(1, 2, 64, 64).astype(np.float32)
    block = codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="V")

    assert block.bits == 5
    assert block.codes_per_word == 6
    assert block.words_per_vector == 11
    assert block.packed_codes.shape == (1, 2, 64, 11)
    assert block.scales.shape == (1, 2, 64, 1)
    block.validate()


def test_numpy_codec_v5_vector_boundary_isolation() -> None:
    """Adjacent vectors must not leak codes across boundaries."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    B, H, T, D = 1, 1, 2, 64
    x = np.random.randn(B, H, T, D).astype(np.float32)
    block = codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="V")

    # Unpack and verify each vector independently
    unpacked = codec.decode_bhtd(block)
    # Round-trip should be close (quantization is lossy)
    np.testing.assert_allclose(unpacked, x, atol=0.15, rtol=0.01)


def test_numpy_codec_roundtrip_k8() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    x = np.random.randn(1, 2, 64, 64).astype(np.float32)
    block = codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="K")
    decoded = codec.decode_bhtd(block)

    # K8 quantization with WHT should be fairly accurate
    np.testing.assert_allclose(decoded, x, atol=0.05, rtol=0.01)


def test_numpy_codec_roundtrip_v5() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    x = np.random.randn(1, 2, 64, 64).astype(np.float32)
    block = codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="V")
    decoded = codec.decode_bhtd(block)

    # V5 is coarser
    np.testing.assert_allclose(decoded, x, atol=0.30, rtol=0.05)


def test_numpy_codec_wht_self_inverse() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import _numpy_wht64

    x = np.random.randn(2, 3, 64).astype(np.float32)
    h1 = _numpy_wht64(x)
    h2 = _numpy_wht64(h1)
    np.testing.assert_allclose(h2, x, atol=1e-5, rtol=1e-5)


def test_numpy_codec_sign_determinism() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import _numpy_hash_signs

    x = np.random.randn(2, 3, 64).astype(np.float32)
    s1 = _numpy_hash_signs(x, seed=42)
    s2 = _numpy_hash_signs(x, seed=42)
    np.testing.assert_array_equal(s1, s2)

    s3 = _numpy_hash_signs(x, seed=43)
    assert not np.allclose(s1, s3)


def test_numpy_codec_different_key_value_group_sizes() -> None:
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    x = np.random.randn(1, 2, 32, 64).astype(np.float32)
    k_block = k_codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="K")
    v_block = v_codec.encode_bhtd(x, logical_start=0, layer_id=0, stream_id="V")

    assert k_block.group_size == 64
    assert v_block.group_size == 64
    assert k_block.bits == 8
    assert v_block.bits == 5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_numpy_vs_mlx_k8_parity() -> None:
    """NumPy and MLX must produce identical packed uint32 words and scales for K8."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec as MlxCodec
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    np.random.seed(123)
    x_np = np.random.randn(1, 2, 64, 64).astype(np.float32)
    x_mlx = mx.array(x_np)

    np_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    mlx_codec = MlxCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)

    np_block = np_codec.encode_bhtd(x_np, logical_start=0, layer_id=0, stream_id="K")
    mlx_block = mlx_codec.encode_bhtd(x_mlx, logical_start=0, layer_id=0, stream_id="K")

    np.testing.assert_array_equal(np_block.packed_codes, np.array(mlx_block.packed_codes))
    np.testing.assert_allclose(np_block.scales, np.array(mlx_block.scales), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_numpy_vs_mlx_v5_parity() -> None:
    """NumPy and MLX must produce identical packed uint32 words and scales for V5."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec as MlxCodec
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    np.random.seed(456)
    x_np = np.random.randn(1, 2, 64, 64).astype(np.float32)
    x_mlx = mx.array(x_np)

    np_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    mlx_codec = MlxCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    np_block = np_codec.encode_bhtd(x_np, logical_start=0, layer_id=0, stream_id="V")
    mlx_block = mlx_codec.encode_bhtd(x_mlx, logical_start=0, layer_id=0, stream_id="V")

    np.testing.assert_array_equal(np_block.packed_codes, np.array(mlx_block.packed_codes))
    np.testing.assert_allclose(np_block.scales, np.array(mlx_block.scales), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_numpy_vs_mlx_decode_parity() -> None:
    """NumPy decode must match MLX decode for the same block."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec as MlxCodec
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    np.random.seed(789)
    x_np = np.random.randn(1, 2, 64, 64).astype(np.float32)
    x_mlx = mx.array(x_np)

    np_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    mlx_codec = MlxCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)

    mlx_block = mlx_codec.encode_bhtd(x_mlx, logical_start=0, layer_id=0, stream_id="K")

    # Decode with both
    mlx_decoded = np.array(mlx_codec.decode_bhtd(mlx_block))
    # Convert MLX arrays in the block to NumPy for the NumPy decoder
    np_block = mlx_block
    if hasattr(np_block.packed_codes, "astype") and str(type(np_block.packed_codes)).startswith("<class 'mlx"):
        np_block = np_block.__class__(
            packed_codes=np.array(np_block.packed_codes),
            scales=np.array(np_block.scales),
            format_version=np_block.format_version,
            tensor_layout=np_block.tensor_layout,
            packing_layout=np_block.packing_layout,
            scale_layout=np_block.scale_layout,
            preconditioner=np_block.preconditioner,
            batch_size=np_block.batch_size,
            n_kv_heads=np_block.n_kv_heads,
            token_count=np_block.token_count,
            head_dim=np_block.head_dim,
            logical_start=np_block.logical_start,
            logical_end=np_block.logical_end,
            bits=np_block.bits,
            group_size=np_block.group_size,
            groups_per_vector=np_block.groups_per_vector,
            codes_per_word=np_block.codes_per_word,
            words_per_vector=np_block.words_per_vector,
            original_value_count=np_block.original_value_count,
            padded_value_count=np_block.padded_value_count,
            original_dtype=np_block.original_dtype,
            sign_seed=np_block.sign_seed,
            sign_algorithm=np_block.sign_algorithm,
            layer_id=np_block.layer_id,
            stream_id=np_block.stream_id,
        )
    np_decoded = np_codec.decode_bhtd(np_block)

    np.testing.assert_allclose(mlx_decoded, np_decoded, atol=1e-5, rtol=1e-5)
