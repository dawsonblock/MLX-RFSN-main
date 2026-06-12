"""Tests for CartesianCodec — stateless K8/V5 codec."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_k8_encode_decode_roundtrip() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    x = mx.random.normal(shape=(128, 64)).astype(mx.float32)

    block = codec.encode(x)
    decoded = codec.decode(block)

    # Slice back to original size (padding removed)
    decoded_reshaped = decoded.reshape(x.shape)
    max_err = mx.max(mx.abs(x - decoded_reshaped)).item()
    assert max_err < 0.5, f"K8 roundtrip max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_v5_encode_decode_roundtrip() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=5, group_size=64)
    x = mx.random.normal(shape=(64, 64)).astype(mx.float32)

    block = codec.encode(x)
    decoded = codec.decode(block)
    decoded_reshaped = decoded.reshape(x.shape)

    max_err = mx.max(mx.abs(x - decoded_reshaped)).item()
    # 5-bit has more error than 8-bit
    assert max_err < 2.0, f"V5 roundtrip max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_payload_bytes_matches_estimate() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    x = mx.random.normal(shape=(128, 64)).astype(mx.float32)

    block = codec.encode(x)
    actual = block.payload_bytes()
    estimated = codec.estimate_bytes(block)
    assert actual == estimated, f"actual={actual} != estimated={estimated}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_estimate_bytes_for_shape_matches_actual() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    shape = (128, 64)
    x = mx.random.normal(shape=shape).astype(mx.float32)

    block = codec.encode(x)
    actual = block.payload_bytes()
    estimated = codec.estimate_bytes_for_shape(shape)
    assert actual == estimated, f"actual={actual} != estimated={estimated}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_reference_finite() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.random.normal(shape=(4, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    assert mx.all(mx.isfinite(y)).item()
    assert y.shape == x.shape


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_hash_signs_deterministic() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.ones((4, 64))
    y1 = CartesianCodec.apply_hash_signs(x, seed=42)
    y2 = CartesianCodec.apply_hash_signs(x, seed=42)
    assert mx.all(y1 == y2).item()
