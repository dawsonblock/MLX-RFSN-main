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


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_hash_signs_different_seeds() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.ones((4, 64))
    y1 = CartesianCodec.apply_hash_signs(x, seed=42)
    y2 = CartesianCodec.apply_hash_signs(x, seed=43)
    # Different seeds should almost certainly produce different signs
    assert not mx.all(y1 == y2).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_hash_signs_values_are_only_plus_minus_one() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.arange(64).astype(mx.float32).reshape(4, 16)
    y = CartesianCodec.apply_hash_signs(x, seed=7)
    # Every element should be either +value or -value
    # Use where to avoid boolean indexing
    safe_x = mx.where(x == 0, 1.0, x)
    ratio = y / safe_x
    is_plus_one = mx.abs(ratio - 1.0) < 1e-5
    is_minus_one = mx.abs(ratio + 1.0) < 1e-5
    valid = mx.where(x == 0, True, mx.logical_or(is_plus_one, is_minus_one))
    assert mx.all(valid).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_reference_preserves_shape() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.random.normal(shape=(3, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    assert y.shape == x.shape


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_double_transform_approximates_identity() -> None:
    """WHT(WHT(x)) ≈ n * x for unnormalised transform."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.random.normal(shape=(2, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    z = CartesianCodec.apply_wht(y)
    # For unnormalised WHT, WHT(WHT(x)) = n * x
    n = x.shape[-1]
    expected = x * n
    max_err = mx.max(mx.abs(z - expected)).item()
    assert max_err < 1e-3, f"WHT(WHT(x)) max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_orthogonality_constant_signal() -> None:
    """A constant signal should have energy only in the first bin."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.ones((1, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    # First element should be large (sum of all ones)
    assert y[0, 0].item() == pytest.approx(64.0, abs=1e-4)
    # Remaining elements should be zero (differences cancel)
    assert mx.max(mx.abs(y[0, 1:])).item() < 1e-4
