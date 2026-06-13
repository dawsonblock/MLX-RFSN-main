"""Tests for CartesianCodec — stateless K8/V5 codec."""
from __future__ import annotations

import math

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
    """WHT(WHT(x)) ≈ x for orthonormal transform."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.random.normal(shape=(2, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    z = CartesianCodec.apply_wht(y)
    # For orthonormal WHT, WHT(WHT(x)) = x
    max_err = mx.max(mx.abs(z - x)).item()
    assert max_err < 1e-3, f"WHT(WHT(x)) max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_orthogonality_constant_signal() -> None:
    """A constant signal should have energy only in the first bin."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    x = mx.ones((1, 64)).astype(mx.float32)
    y = CartesianCodec.apply_wht(x)
    # First element should be sum of all ones / sqrt(n)
    n = x.shape[-1]
    assert y[0, 0].item() == pytest.approx(n / math.sqrt(n), abs=1e-4)
    # Remaining elements should be zero (differences cancel)
    assert mx.max(mx.abs(y[0, 1:])).item() < 1e-4


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_wht_sign_roundtrip() -> None:
    """Encode/decode with WHT and signs must roundtrip within quant error."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    x = mx.random.normal(shape=(128, 64)).astype(mx.float32)

    block = codec.encode(x)
    assert block.wht_applied is True
    assert block.sign_seed == 42

    decoded = codec.decode(block)
    decoded_reshaped = decoded.reshape(x.shape)
    max_err = mx.max(mx.abs(x - decoded_reshaped)).item()
    assert max_err < 0.5, f"WHT+sign roundtrip max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_decode_trims_group_padding() -> None:
    """V2 decode must trim padding elements so size matches num_elements."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    # 65 elements forces 63 pad elements to reach 128 (next multiple of 64)
    x = mx.random.normal(shape=(65,)).astype(mx.float32)

    block = codec.encode(x)
    assert block.num_elements == 65
    assert block.n_values == 128  # padded to group_size

    decoded = codec.decode(block)
    assert int(decoded.size) == 65, f"Expected 65, got {decoded.size}"
    assert mx.max(mx.abs(decoded - x)).item() < 0.5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_decode_restores_original_dtype() -> None:
    """V2 decode must restore the original dtype from PackedBlock metadata."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    x = mx.random.normal(shape=(128, 64)).astype(mx.float16)

    block = codec.encode(x)
    assert block.original_dtype == "float16"

    decoded = codec.decode(block)
    assert decoded.dtype == mx.float16, f"Expected float16, got {decoded.dtype}"
    decoded_reshaped = decoded.reshape(x.shape)
    max_err = mx.max(mx.abs(x.astype(mx.float32) - decoded_reshaped.astype(mx.float32))).item()
    assert max_err < 0.5, f"float16 roundtrip max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_decode_restores_dtype_and_trims_padding() -> None:
    """V2 decode must restore dtype AND trim padding simultaneously."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    # 65 elements → 63 pad → 128 total; not a multiple of 64
    x = mx.random.normal(shape=(65,)).astype(mx.float16)

    block = codec.encode(x)
    assert block.original_dtype == "float16"
    assert block.num_elements == 65

    decoded = codec.decode(block)
    assert decoded.dtype == mx.float16, f"Expected float16, got {decoded.dtype}"
    assert int(decoded.size) == 65, f"Expected 65, got {decoded.size}"
    assert mx.max(mx.abs(decoded.astype(mx.float32) - x.astype(mx.float32))).item() < 0.5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_decode_v1_block_no_dtype_restore() -> None:
    """V1 blocks must not have dtype restored (num_elements==0, version==1)."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    codec = CartesianCodec(bits=8, group_size=64)
    x = mx.random.normal(shape=(128, 64)).astype(mx.float32)
    block_v2 = codec.encode(x)

    # Simulate a V1 block by resetting format_version and num_elements
    import dataclasses
    block_v1 = dataclasses.replace(
        block_v2,
        format_version=1,
        num_elements=0,
        original_dtype="float16",  # V1 default
    )

    decoded = codec.decode(block_v1)
    # V1 block should return float32 (no dtype restoration), not float16
    assert decoded.dtype == mx.float32, f"V1 block should stay float32, got {decoded.dtype}"
    assert int(decoded.size) == 128 * 64  # no trimming for V1


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_v3_block_validate_rejects_negative_token_count() -> None:
    import mlx.core as mx

    from rfsn_v10.cache.contracts import PackedBlock

    block = PackedBlock(
        packed_codes=mx.array([0], dtype=mx.uint32),
        scales=mx.array([1.0], dtype=mx.float32),
        token_count=-1,
        bits=8,
        group_size=64,
        n_values=64,
        format_version=3,
    )
    with pytest.raises(ValueError, match="token_count"):
        block.validate()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_v3_block_validate_rejects_mismatched_n_values() -> None:
    import mlx.core as mx

    from rfsn_v10.cache.contracts import PackedBlock

    block = PackedBlock(
        packed_codes=mx.array([0] * 32, dtype=mx.uint32),
        scales=mx.array([1.0], dtype=mx.float32),
        token_count=1,
        bits=8,
        group_size=64,
        n_values=100,  # Wrong: 65 elements padded to 128
        num_elements=65,
        format_version=3,
    )
    with pytest.raises(ValueError, match="n_values"):
        block.validate()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_group_size_must_be_vector_aligned() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    with pytest.raises(ValueError, match="group_size"):
        CartesianCodec(bits=8, group_size=32)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_default_codec_has_wht_enabled() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    codec = CartesianCodec()
    assert codec.use_wht is True
    assert codec.sign_seed == 42
