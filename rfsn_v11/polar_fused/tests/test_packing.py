"""Tests for GPU-native bit packing."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.packing import pack_indices, unpack_indices


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_exact_round_trip(bits: int) -> None:
    """unpack(pack(x)) == x for all supported bit widths."""
    n_centroids = 2 ** bits
    shape = (4, 8, 64)
    indices = mx.random.randint(0, n_centroids, shape).astype(mx.uint8)
    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, original_dim=64)
    assert mx.array_equal(indices, unpacked)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_every_code_value(bits: int) -> None:
    """All possible code values survive round trip."""
    n_centroids = 2 ** bits
    shape = (1, 1, n_centroids)
    indices = mx.arange(n_centroids, dtype=mx.uint8).reshape(shape)
    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, original_dim=n_centroids)
    assert mx.array_equal(indices, unpacked)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dimension_not_divisible(bits: int) -> None:
    """Dimensions that need padding survive round trip."""
    n_centroids = 2 ** bits
    # Pick a dimension that is NOT a multiple of values_per_word
    values_per_word = {2: 16, 3: 10, 4: 8}[bits]
    original_dim = values_per_word - 1
    indices = mx.random.randint(0, n_centroids, (2, 3, original_dim)).astype(mx.uint8)
    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, original_dim=original_dim)
    assert mx.array_equal(indices, unpacked)
    assert unpacked.shape[-1] == original_dim


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_random_shapes(bits: int) -> None:
    """Various batch shapes survive round trip."""
    n_centroids = 2 ** bits
    for shape in [(1, 1, 64), (2, 4, 128), (3, 1, 256), (1, 8, 32)]:
        indices = mx.random.randint(0, n_centroids, shape).astype(mx.uint8)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, original_dim=shape[-1])
        assert mx.array_equal(indices, unpacked)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_all_zeros(bits: int) -> None:
    """All-zero indices pack and unpack correctly."""
    indices = mx.zeros((2, 2, 64), dtype=mx.uint8)
    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, original_dim=64)
    assert mx.array_equal(indices, unpacked)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_all_maximum(bits: int) -> None:
    """All-maximum indices pack and unpack correctly."""
    max_val = 2 ** bits - 1
    indices = mx.full((2, 2, 64), max_val, dtype=mx.uint8)
    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, original_dim=64)
    assert mx.array_equal(indices, unpacked)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_invalid_bits() -> None:
    indices = mx.zeros((1, 1, 64), dtype=mx.uint8)
    with pytest.raises(ValueError, match="bits must be"):
        pack_indices(indices, 5)
    with pytest.raises(ValueError, match="bits must be"):
        unpack_indices(indices, 5, 64)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_packed_dtype_is_uint32() -> None:
    indices = mx.zeros((1, 1, 64), dtype=mx.uint8)
    packed = pack_indices(indices, 4)
    assert packed.dtype == mx.uint32
