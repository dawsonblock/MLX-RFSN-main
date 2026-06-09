"""
Bit-packing utilities for TurboQuant compressed storage.

Fully vectorized using MLX operations — no Python loops.
"""

import mlx.core as mx


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack low-bit indices into uint32 arrays (vectorized).

    Args:
        indices: (..., dim) uint8 array with values in [0, 2^bits - 1]
        bits: Bits per index (1-4)

    Returns:
        packed: (..., n_packed) uint32 array
    """
    shape = indices.shape
    dim = shape[-1]
    vals_per_int = 32 // bits
    n_packed = (dim + vals_per_int - 1) // vals_per_int

    # Pad to multiple of vals_per_int
    pad_size = n_packed * vals_per_int - dim
    if pad_size > 0:
        pad_shape = (*shape[:-1], pad_size)
        indices = mx.concatenate([indices, mx.zeros(pad_shape, dtype=indices.dtype)], axis=-1)

    # Reshape: (..., n_packed, vals_per_int)
    reshaped = indices.reshape(*shape[:-1], n_packed, vals_per_int).astype(mx.uint32)

    # Shift each value by its position: val << (i * bits)
    shifts = mx.arange(vals_per_int).astype(mx.uint32) * bits
    shifted = reshaped << shifts

    # OR-reduce to pack
    packed = shifted[..., 0]
    for i in range(1, vals_per_int):
        packed = packed | shifted[..., i]

    return packed


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 array back to individual indices (vectorized).

    Args:
        packed: (..., n_packed) uint32 array
        bits: Bits per index (1-4)
        dim: Original dimension

    Returns:
        indices: (..., dim) uint8 array
    """
    shape = packed.shape
    vals_per_int = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)

    # Expand: (..., n_packed) -> (..., n_packed, vals_per_int)
    expanded = mx.expand_dims(packed, axis=-1)  # (..., n_packed, 1)
    shifts = mx.arange(vals_per_int).astype(mx.uint32) * bits
    extracted = (expanded >> shifts) & mask  # (..., n_packed, vals_per_int)

    # Flatten last two dims and trim to original dim
    full_dim = shape[-1] * vals_per_int
    flat = extracted.reshape(*shape[:-1], full_dim)
    return flat[..., :dim].astype(mx.uint8)


def pack_signs(signs: mx.array) -> mx.array:
    """Pack boolean sign bits into uint32 arrays (vectorized).

    Args:
        signs: (..., dim) bool array

    Returns:
        packed: (..., ceil(dim/32)) uint32 array
    """
    return pack_indices(signs.astype(mx.uint8), bits=1)


def unpack_signs(packed: mx.array, dim: int) -> mx.array:
    """Unpack uint32 array back to boolean sign bits.

    Args:
        packed: (..., n_packed) uint32 array
        dim: Original dimension

    Returns:
        signs: (..., dim) bool array
    """
    return unpack_indices(packed, bits=1, dim=dim).astype(mx.bool_)


def packed_nbytes(dim: int, bits: int, has_qjl: bool = False) -> int:
    """Calculate packed bytes per vector."""
    vals_per_int = 32 // bits
    n_packed_indices = (dim + vals_per_int - 1) // vals_per_int
    index_bytes = n_packed_indices * 4

    norm_bytes = 4

    sign_bytes = 0
    if has_qjl:
        n_packed_signs = (dim + 31) // 32
        sign_bytes = n_packed_signs * 4 + 4

    return index_bytes + norm_bytes + sign_bytes
