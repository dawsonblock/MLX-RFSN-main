"""GPU-native bit packing and unpacking for Polar indices.

No NumPy conversion.  All operations stay on the MLX device.
"""
from __future__ import annotations

from typing import Any

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


# Bits per word for each supported width
_VALUES_PER_WORD: dict[int, int] = {2: 16, 3: 10, 4: 8}
_PACK_MASK: dict[int, int] = {2: 0x3, 3: 0x7, 4: 0xF}


def pack_indices(indices: Any, bits: int) -> Any:
    """Pack uint8 indices into uint32 words.

    Parameters
    ----------
    indices
        Array of uint8 codebook indices.  Last dimension is the coordinate
        axis (head_dim).  Any leading dimensions are treated as batch.
    bits
        Bit width (2, 3, or 4).

    Returns
    -------
    packed
        uint32 array where the last dimension is
        ``ceil(head_dim / values_per_word)``.
    """
    if mx is None:
        raise RuntimeError("MLX is not installed")
    if bits not in _VALUES_PER_WORD:
        raise ValueError(f"bits must be 2, 3, or 4; got {bits}")

    values_per_word = _VALUES_PER_WORD[bits]
    mask = _PACK_MASK[bits]

    # Ensure uint8
    indices = indices.astype(mx.uint8)
    head_dim = indices.shape[-1]
    n_words = (head_dim + values_per_word - 1) // values_per_word

    # Pad last dimension to multiple of values_per_word
    pad = values_per_word * n_words - head_dim
    if pad:
        pad_shape = list(indices.shape[:-1]) + [pad]
        indices = mx.concatenate([indices, mx.zeros(pad_shape, mx.uint8)], axis=-1)

    # Reshape so last dim becomes (n_words, values_per_word)
    batch_shape = indices.shape[:-1]
    flat = indices.reshape(*batch_shape, n_words, values_per_word)

    # Build the packed word using left shifts and bitwise OR
    # We accumulate into uint32
    packed = mx.zeros(flat.shape[:-1], mx.uint32)
    for i in range(values_per_word):
        val = flat[..., i].astype(mx.uint32) & mask
        packed = mx.bitwise_or(packed, mx.left_shift(val, i * bits))

    return packed


def unpack_indices(packed: Any, bits: int, original_dim: int) -> Any:
    """Unpack uint32 words back to uint8 indices.

    Parameters
    ----------
    packed
        uint32 array from :func:`pack_indices`.
    bits
        Bit width (2, 3, or 4).
    original_dim
        The original head_dim before padding.

    Returns
    -------
    indices
        uint8 array with last dimension ``original_dim``.
    """
    if mx is None:
        raise RuntimeError("MLX is not installed")
    if bits not in _VALUES_PER_WORD:
        raise ValueError(f"bits must be 2, 3, or 4; got {bits}")

    values_per_word = _VALUES_PER_WORD[bits]
    mask = _PACK_MASK[bits]

    # Extract each slot using right shift and mask
    slots: list[Any] = []
    for i in range(values_per_word):
        slot = mx.bitwise_and(mx.right_shift(packed, i * bits), mask).astype(mx.uint8)
        slots.append(slot)

    # Concatenate along last dimension
    unpacked = mx.stack(slots, axis=-1)

    # Reshape to flatten the word/slot hierarchy
    batch_shape = packed.shape[:-1]
    unpacked = unpacked.reshape(*batch_shape, -1)

    # Trim padding
    if unpacked.shape[-1] > original_dim:
        unpacked = unpacked[..., :original_dim]

    return unpacked
