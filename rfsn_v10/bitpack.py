#!/usr/bin/env python3
"""
RFSN v10 - Bit-Packed Quantization.

Pack and unpack integer codes into compact bit representations using MLX.
Supports bit widths 2-8 with exact roundtrip guarantees.
"""
from __future__ import annotations

from .compat import mx


def _float_dtypes():
    """Return the set of MLX float dtypes (lazy, avoids import-time MLX access)."""
    return {
        dt for dt in [
            getattr(mx, "float16", None),
            getattr(mx, "float32", None),
            getattr(mx, "bfloat16", None),
        ]
        if dt is not None
}


class BitPackedQuantizer:
    """Pack and unpack integer codes into compact bit representations."""

    @staticmethod
    def pack(x: mx.array, bits: int) -> tuple[mx.array, int]:
        """Pack integer codes into a compact uint32 array.

        Args:
            x: Integer codes (any integer dtype or integer-valued float).
            bits: Bit width per code (2-8).

        Returns:
            (packed_uint32_array, n_values)

        Raises:
            ValueError: If bits not in [2,8], empty input, fractional floats,
                negative values, or out-of-range codes.
        """
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise ValueError(f"bits must be an integer in [2, 8], got {bits}")

        if x.size == 0:
            raise ValueError("Cannot pack empty array")

        # Handle float inputs: check integer-valued
        if x.dtype in _float_dtypes():
            x_rounded = mx.round(x)
            if not mx.all(x == x_rounded).item():
                raise ValueError(
                    "Float codes must be integer-valued, got non-integer values"
                )
            x = x_rounded.astype(mx.int32)

        # Check for negative values (for signed integer inputs)
        if x.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
            if mx.any(x < 0).item():
                raise ValueError("Codes cannot be negative")

        x = x.astype(mx.uint32)

        # Check range
        max_val = (1 << bits) - 1
        if mx.any(x > max_val).item():
            raise ValueError(
                f"Codes exceed maximum value {max_val} for {bits} bits"
            )

        n_values = int(x.size)
        codes_per_word = 32 // bits
        n_words = (n_values + codes_per_word - 1) // codes_per_word

        # Pad to multiple of codes_per_word
        pad_len = (codes_per_word - (n_values % codes_per_word)) % codes_per_word
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((pad_len,), dtype=mx.uint32)])

        # Reshape into groups and pack each group into one uint32 word
        x = x.reshape(-1, codes_per_word)
        packed = mx.zeros((n_words,), dtype=mx.uint32)
        for i in range(codes_per_word):
            packed = packed | (x[:, i] << (i * bits))

        return packed, n_values

    @staticmethod
    def unpack(packed: mx.array, n_values: int, bits: int) -> mx.array:
        """Unpack codes from a compact uint32 array.

        Args:
            packed: Packed uint32 array.
            n_values: Number of codes to extract.
            bits: Bit width per code (2-8).

        Returns:
            Unpacked uint32 array of length n_values.

        Raises:
            ValueError: If bits not in [2,8], n_values <= 0, empty buffer,
                or packed buffer too small.
        """
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise ValueError(f"bits must be an integer in [2, 8], got {bits}")

        if n_values <= 0:
            raise ValueError(f"n_values must be positive, got {n_values}")

        if packed.size == 0:
            raise ValueError("Cannot unpack from empty buffer")

        codes_per_word = 32 // bits
        required_words = (n_values + codes_per_word - 1) // codes_per_word

        if packed.size < required_words:
            raise ValueError(
                f"Packed buffer too small: need {required_words} words "
                f"for {n_values} values at {bits} bits, got {packed.size}"
            )

        # Vectorized extraction of all code slots in one broadcasted op.
        packed_view = packed[:required_words].reshape(-1, 1).astype(mx.uint32)
        shifts = (mx.arange(codes_per_word, dtype=mx.uint32) * bits).reshape(1, -1)
        mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
        codes = (packed_view >> shifts) & mask

        return codes.reshape(-1)[:n_values]
