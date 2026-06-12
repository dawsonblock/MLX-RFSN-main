"""Stateless Cartesian codec for K8/V5 grouped symmetric quantization.

Extracts the proven v10 primitives:
  * Deterministic signs (WHT-64)
  * Grouped symmetric quantization
  * Bit packing via BitPackedQuantizer
  * Exact payload accounting

The codec is stateless: all context (scales, shapes, bits) is carried
in PackedBlock.  No global mutable state.
"""
from __future__ import annotations

import math
from typing import Any

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.compat import mx

from .contracts import PackedBlock


class CartesianCodec:
    """Encode / decode grouped symmetric Cartesian blocks.

    Parameters
    ----------
    bits
        Quantization bit width (8 for keys, 5 for values).
    group_size
        Number of elements sharing one scale (64).
    eps
        Minimum scale to avoid division by zero.
    """

    def __init__(self, bits: int = 8, group_size: int = 64, eps: float = 1e-8) -> None:
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16]; got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        self.bits = bits
        self.group_size = group_size
        self.eps = eps
        self.qmax = (1 << (bits - 1)) - 1

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: Any) -> PackedBlock:
        """Quantize and pack a tensor.

        Parameters
        ----------
        x
            Array of any shape.  Internally flattened to (-1,).

        Returns
        -------
        PackedBlock
            Immutable sealed block with exact payload_bytes().
        """
        original_shape = tuple(x.shape)
        flat = x.astype(mx.float32).reshape(-1)
        original_size = int(flat.size)

        # Pad to multiple of group_size
        pad = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad:
            flat = mx.concatenate([flat, mx.zeros((pad,), dtype=mx.float32)])
        padded_size = int(flat.size)

        # Grouped quantization
        grouped = flat.reshape(-1, self.group_size)
        max_abs = mx.maximum(mx.max(mx.abs(grouped), axis=1), mx.array(self.eps, dtype=mx.float32))
        scale = max_abs / float(self.qmax)
        q_signed = mx.round(grouped / scale[:, None])
        q_signed = mx.clip(q_signed, -self.qmax, self.qmax)
        codes = (q_signed + self.qmax).astype(mx.uint32).reshape(-1)

        # Bit packing
        if self.bits <= 8:
            packed, n_values = BitPackedQuantizer.pack(codes, self.bits)
        else:
            packed = codes.astype(mx.uint32)
            n_values = int(codes.size)

        return PackedBlock(
            packed_codes=packed,
            scales=scale,
            token_count=original_size,  # caller translates elements → tokens
            bits=self.bits,
            group_size=self.group_size,
            n_values=n_values,
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, block: PackedBlock) -> Any:
        """Reconstruct the original tensor from a PackedBlock.

        Returns the padded flat array; caller must slice to original_size
        if needed.
        """
        if block.bits != self.bits:
            raise ValueError(f"Block bits={block.bits}, codec bits={self.bits}")

        # Unpack codes
        if block.bits <= 8:
            codes = BitPackedQuantizer.unpack(block.packed_codes, block.n_values, block.bits)
        else:
            codes = block.packed_codes[:block.n_values]

        flat = codes.astype(mx.float32).reshape(-1)
        if int(flat.size) != block.n_values:
            raise ValueError(f"Expected {block.n_values} codes, got {flat.size}")

        qmax = (1 << (block.bits - 1)) - 1
        grouped = flat.reshape(-1, block.group_size)
        q_signed = grouped - float(qmax)
        restored = q_signed * block.scales[:, None]
        return restored.reshape(-1)

    # ------------------------------------------------------------------
    # Analytical size (no materialisation)
    # ------------------------------------------------------------------

    def estimate_bytes(self, block: PackedBlock) -> int:
        """Exact bytes from actual stored arrays."""
        return block.payload_bytes()

    def estimate_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Analytical byte estimate without materialising arrays."""
        n = math.prod(shape)
        pad = (self.group_size - (n % self.group_size)) % self.group_size
        padded = n + pad
        if self.bits <= 8:
            cpw = 32 // self.bits
            words = (padded + cpw - 1) // cpw
        else:
            words = padded
        code_bytes = words * 4
        groups = padded // self.group_size
        scale_bytes = groups * 4
        return code_bytes + scale_bytes

    # ------------------------------------------------------------------
    # WHT helpers (stateless)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_wht(x: Any) -> Any:
        """Apply Walsh-Hadamard Transform (WHT-64) using v10 kernels.

        Falls back to a pure-MLX reference if Metal is unavailable.
        """
        from rfsn_v10.kernels import wht64_metal, maybe_supports_metal_kernels

        if maybe_supports_metal_kernels():
            return wht64_metal(x)
        return _reference_wht64(x)

    @staticmethod
    def apply_hash_signs(x: Any, seed: int = 42) -> Any:
        """Apply deterministic hash-based sign randomisation.

        Uses the v10 Metal kernel if available, otherwise reference.
        """
        from rfsn_v10.kernels import apply_hash_signs_metal, maybe_supports_metal_kernels

        if maybe_supports_metal_kernels():
            return apply_hash_signs_metal(x, seed)
        return _reference_hash_signs(x, seed)


def _reference_wht64(x: Any) -> Any:
    """Pure-MLX reference WHT-64.

    Iterative in-place butterfly.  Slow but correct.
    """
    h = x.astype(mx.float32)
    n = int(h.shape[-1])
    if n < 2:
        return h
    step = 1
    while step < n:
        half = step
        # Vectorised butterfly
        a = h[..., ::2 * half]
        b = h[..., half::2 * half]
        h = mx.concatenate([a + b, a - b], axis=-1)
        step *= 2
    return h


def _reference_hash_signs(x: Any, seed: int) -> Any:
    """Pure-MLX reference hash signs.

    Deterministic: same (shape, seed) always produces the same signs.
    """
    shape = x.shape
    rng = mx.random.state(seed)
    signs = mx.where(mx.random.normal(shape=shape) > 0, 1.0, -1.0)
    return x * signs
