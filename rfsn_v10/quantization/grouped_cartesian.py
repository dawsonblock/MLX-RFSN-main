#!/usr/bin/env python3
"""
Grouped Cartesian quantization reference implementation.
Uses symmetric signed quantization:
    q_signed = round(x / scale)
    code = q_signed + qmax
For bits=5:
    qmax = 15
    signed range = [-15, 15]
    code range = [0, 30]
    one code value remains unused. This preserves exact zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rfsn_v10.compat import mx

from rfsn_v10.bitpack import BitPackedQuantizer

from .polar_quant import PackedCodeBuffer


@dataclass
class PackedCartesianCodes:
    packed_codes: PackedCodeBuffer
    scale: mx.array
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int


@dataclass
class CartesianPacked:
    codes: mx.array
    scale: mx.array
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int


class GroupedCartesianQuantizer:
    def __init__(
        self,
        bits: int = 6,
        group_size: int = 64,
        eps: float = 1e-8,
    ):
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16]. Got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.bits = bits
        self.group_size = group_size
        self.eps = eps

    def quantize(self, x: mx.array) -> PackedCartesianCodes:
        original_shape = tuple(x.shape)
        flat = x.astype(mx.float32).reshape(-1)
        original_size = int(flat.size)
        pad = (self.group_size - (original_size % self.group_size))
        pad %= self.group_size
        if pad:
            flat = mx.concatenate(
                [flat, mx.zeros((pad,), dtype=flat.dtype)], axis=0
            )
        padded_size = int(flat.size)
        grouped = flat.reshape(-1, self.group_size)
        qmax = (1 << (self.bits - 1)) - 1
        max_abs = mx.maximum(
            mx.max(mx.abs(grouped), axis=1),
            mx.array(self.eps, dtype=mx.float32),
        )
        scale = max_abs / float(qmax)
        q_signed = mx.round(grouped / scale[:, None])
        q_signed = mx.clip(q_signed, -qmax, qmax)
        codes = (q_signed + qmax).astype(mx.uint32).reshape(-1)
        if self.bits <= 8:
            packed, n_values = BitPackedQuantizer.pack(codes, self.bits)
        else:
            import warnings

            warnings.warn(
                f"GroupedCartesianQuantizer bits={self.bits} exceeds 8-bit "
                f"pack limit; falling back to raw uint32 storage. "
                f"True bit-packing only applies to 2–8 bit widths.",
                stacklevel=2,
            )
            packed = codes.astype(mx.uint32)
            n_values = int(codes.size)
        packed_buf = PackedCodeBuffer(
            packed=packed,
            n_values=n_values,
            bits=self.bits,
            original_shape=tuple(codes.shape),
        )
        return PackedCartesianCodes(
            packed_codes=packed_buf,
            scale=scale,
            bits=self.bits,
            group_size=self.group_size,
            original_shape=original_shape,
            original_size=original_size,
            padded_size=padded_size,
        )

    def dequantize(self, packed: PackedCartesianCodes) -> mx.array:
        if packed.bits != self.bits:
            raise ValueError(
                f"Packed bits={packed.bits}, quantizer bits={self.bits}"
            )
        buf = packed.packed_codes
        if buf.bits <= 8:
            codes = BitPackedQuantizer.unpack(
                buf.packed, buf.n_values, buf.bits
            )
        else:
            codes = buf.packed[:buf.n_values]
        flat = codes.astype(mx.float32).reshape(-1)
        if int(flat.size) != packed.padded_size:
            raise ValueError(
                f"Expected {packed.padded_size} codes, got {flat.size}"
            )
        qmax = (1 << (packed.bits - 1)) - 1
        grouped_codes = flat.reshape(-1, packed.group_size)
        q_signed = grouped_codes - float(qmax)
        restored = q_signed * packed.scale[:, None]
        restored = restored.reshape(-1)[: packed.original_size]
        return restored.reshape(packed.original_shape)

    def estimate_bytes(self, packed: PackedCartesianCodes) -> int:
        code_bytes = int(packed.packed_codes.packed.size) * 4
        scale_bytes = int(packed.scale.size) * 4
        return code_bytes + scale_bytes

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
