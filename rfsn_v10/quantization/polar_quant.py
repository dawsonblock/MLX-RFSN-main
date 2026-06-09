#!/usr/bin/env python3
"""
Iterative hierarchical PolarQuant reference implementation.
Correct math:
    even = r * cos(theta)
    odd  = r * sin(theta)
Important:
    Use atan2(odd, even), preserving full quadrant information.
    Do NOT force angles into [0, pi/2] unless the corresponding coordinates are
    guaranteed non-negative. Level 0 coordinates are not guaranteed
    non-negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rfsn_v10.compat import mx

from rfsn_v10.bitpack import BitPackedQuantizer


@dataclass
class PackedCodeBuffer:
    packed: mx.array
    n_values: int
    bits: int
    original_shape: tuple[int, ...]
    dtype: str = "uint32"


@dataclass
class UniformQuantMeta:
    bits: int
    min_val: float
    max_val: float
    original_shape: tuple[int, ...]


@dataclass
class GroupQuantMeta:
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int
    scale: mx.array


@dataclass
class PackedPolarCodes:
    packed_angle_codes: list[PackedCodeBuffer]
    packed_radius_codes: PackedCodeBuffer
    angle_metas: list[UniformQuantMeta]
    radius_meta: GroupQuantMeta
    original_shape: tuple[int, ...]
    levels: int


@dataclass
class PolarPacked:
    """Unpacked raw-code variant kept for backward compatibility."""
    angle_codes: list[mx.array]
    angle_metas: list[UniformQuantMeta]
    radius_codes: mx.array
    radius_meta: GroupQuantMeta
    original_shape: tuple[int, ...]
    levels: int


def _pack_code_buffer(codes: mx.array, bits: int) -> PackedCodeBuffer:
    """Pack codes; for bits > 8 store as uint32 without word packing."""
    import warnings

    flat = codes.reshape(-1)
    if bits <= 8:
        packed, n_values = BitPackedQuantizer.pack(flat, bits)
    else:
        warnings.warn(
            f"Bit width {bits} exceeds 8-bit pack limit; "
            f"falling back to raw uint32 storage. "
            f"True bit-packing only applies to 2–8 bit widths.",
            stacklevel=2,
        )
        packed = flat.astype(mx.uint32)
        n_values = int(flat.size)
    return PackedCodeBuffer(
        packed=packed,
        n_values=n_values,
        bits=bits,
        original_shape=tuple(codes.shape),
    )


def _unpack_code_buffer(buf: PackedCodeBuffer) -> mx.array:
    """Unpack codes; for bits > 8 return raw uint32 slice."""
    if buf.bits <= 8:
        return BitPackedQuantizer.unpack(buf.packed, buf.n_values, buf.bits)
    return buf.packed[:buf.n_values]


def _require_divisible_by_levels(d: int, levels: int) -> None:
    base = 2**levels
    if d % base != 0:
        raise ValueError(
            f"Feature dimension D={d} must be divisible by 2**levels={base}."
        )


def iterative_hierarchical_polar_forward(
    x: mx.array, levels: int = 4, eps: float = 1e-12
) -> tuple[list[mx.array], mx.array]:
    """
    Non-recursive hierarchical polar transform.
    Input:
        x: [..., D]
    Output:
        angles_per_level:
            level 0: [..., D/2]
            level 1: [..., D/4]
            ...
        final_radii:
            [..., D / 2**levels]
    """
    if levels <= 0:
        raise ValueError(f"levels must be positive. Got {levels}")
    d = x.shape[-1]
    _require_divisible_by_levels(d, levels)
    current = x.astype(mx.float32)
    angles: list[mx.array] = []
    for _ in range(levels):
        even = current[..., ::2]
        odd = current[..., 1::2]
        r = mx.sqrt(even * even + odd * odd + eps)
        theta = mx.arctan2(odd, even)  # range [-pi, pi], preserves signs
        angles.append(theta)
        current = r
    final_radii = current
    return angles, final_radii


def iterative_hierarchical_polar_inverse(
    angles_per_level: list[mx.array], final_radii: mx.array
) -> mx.array:
    """
    Exact inverse of iterative_hierarchical_polar_forward when angles/radii
    are unquantized.
    """
    if not angles_per_level:
        return final_radii
    current = final_radii.astype(mx.float32)
    for theta in reversed(angles_per_level):
        if theta.shape[:-1] != current.shape[:-1]:
            raise ValueError(
                f"Batch shape mismatch. theta={theta.shape}, "
                f"current={current.shape}"
            )
        if theta.shape[-1] != current.shape[-1]:
            raise ValueError(
                f"Level shape mismatch. theta last dim={theta.shape[-1]}, "
                f"current last dim={current.shape[-1]}"
            )
        even = current * mx.cos(theta)
        odd = current * mx.sin(theta)
        # Interleave without item assignment.
        stacked = mx.stack([even, odd], axis=-1)
        current = stacked.reshape(
            current.shape[:-1] + (current.shape[-1] * 2,)
        )
    return current


def quantize_uniform_fixed_range(
    x: mx.array, bits: int, min_val: float, max_val: float
) -> tuple[mx.array, UniformQuantMeta]:
    """
    Uniform quantization into unsigned integer codes over a fixed range.
    Used for angles.
    codes in [0, 2**bits - 1]
    """
    if not (2 <= bits <= 16):
        raise ValueError(f"bits must be in [2,16]. Got {bits}")
    if not max_val > min_val:
        raise ValueError("max_val must be greater than min_val")
    qmax = (1 << bits) - 1
    x_clipped = mx.clip(x.astype(mx.float32), min_val, max_val)
    scaled = (x_clipped - min_val) * (qmax / (max_val - min_val))
    codes = mx.round(scaled).astype(mx.uint32)
    meta = UniformQuantMeta(
        bits=bits,
        min_val=float(min_val),
        max_val=float(max_val),
        original_shape=tuple(x.shape),
    )
    return codes, meta


def dequantize_uniform_fixed_range(
    codes: mx.array, meta: UniformQuantMeta
) -> mx.array:
    qmax = (1 << meta.bits) - 1
    x = (
        codes.astype(mx.float32)
        * ((meta.max_val - meta.min_val) / qmax)
        + meta.min_val
    )
    return x.reshape(meta.original_shape)


def quantize_group_unsigned(
    x: mx.array, bits: int, group_size: int = 64, eps: float = 1e-8
) -> tuple[mx.array, GroupQuantMeta]:
    """
    Groupwise unsigned quantization for non-negative radii.
    Uses per-group max scale:
        q = round(x / scale)
        scale = max(x_group) / qmax
    """
    if not (2 <= bits <= 16):
        raise ValueError(f"bits must be in [2,16]. Got {bits}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    original_shape = tuple(x.shape)
    flat = x.astype(mx.float32).reshape(-1)
    original_size = int(flat.size)
    pad = (group_size - (original_size % group_size)) % group_size
    if pad:
        flat = mx.concatenate(
            [flat, mx.zeros((pad,), dtype=flat.dtype)], axis=0
        )
    padded_size = int(flat.size)
    grouped = flat.reshape(-1, group_size)
    qmax = (1 << bits) - 1
    max_abs = mx.maximum(
        mx.max(grouped, axis=1),
        mx.array(eps, dtype=mx.float32),
    )
    scale = max_abs / float(qmax)
    codes = mx.round(grouped / scale[:, None])
    codes = mx.clip(codes, 0, qmax).astype(mx.uint32).reshape(-1)
    meta = GroupQuantMeta(
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        original_size=original_size,
        padded_size=padded_size,
        scale=scale,
    )
    return codes, meta


def dequantize_group_unsigned(
    codes: mx.array, meta: GroupQuantMeta
) -> mx.array:
    flat = codes.astype(mx.float32).reshape(-1)
    if int(flat.size) != meta.padded_size:
        raise ValueError(
            f"Expected {meta.padded_size} codes, got {flat.size}"
        )
    grouped = flat.reshape(-1, meta.group_size)
    restored = grouped * meta.scale[:, None]
    restored = restored.reshape(-1)[: meta.original_size]
    return restored.reshape(meta.original_shape)


class PolarQuantizer:
    """
    Hierarchical PolarQuant reference quantizer.
    Angles use fixed range [-pi, pi] or adaptive per-tensor range.
    Radii use unsigned groupwise quantization.
    """

    def __init__(
        self,
        levels: int = 4,
        angle_bits: int = 5,
        radius_bits: int = 8,
        radius_group_size: int = 64,
        adaptive_angle_range: bool = False,
    ):
        if not (1 <= angle_bits <= 16):
            raise ValueError(
                f"angle_bits must be in [1, 16], got {angle_bits}"
            )
        if not (1 <= radius_bits <= 16):
            raise ValueError(
                f"radius_bits must be in [1, 16], got {radius_bits}"
            )
        if radius_group_size <= 0:
            raise ValueError(
                f"radius_group_size must be positive, got {radius_group_size}"
            )
        self.levels = levels
        self.angle_bits = angle_bits
        self.radius_bits = radius_bits
        self.radius_group_size = radius_group_size
        self.adaptive_angle_range = adaptive_angle_range

    def quantize(self, x: mx.array) -> PackedPolarCodes:
        angles, final_radii = iterative_hierarchical_polar_forward(
            x, self.levels
        )
        packed_angle_codes: list[PackedCodeBuffer] = []
        angle_metas: list[UniformQuantMeta] = []
        for theta in angles:
            if self.adaptive_angle_range:
                min_val = float(mx.min(theta))
                max_val = float(mx.max(theta))
                if max_val <= min_val:
                    max_val = min_val + 1e-6
            else:
                min_val = -math.pi
                max_val = math.pi
            codes, meta = quantize_uniform_fixed_range(
                theta,
                bits=self.angle_bits,
                min_val=min_val,
                max_val=max_val,
            )
            packed_angle_codes.append(
                _pack_code_buffer(codes, self.angle_bits)
            )
            angle_metas.append(meta)
        radius_codes, radius_meta = quantize_group_unsigned(
            final_radii,
            bits=self.radius_bits,
            group_size=self.radius_group_size,
        )
        packed_radius_codes = _pack_code_buffer(
            radius_codes, self.radius_bits
        )
        return PackedPolarCodes(
            packed_angle_codes=packed_angle_codes,
            packed_radius_codes=packed_radius_codes,
            angle_metas=angle_metas,
            radius_meta=radius_meta,
            original_shape=tuple(x.shape),
            levels=self.levels,
        )

    def dequantize(self, packed: PackedPolarCodes) -> mx.array:
        if packed.levels != self.levels:
            raise ValueError(
                f"Packed levels={packed.levels}, "
                f"quantizer levels={self.levels}"
            )
        angles = [
            dequantize_uniform_fixed_range(
                _unpack_code_buffer(buf), meta
            )
            for buf, meta in zip(
                packed.packed_angle_codes, packed.angle_metas
            )
        ]
        final_radii = dequantize_group_unsigned(
            _unpack_code_buffer(packed.packed_radius_codes),
            packed.radius_meta,
        )
        restored = iterative_hierarchical_polar_inverse(angles, final_radii)
        return restored.reshape(packed.original_shape)

    def estimate_bytes(self, packed: PackedPolarCodes) -> int:
        """
        Actual byte accounting of packed buffers plus float32 scales.
        """
        angle_bytes = sum(
            int(buf.packed.size) * 4 for buf in packed.packed_angle_codes
        )
        radius_bytes = int(packed.packed_radius_codes.packed.size) * 4
        scale_bytes = int(packed.radius_meta.scale.size) * 4
        # Angle metadata: 4 floats per meta
        meta_bytes = len(packed.angle_metas) * 4 * 4
        return angle_bytes + radius_bytes + scale_bytes + meta_bytes

    def estimate_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Analytical byte estimate without materialising arrays."""
        n = math.prod(shape)
        angle_bytes = 0
        cpw = 32 // self.angle_bits
        for _ in range(self.levels):
            if self.angle_bits <= 8:
                words = (n + cpw - 1) // cpw
            else:
                words = n
            angle_bytes += words * 4
        if self.radius_bits <= 8:
            cpw_r = 32 // self.radius_bits
            radius_words = (n + cpw_r - 1) // cpw_r
        else:
            radius_words = n
        radius_bytes = radius_words * 4
        groups = (n + self.radius_group_size - 1) // self.radius_group_size
        scale_bytes = groups * 4
        meta_bytes = self.levels * 4 * 4
        return angle_bytes + radius_bytes + scale_bytes + meta_bytes
