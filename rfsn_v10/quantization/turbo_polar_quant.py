#!/usr/bin/env python3
"""
TurboPolarQuant: WHT preconditioning + single-level polar for K,
grouped symmetric for V.  Attempts to beat the stable k8_v5_gs64
baseline by using polar decomposition on keys where it matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rfsn_v10.compat import mx

from rfsn_v10.quantization.grouped_cartesian import GroupedCartesianQuantizer
from rfsn_v10.quantization.polar_quant import PolarQuantizer

# PackedPolarCodes and PackedCartesianCodes are used implicitly
# through the quantizer return types.


def _wht(x: mx.array) -> mx.array:
    """Recursive normalized Walsh-Hadamard transform."""
    d = x.shape[-1]
    if d == 1:
        return x
    if d % 2 != 0:
        raise ValueError("WHT requires last dim to be power of 2")
    even = x[..., ::2]
    odd = x[..., 1::2]
    h1 = _wht(even + odd)
    h2 = _wht(even - odd)
    stacked = mx.stack([h1, h2], axis=-1)
    out = stacked.reshape(x.shape)
    return out / math.sqrt(2)


@dataclass
class TurboPolarPacked:
    k_polar: object
    v_cartesian: object
    original_shape_k: tuple[int, ...]
    original_shape_v: tuple[int, ...]


class TurboPolarQuantizer:
    """
    Quantizer that uses:
      - WHT preconditioning
      - Single-level polar quantization for K
      - Grouped Cartesian quantization for V
    """

    def __init__(
        self,
        feature_dim: int,
        k_angle_bits: int = 8,
        k_radius_bits: int = 9,
        v_bits: int = 7,
        group_size: int = 64,
        adaptive_angle_range: bool = False,
    ):
        if feature_dim <= 0 or (feature_dim & (feature_dim - 1)) != 0:
            raise ValueError("feature_dim must be a power of 2")
        self.feature_dim = feature_dim
        self.k_polar = PolarQuantizer(
            levels=1,
            angle_bits=k_angle_bits,
            radius_bits=k_radius_bits,
            radius_group_size=group_size,
            adaptive_angle_range=adaptive_angle_range,
        )
        self.v_cart = GroupedCartesianQuantizer(
            bits=v_bits,
            group_size=group_size,
        )

    def quantize(self, k: mx.array, v: mx.array) -> TurboPolarPacked:
        k_wht = _wht(k.astype(mx.float32))
        v_wht = _wht(v.astype(mx.float32))
        k_packed = self.k_polar.quantize(k_wht)
        v_packed = self.v_cart.quantize(v_wht)
        return TurboPolarPacked(
            k_polar=k_packed,
            v_cartesian=v_packed,
            original_shape_k=tuple(k.shape),
            original_shape_v=tuple(v.shape),
        )

    def dequantize(self, packed: TurboPolarPacked) -> tuple[mx.array, mx.array]:
        k_wht = self.k_polar.dequantize(packed.k_polar)
        v_wht = self.v_cart.dequantize(packed.v_cartesian)
        k = _wht(k_wht)
        v = _wht(v_wht)
        return k.reshape(packed.original_shape_k), v.reshape(
            packed.original_shape_v
        )

    def estimate_bytes(self, packed: TurboPolarPacked) -> int:
        return (
            self.k_polar.estimate_bytes(packed.k_polar)
            + self.v_cart.estimate_bytes(packed.v_cartesian)
        )

    def compression_ratio(
        self, packed: TurboPolarPacked, k: mx.array, v: mx.array
    ) -> float:
        fp16 = int(k.size + v.size) * 2
        compressed = self.estimate_bytes(packed)
        return fp16 / max(compressed, 1)
