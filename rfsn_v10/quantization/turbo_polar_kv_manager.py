#!/usr/bin/env python3
"""KV manager that uses TurboPolarQuantizer."""
from __future__ import annotations

from rfsn_v10.compat import mx

from rfsn_v10.quantization.turbo_polar_quant import TurboPolarQuantizer


class TurboPolarKVManager:
    def __init__(
        self,
        feature_dim: int = 64,
        k_angle_bits: int = 8,
        k_radius_bits: int = 8,
        v_bits: int = 7,
        group_size: int = 64,
        adaptive_angle_range: bool = False,
    ):
        self.mode = "turbo_polar"
        self.feature_dim = feature_dim
        self.quantizer = TurboPolarQuantizer(
            feature_dim=feature_dim,
            k_angle_bits=k_angle_bits,
            k_radius_bits=k_radius_bits,
            v_bits=v_bits,
            group_size=group_size,
            adaptive_angle_range=adaptive_angle_range,
        )

    def quantize(self, k: mx.array, v: mx.array):
        return self.quantizer.quantize(k, v)

    def dequantize(self, packed):
        return self.quantizer.dequantize(packed)

    def estimate_bytes(self, packed) -> int:
        return self.quantizer.estimate_bytes(packed)

    def compression_ratio(self, packed, k: mx.array, v: mx.array) -> float:
        return self.quantizer.compression_ratio(packed, k, v)
