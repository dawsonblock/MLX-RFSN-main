#!/usr/bin/env python3
"""
Experimental quantized KV manager.
Stable runtime should still default to the already validated Cartesian
k8_v5 path.
This manager is for experimental IsoQuant + Polar/Hybrid validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from rfsn_v10.compat import mx

from .hybrid_polar_cartesian import HybridPolarCartesianQuantizer
from .qjl_score_correction import QJLScoreCorrector, QJLSketch

QuantMode = Literal[
    "none",
    "hybrid_polar_cartesian",
]


@dataclass
class QuantizedKVPacket:
    mode: str
    k: Any
    v: Any
    original_k_shape: tuple[int, ...]
    original_v_shape: tuple[int, ...]
    k_bytes: int
    v_bytes: int
    fp16_bytes: int
    uses_qjl: bool = False
    k_qjl: QJLSketch | None = None
    v_qjl: QJLSketch | None = None


class QuantizedKVManager:
    def __init__(
        self,
        mode: QuantMode = "none",
        feature_dim: int = 64,
        polar_ratio: float = 0.65,
        polar_levels: int = 4,
        k_angle_bits: int = 5,
        k_radius_bits: int = 8,
        v_angle_bits: int = 4,
        v_radius_bits: int = 6,
        cartesian_bits: int = 6,
        group_size: int = 64,
        k_polar_enabled: bool = True,
        v_polar_enabled: bool = True,
        adaptive_angle_range: bool = False,
        use_qjl_score_correction: bool = False,
        qjl_proj_dim: int = 64,
    ):
        self.mode = mode
        self.feature_dim = feature_dim
        self.use_qjl_score_correction = use_qjl_score_correction
        self.k_quant = HybridPolarCartesianQuantizer(
            feature_dim=feature_dim,
            polar_ratio=polar_ratio,
            polar_levels=polar_levels,
            polar_angle_bits=k_angle_bits,
            polar_radius_bits=k_radius_bits,
            cartesian_bits=cartesian_bits,
            group_size=group_size,
            polar_enabled=k_polar_enabled,
            adaptive_angle_range=adaptive_angle_range,
        )
        self.v_quant = HybridPolarCartesianQuantizer(
            feature_dim=feature_dim,
            polar_ratio=polar_ratio,
            polar_levels=polar_levels,
            polar_angle_bits=v_angle_bits,
            polar_radius_bits=v_radius_bits,
            cartesian_bits=cartesian_bits,
            group_size=group_size,
            polar_enabled=v_polar_enabled,
            adaptive_angle_range=adaptive_angle_range,
        )
        self.qjl = QJLScoreCorrector(
            feature_dim=feature_dim,
            proj_dim=qjl_proj_dim,
        )

    @staticmethod
    def _fp16_bytes(k: mx.array, v: mx.array) -> int:
        return int(k.size + v.size) * 2

    def quantize(self, k: mx.array, v: mx.array) -> QuantizedKVPacket:
        if k.shape != v.shape:
            raise ValueError(
                f"K/V shape mismatch: {k.shape} vs {v.shape}"
            )
        if len(k.shape) != 4:
            raise ValueError(
                f"Expected K/V shape [B,H,T,D], got {k.shape}"
            )
        if k.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected D={self.feature_dim}, got {k.shape[-1]}"
            )
        fp16_bytes = self._fp16_bytes(k, v)
        if self.mode == "none":
            return QuantizedKVPacket(
                mode="none",
                k=k,
                v=v,
                original_k_shape=tuple(k.shape),
                original_v_shape=tuple(v.shape),
                k_bytes=int(k.size) * 2,
                v_bytes=int(v.size) * 2,
                fp16_bytes=fp16_bytes,
            )
        if self.mode != "hybrid_polar_cartesian":
            raise ValueError(f"Unsupported quant mode: {self.mode}")
        k_packed = self.k_quant.quantize(k)
        v_packed = self.v_quant.quantize(v)
        k_rec_for_qjl = None
        v_rec_for_qjl = None
        k_qjl = None
        v_qjl = None
        if self.use_qjl_score_correction:
            # QJL sketches the residual between original and base
            # reconstruction. It is used for score correction, not
            # dequant reconstruction.
            k_rec_for_qjl = self.k_quant.dequantize(k_packed)
            v_rec_for_qjl = self.v_quant.dequantize(v_packed)
            k_qjl = self.qjl.sketch_residual(
                k.astype(mx.float32)
                - k_rec_for_qjl.astype(mx.float32)
            )
            v_qjl = self.qjl.sketch_residual(
                v.astype(mx.float32)
                - v_rec_for_qjl.astype(mx.float32)
            )
        k_bytes = self.k_quant.estimate_bytes(k_packed)
        v_bytes = self.v_quant.estimate_bytes(v_packed)
        # QJL signs are 1 bit each plus residual norms as fp32.
        if k_qjl is not None:
            k_bytes += (int(k_qjl.signs.size) + 7) // 8
            k_bytes += int(k_qjl.residual_norm.size) * 4
        if v_qjl is not None:
            v_bytes += (int(v_qjl.signs.size) + 7) // 8
            v_bytes += int(v_qjl.residual_norm.size) * 4
        return QuantizedKVPacket(
            mode=self.mode,
            k=k_packed,
            v=v_packed,
            original_k_shape=tuple(k.shape),
            original_v_shape=tuple(v.shape),
            k_bytes=k_bytes,
            v_bytes=v_bytes,
            fp16_bytes=fp16_bytes,
            uses_qjl=self.use_qjl_score_correction,
            k_qjl=k_qjl,
            v_qjl=v_qjl,
        )

    def dequantize(
        self, packet: QuantizedKVPacket
    ) -> tuple[mx.array, mx.array]:
        if packet.mode == "none":
            return packet.k, packet.v
        if packet.mode != "hybrid_polar_cartesian":
            raise ValueError(
                f"Unsupported packet mode: {packet.mode}"
            )
        k_rec = self.k_quant.dequantize(packet.k)
        v_rec = self.v_quant.dequantize(packet.v)
        # Do NOT apply QJL here. QJL is score correction only.
        return k_rec, v_rec

    def estimate_bytes(self, packet: QuantizedKVPacket) -> int:
        """Return packed-buffer byte estimate for the K/V packet."""
        return int(packet.k_bytes + packet.v_bytes)

    def memory_report(self, packet: QuantizedKVPacket) -> dict[str, Any]:
        compressed = self.estimate_bytes(packet)
        ratio = packet.fp16_bytes / compressed if compressed > 0 else 1.0
        return {
            "fp16_kv_bytes": int(packet.fp16_bytes),
            "k_compressed_bytes": int(packet.k_bytes),
            "v_compressed_bytes": int(packet.v_bytes),
            "total_compressed_bytes": int(compressed),
            "actual_compression_ratio": float(ratio),
            "uses_qjl": bool(packet.uses_qjl),
        }

    def compression_ratio(self, packet: QuantizedKVPacket) -> float:
        compressed = packet.k_bytes + packet.v_bytes
        if compressed <= 0:
            return 1.0
        return packet.fp16_bytes / compressed

    def corrected_key_attention_scores(
        self, queries: mx.array, packet: QuantizedKVPacket
    ) -> mx.array:
        """
        Experimental attention-score correction for keys only.
        Returns:
            [B,H,Tq,Tk] scaled scores
        If QJL is not enabled, returns normal dequantized-key scores.
        """
        k_rec, _ = self.dequantize(packet)
        if not packet.uses_qjl or packet.k_qjl is None:
            return (
                queries.astype(mx.float32)
                @ k_rec.astype(mx.float32).transpose(0, 1, 3, 2)
            ) / math.sqrt(float(self.feature_dim))
        return self.qjl.corrected_attention_scores(
            queries, k_rec, packet.k_qjl
        )

