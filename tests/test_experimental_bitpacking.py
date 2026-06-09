#!/usr/bin/env python3
"""
RFSN v10 Experimental Quantization — Bit-Packing Acceptance Tests.

Verifies that experimental quantizers (Polar, Cartesian, Hybrid, TurboPolar)
produce real packed buffers, report honest compression, and roundtrip
with acceptable quality.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.grouped_cartesian import GroupedCartesianQuantizer
from rfsn_v10.quantization.hybrid_polar_cartesian import (
    HybridPolarCartesianQuantizer,
)
from rfsn_v10.quantization.polar_quant import PolarQuantizer
from rfsn_v10.quantization.turbo_polar_quant import TurboPolarQuantizer

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cosine(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_f * b_f).item()
    na = mx.sqrt(mx.sum(a_f * a_f)).item()
    nb = mx.sqrt(mx.sum(b_f * b_f)).item()
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _make_kv(shape: tuple[int, ...]) -> tuple[mx.array, mx.array]:
    mx.random.seed(42)
    k = mx.random.normal(shape).astype(mx.float16)
    v = mx.random.normal(shape).astype(mx.float16)
    return k, v


# ------------------------------------------------------------------
# PolarQuantizer
# ------------------------------------------------------------------

class TestPolarQuantizer:
    def test_packed_buffer_exists(self):
        q = PolarQuantizer(levels=4, angle_bits=5, radius_bits=8)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        assert len(packed.packed_angle_codes) == 4
        for buf in packed.packed_angle_codes:
            assert buf.packed is not None
            assert buf.packed.size > 0
        assert packed.packed_radius_codes.packed is not None
        assert packed.packed_radius_codes.packed.size > 0

    def test_packed_bytes_less_than_raw(self):
        q = PolarQuantizer(levels=4, angle_bits=5, radius_bits=8)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        # Total raw uint32 bytes = sum of all code sizes * 4
        raw_angle_bytes = sum(
            buf.n_values * 4 for buf in packed.packed_angle_codes
        )
        raw_radius_bytes = packed.packed_radius_codes.n_values * 4
        raw_total = raw_angle_bytes + raw_radius_bytes
        packed_total = q.estimate_bytes(packed)
        assert packed_total < raw_total, (
            f"Packed bytes {packed_total} should be < raw bytes {raw_total}"
        )

    def test_dequantize_shape_matches(self):
        q = PolarQuantizer(levels=4, angle_bits=5, radius_bits=8)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        assert rec.shape == x.shape

    def test_cosine_above_threshold(self):
        q = PolarQuantizer(levels=4, angle_bits=6, radius_bits=8)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        c = _cosine(x, rec)
        assert c >= 0.95, f"cosine={c}, expected >= 0.95"

    def test_compression_ratio_greater_than_one(self):
        q = PolarQuantizer(levels=4, angle_bits=5, radius_bits=8)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        fp16_bytes = int(x.size) * 2
        compressed = q.estimate_bytes(packed)
        ratio = fp16_bytes / max(compressed, 1)
        assert ratio > 1.0, f"compression_ratio={ratio}, expected > 1.0"


# ------------------------------------------------------------------
# GroupedCartesianQuantizer
# ------------------------------------------------------------------

class TestGroupedCartesianQuantizer:
    def test_packed_buffer_exists(self):
        q = GroupedCartesianQuantizer(bits=5, group_size=64)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        assert packed.packed_codes.packed is not None
        assert packed.packed_codes.packed.size > 0

    def test_packed_bytes_less_than_raw(self):
        q = GroupedCartesianQuantizer(bits=5, group_size=64)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        raw_bytes = packed.packed_codes.n_values * 4
        packed_bytes = q.estimate_bytes(packed)
        assert packed_bytes < raw_bytes, (
            f"Packed {packed_bytes} should be < raw {raw_bytes}"
        )

    def test_dequantize_shape_matches(self):
        q = GroupedCartesianQuantizer(bits=5, group_size=64)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        assert rec.shape == x.shape

    def test_cosine_above_threshold(self):
        q = GroupedCartesianQuantizer(bits=6, group_size=64)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        c = _cosine(x, rec)
        assert c >= 0.98, f"cosine={c}, expected >= 0.98"

    def test_compression_ratio_greater_than_one(self):
        q = GroupedCartesianQuantizer(bits=5, group_size=64)
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        fp16_bytes = int(x.size) * 2
        compressed = q.estimate_bytes(packed)
        ratio = fp16_bytes / max(compressed, 1)
        assert ratio > 1.0, f"compression_ratio={ratio}, expected > 1.0"


# ------------------------------------------------------------------
# HybridPolarCartesianQuantizer
# ------------------------------------------------------------------

class TestHybridPolarCartesianQuantizer:
    def test_packed_buffer_exists(self):
        q = HybridPolarCartesianQuantizer(
            feature_dim=64,
            polar_ratio=0.65,
            polar_levels=4,
            polar_angle_bits=5,
            polar_radius_bits=8,
            cartesian_bits=5,
            group_size=64,
        )
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        assert packed.polar is not None
        assert packed.cartesian is not None
        assert len(packed.polar.packed_angle_codes) == 4
        for buf in packed.polar.packed_angle_codes:
            assert buf.packed is not None
        assert packed.polar.packed_radius_codes.packed is not None
        assert packed.cartesian.packed_codes.packed is not None

    def test_dequantize_shape_matches(self):
        q = HybridPolarCartesianQuantizer(
            feature_dim=64,
            polar_ratio=0.65,
            polar_levels=4,
            polar_angle_bits=5,
            polar_radius_bits=8,
            cartesian_bits=5,
            group_size=64,
        )
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        assert rec.shape == x.shape

    def test_cosine_above_threshold(self):
        q = HybridPolarCartesianQuantizer(
            feature_dim=64,
            polar_ratio=0.65,
            polar_levels=4,
            polar_angle_bits=6,
            polar_radius_bits=8,
            cartesian_bits=6,
            group_size=64,
        )
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        rec = q.dequantize(packed)
        c = _cosine(x, rec)
        assert c >= 0.95, f"cosine={c}, expected >= 0.95"

    def test_compression_ratio_greater_than_one(self):
        q = HybridPolarCartesianQuantizer(
            feature_dim=64,
            polar_ratio=0.65,
            polar_levels=4,
            polar_angle_bits=5,
            polar_radius_bits=8,
            cartesian_bits=5,
            group_size=64,
        )
        x = mx.random.normal((2, 4, 16, 64)).astype(mx.float16)
        packed = q.quantize(x)
        fp16_bytes = int(x.size) * 2
        compressed = q.estimate_bytes(packed)
        ratio = fp16_bytes / max(compressed, 1)
        assert ratio > 1.0, f"compression_ratio={ratio}, expected > 1.0"


# ------------------------------------------------------------------
# TurboPolarQuantizer
# ------------------------------------------------------------------

class TestTurboPolarQuantizer:
    def test_packed_buffer_exists(self):
        q = TurboPolarQuantizer(
            feature_dim=64, k_angle_bits=8, k_radius_bits=8, v_bits=7
        )
        k, v = _make_kv((2, 4, 16, 64))
        packed = q.quantize(k, v)
        assert packed.k_polar is not None
        assert packed.v_cartesian is not None
        assert packed.k_polar.packed_radius_codes.packed is not None
        assert packed.v_cartesian.packed_codes.packed is not None

    def test_dequantize_shape_matches(self):
        q = TurboPolarQuantizer(
            feature_dim=64, k_angle_bits=8, k_radius_bits=8, v_bits=7
        )
        k, v = _make_kv((2, 4, 16, 64))
        packed = q.quantize(k, v)
        rk, rv = q.dequantize(packed)
        assert rk.shape == k.shape
        assert rv.shape == v.shape

    def test_cosine_above_threshold(self):
        q = TurboPolarQuantizer(
            feature_dim=64, k_angle_bits=8, k_radius_bits=8, v_bits=7
        )
        k, v = _make_kv((2, 4, 16, 64))
        packed = q.quantize(k, v)
        rk, rv = q.dequantize(packed)
        ck = _cosine(k, rk)
        cv = _cosine(v, rv)
        assert ck >= 0.95, f"key cosine={ck}, expected >= 0.95"
        assert cv >= 0.95, f"value cosine={cv}, expected >= 0.95"

    def test_compression_ratio_greater_than_one(self):
        q = TurboPolarQuantizer(
            feature_dim=64, k_angle_bits=8, k_radius_bits=8, v_bits=7
        )
        k, v = _make_kv((2, 4, 16, 64))
        packed = q.quantize(k, v)
        fp16_bytes = (int(k.size) + int(v.size)) * 2
        compressed = q.estimate_bytes(packed)
        ratio = fp16_bytes / max(compressed, 1)
        assert ratio > 1.0, (
            f"compression_ratio={ratio}, expected > 1.0"
        )

    def test_k_radius_bits_nine_fallback(self):
        """When radius_bits=9 (>8), codes should still store and roundtrip."""
        q = TurboPolarQuantizer(
            feature_dim=64, k_angle_bits=8, k_radius_bits=9, v_bits=7
        )
        k, v = _make_kv((2, 4, 16, 64))
        packed = q.quantize(k, v)
        rk, rv = q.dequantize(packed)
        assert rk.shape == k.shape
        assert rv.shape == v.shape
        # Verify estimate_bytes reflects raw uint32 storage (no packing gain)
        estimated = q.estimate_bytes(packed)
        radius_n_values = packed.k_polar.packed_radius_codes.n_values
        raw_radius_bytes = radius_n_values * 4
        assert estimated >= raw_radius_bytes, (
            f"estimated={estimated} should cover raw uint32 bytes {raw_radius_bytes}"
        )
