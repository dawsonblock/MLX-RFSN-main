#!/usr/bin/env python3
"""RFSN v10 — Turbo Polar quantization tests.

Covers TurboPolarQuantizer initialization, K/V quantization roundtrip,
packed format, and compression ratio.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.turbo_polar_quant import (
    TurboPolarPacked,
    TurboPolarQuantizer,
)


class TestTurboPolarQuantizerInit:
    def test_requires_power_of_two_feature_dim(self):
        with pytest.raises(ValueError, match="power of 2"):
            TurboPolarQuantizer(feature_dim=63)

    def test_accepts_valid_feature_dim(self):
        q = TurboPolarQuantizer(feature_dim=64)
        assert q.feature_dim == 64

    def test_custom_bits(self):
        q = TurboPolarQuantizer(feature_dim=64, k_angle_bits=4, k_radius_bits=6, v_bits=5)
        assert q.k_polar.angle_bits == 4
        assert q.k_polar.radius_bits == 6
        assert q.v_cart.bits == 5

    def test_adaptive_angle_range(self):
        q = TurboPolarQuantizer(feature_dim=64, adaptive_angle_range=True)
        assert q.k_polar.adaptive_angle_range is True


class TestTurboPolarQuantizeDequantize:
    def test_roundtrip_basic(self):
        q = TurboPolarQuantizer(feature_dim=64)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        assert isinstance(packed, TurboPolarPacked)
        k_rec, v_rec = q.dequantize(packed)
        mx.eval(k_rec, v_rec)
        assert k_rec.shape == k.shape
        assert v_rec.shape == v.shape

    def test_roundtrip_4bit(self):
        q = TurboPolarQuantizer(feature_dim=64, k_angle_bits=4, k_radius_bits=4, v_bits=4)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        k_rec, v_rec = q.dequantize(packed)
        mx.eval(k_rec, v_rec)
        assert k_rec.shape == k.shape
        assert v_rec.shape == v.shape

    def test_quantization_error_bounded(self):
        q = TurboPolarQuantizer(feature_dim=64, k_angle_bits=6, k_radius_bits=8, v_bits=7)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        k_rec, v_rec = q.dequantize(packed)
        mx.eval(k_rec, v_rec)
        k_rel = mx.mean(mx.abs(k_rec - k)) / mx.maximum(mx.mean(mx.abs(k)), 1e-8)
        v_rel = mx.mean(mx.abs(v_rec - v)) / mx.maximum(mx.mean(mx.abs(v)), 1e-8)
        assert k_rel.item() < 0.2
        assert v_rel.item() < 0.2


class TestTurboPolarPacked:
    def test_attributes_present(self):
        q = TurboPolarQuantizer(feature_dim=64)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        assert hasattr(packed, "k_polar")
        assert hasattr(packed, "v_cartesian")
        assert hasattr(packed, "original_shape_k")
        assert hasattr(packed, "original_shape_v")
        assert packed.original_shape_k == k.shape
        assert packed.original_shape_v == v.shape


class TestEstimateBytes:
    def test_estimate_positive(self):
        q = TurboPolarQuantizer(feature_dim=64)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        est = q.estimate_bytes(packed)
        assert isinstance(est, int)
        assert est > 0

    def test_compression_ratio(self):
        q = TurboPolarQuantizer(feature_dim=64)
        k = mx.random.normal((1, 4, 128, 64))
        v = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(k, v)
        ratio = q.compression_ratio(packed, k, v)
        assert ratio > 1.0
