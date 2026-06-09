#!/usr/bin/env python3
"""RFSN v10 — Grouped Cartesian quantization tests.

Covers quantize/dequantize roundtrip, packed format, bit-width edge cases,
and scale consistency.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.grouped_cartesian import (
    GroupedCartesianQuantizer,
    PackedCartesianCodes,
)


class TestGroupedCartesianQuantizerInit:
    def test_default_init(self):
        q = GroupedCartesianQuantizer()
        assert q.bits == 6
        assert q.group_size == 64

    def test_custom_init(self):
        q = GroupedCartesianQuantizer(bits=4, group_size=32)
        assert q.bits == 4
        assert q.group_size == 32

    def test_rejects_bits_too_low(self):
        with pytest.raises(ValueError, match="bits"):
            GroupedCartesianQuantizer(bits=1)

    def test_rejects_bits_too_high(self):
        with pytest.raises(ValueError, match="bits"):
            GroupedCartesianQuantizer(bits=17)


class TestQuantizeDequantize:
    def test_roundtrip_8bit(self):
        q = GroupedCartesianQuantizer(bits=8, group_size=64)
        x = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(x)
        assert isinstance(packed, PackedCartesianCodes)
        recovered = q.dequantize(packed)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_roundtrip_4bit(self):
        q = GroupedCartesianQuantizer(bits=4, group_size=32)
        x = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(x)
        recovered = q.dequantize(packed)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_roundtrip_6bit(self):
        q = GroupedCartesianQuantizer(bits=6, group_size=64)
        x = mx.random.normal((1, 4, 256, 64))
        packed = q.quantize(x)
        recovered = q.dequantize(packed)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_scale_positive(self):
        q = GroupedCartesianQuantizer(bits=8, group_size=64)
        x = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(x)
        mx.eval(packed.scale)
        assert mx.all(packed.scale > 0).item()

    def test_quantization_error_bounded(self):
        q = GroupedCartesianQuantizer(bits=8, group_size=64)
        x = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(x)
        recovered = q.dequantize(packed)
        mx.eval(recovered)
        rel_error = mx.mean(mx.abs(recovered - x)) / mx.maximum(mx.mean(mx.abs(x)), 1e-8)
        assert rel_error.item() < 0.05  # 5% relative error


class TestPackedFormat:
    def test_packed_codes_is_packed_code_buffer(self):
        from rfsn_v10.quantization.polar_quant import PackedCodeBuffer
        q = GroupedCartesianQuantizer(bits=6, group_size=64)
        x = mx.random.normal((1, 4, 128, 64))
        packed = q.quantize(x)
        assert isinstance(packed.packed_codes, PackedCodeBuffer)

    def test_metadata_preserved(self):
        q = GroupedCartesianQuantizer(bits=5, group_size=32)
        x = mx.random.normal((2, 8, 256, 64))
        packed = q.quantize(x)
        assert packed.bits == 5
        assert packed.group_size == 32
        assert packed.original_shape == x.shape


class TestCartesianPacked:
    def test_direct_quantize_unpack(self):
        q = GroupedCartesianQuantizer(bits=6, group_size=64)
        x = mx.random.normal((1, 4, 128, 64))
        cart = q.quantize(x)
        # CartesianPacked is the raw-code variant
        recovered = q.dequantize(cart)
        mx.eval(recovered)
        assert recovered.shape == x.shape
