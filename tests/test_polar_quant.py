#!/usr/bin/env python3
"""RFSN v10 — Polar quantization tests.

Covers pack/unpack buffer helpers, iterative hierarchical polar
forward/inverse, and uniform/grouped quantizer functions.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.polar_quant import (
    PackedCodeBuffer,
    PolarQuantizer,
    _pack_code_buffer,
    _unpack_code_buffer,
    dequantize_group_unsigned,
    dequantize_uniform_fixed_range,
    iterative_hierarchical_polar_forward,
    iterative_hierarchical_polar_inverse,
    quantize_group_unsigned,
    quantize_uniform_fixed_range,
)


class TestPackUnpackCodeBuffer:
    def test_pack_unpack_roundtrip_8bit(self):
        codes = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint32)
        buf = _pack_code_buffer(codes, bits=8)
        assert isinstance(buf, PackedCodeBuffer)
        assert buf.bits == 8
        unpacked = _unpack_code_buffer(buf)
        mx.eval(unpacked)
        assert mx.array_equal(unpacked, codes).item()

    def test_pack_unpack_roundtrip_4bit(self):
        codes = mx.array([0, 1, 2, 15], dtype=mx.uint32)
        buf = _pack_code_buffer(codes, bits=4)
        unpacked = _unpack_code_buffer(buf)
        mx.eval(unpacked)
        assert mx.array_equal(unpacked, codes).item()

    def test_pack_fallback_for_9bit(self):
        codes = mx.array([0, 100, 200], dtype=mx.uint32)
        buf = _pack_code_buffer(codes, bits=9)
        assert buf.bits == 9
        unpacked = _unpack_code_buffer(buf)
        mx.eval(unpacked)
        assert mx.array_equal(unpacked, codes).item()


class TestUniformFixedRange:
    def test_quantize_dequantize_roundtrip(self):
        x = mx.array([0.0, 0.25, 0.5, 0.75, 1.0])
        codes, meta = quantize_uniform_fixed_range(x, bits=4, min_val=0.0, max_val=1.0)
        recovered = dequantize_uniform_fixed_range(codes, meta)
        mx.eval(recovered)
        assert recovered.shape == x.shape
        # Quantization introduces small error
        assert mx.max(mx.abs(recovered - x)).item() < 0.1

    def test_quantize_clips_out_of_range(self):
        x = mx.array([-1.0, 0.0, 1.0, 2.0])
        codes, meta = quantize_uniform_fixed_range(x, bits=4, min_val=0.0, max_val=1.0)
        assert mx.max(codes).item() <= (1 << 4) - 1


class TestGroupUnsigned:
    def test_quantize_dequantize_roundtrip(self):
        x = mx.random.normal((64,))
        codes, meta = quantize_group_unsigned(x, bits=4, group_size=16)
        recovered = dequantize_group_unsigned(codes, meta)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_scale_is_positive(self):
        x = mx.random.normal((64,))
        codes, meta = quantize_group_unsigned(x, bits=4, group_size=16)
        mx.eval(meta.scale)
        assert mx.all(meta.scale > 0).item()


class TestIterativeHierarchicalPolar:
    def test_forward_inverse_roundtrip(self):
        x = mx.random.normal((1, 4, 64, 64))
        levels = 2
        angles, radii = iterative_hierarchical_polar_forward(x, levels=levels)
        recovered = iterative_hierarchical_polar_inverse(angles, radii)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_forward_returns_correct_shapes(self):
        x = mx.random.normal((1, 4, 64, 64))
        levels = 2
        angles, radii = iterative_hierarchical_polar_forward(x, levels=levels)
        assert len(angles) == levels
        # Each level halves the last dimension
        assert angles[0].shape[-1] == 32
        assert angles[1].shape[-1] == 16
        assert radii.shape[-1] == 16

    def test_levels_larger_than_zero(self):
        with pytest.raises(ValueError):
            iterative_hierarchical_polar_forward(
                mx.random.normal((1, 4, 64, 64)), levels=0,
            )

    def test_divisible_by_levels_required(self):
        with pytest.raises(ValueError, match="divisible"):
            iterative_hierarchical_polar_forward(
                mx.random.normal((1, 4, 64, 63)), levels=2,
            )


class TestPolarQuantizer:
    def test_init_defaults(self):
        q = PolarQuantizer()
        assert q.levels == 4
        assert q.angle_bits == 5
        assert q.radius_bits == 8

    def test_quantize_dequantize_roundtrip(self):
        q = PolarQuantizer(levels=2, angle_bits=4, radius_bits=4)
        x = mx.random.normal((1, 4, 64, 64))
        packed = q.quantize(x)
        recovered = q.dequantize(packed)
        mx.eval(recovered)
        assert recovered.shape == x.shape

    def test_estimate_bytes(self):
        q = PolarQuantizer(levels=1, angle_bits=4, radius_bits=4)
        x = mx.random.normal((1, 4, 64, 64))
        packed = q.quantize(x)
        bytes_est = q.estimate_bytes(packed)
        assert isinstance(bytes_est, int)
        assert bytes_est > 0
