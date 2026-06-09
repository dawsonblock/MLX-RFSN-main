"""Bit-pack fuzz tests — random shapes, dtypes, edge cases."""
from __future__ import annotations

import random

import pytest

from rfsn_v10.bitpack import BitPackedQuantizer

mx = pytest.importorskip("mlx.core")


class TestBitPackFuzz:
    def test_random_shapes_and_bits(self):
        for _ in range(50):
            bits = random.randint(2, 8)
            n_values = random.randint(1, 1000)
            max_val = (1 << bits) - 1
            codes = mx.random.randint(0, max_val, (n_values,)).astype(
                mx.uint32
            )
            packed, n = BitPackedQuantizer.pack(codes, bits)
            assert n == n_values
            unpacked = BitPackedQuantizer.unpack(packed, n, bits)
            assert mx.all(unpacked == codes).item()

    def test_boundary_exact_codes_per_word(self):
        for bits in range(2, 9):
            codes_per_word = 32 // bits
            max_val = (1 << bits) - 1
            # Generate codes that fit in one word and are within max_val
            codes = (mx.arange(codes_per_word) % (max_val + 1)).astype(
                mx.uint32
            )
            packed, n = BitPackedQuantizer.pack(codes, bits)
            unpacked = BitPackedQuantizer.unpack(packed, n, bits)
            assert mx.all(unpacked == codes).item()

    def test_all_zeros(self):
        for bits in range(2, 9):
            codes = mx.zeros((100,), dtype=mx.uint32)
            packed, n = BitPackedQuantizer.pack(codes, bits)
            unpacked = BitPackedQuantizer.unpack(packed, n, bits)
            assert mx.all(unpacked == codes).item()

    def test_all_max_values(self):
        for bits in range(2, 9):
            max_val = (1 << bits) - 1
            codes = mx.full((100,), max_val, dtype=mx.uint32)
            packed, n = BitPackedQuantizer.pack(codes, bits)
            unpacked = BitPackedQuantizer.unpack(packed, n, bits)
            assert mx.all(unpacked == codes).item()

    def test_roundtrip_from_int_dtypes(self):
        dtypes = [mx.int8, mx.int16, mx.int32, mx.int64]
        for dtype in dtypes:
            codes = mx.array([0, 1, 2, 3], dtype=dtype)
            packed, n = BitPackedQuantizer.pack(codes, bits=4)
            unpacked = BitPackedQuantizer.unpack(packed, n, bits=4)
            assert mx.all(unpacked == codes).item()
