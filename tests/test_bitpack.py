#!/usr/bin/env python3
"""
RFSN v10 - Bitpack Stress and Benchmark Tests.
Deterministic correctness tests for BitPackedQuantizer.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
from rfsn_v10.bitpack import BitPackedQuantizer

# --- Roundtrip edge cases ---

def test_roundtrip_all_zeros():
    for bits in range(2, 9):
        x = mx.zeros((1000,), dtype=mx.uint32)
        packed, n = BitPackedQuantizer.pack(x, bits)
        unpacked = BitPackedQuantizer.unpack(packed, n, bits)
        mx.eval(unpacked)
        assert mx.array_equal(x, unpacked).item()


def test_roundtrip_all_max_codes():
    for bits in range(2, 9):
        max_val = (1 << bits) - 1
        x = mx.array([max_val] * 1000, dtype=mx.uint32)
        packed, n = BitPackedQuantizer.pack(x, bits)
        unpacked = BitPackedQuantizer.unpack(packed, n, bits)
        mx.eval(unpacked)
        assert mx.array_equal(x, unpacked).item()


def test_roundtrip_alternating_min_max():
    for bits in range(2, 9):
        max_val = (1 << bits) - 1
        vals = [0 if i % 2 == 0 else max_val for i in range(1000)]
        x = mx.array(vals, dtype=mx.uint32)
        packed, n = BitPackedQuantizer.pack(x, bits)
        unpacked = BitPackedQuantizer.unpack(packed, n, bits)
        mx.eval(unpacked)
        assert mx.array_equal(x, unpacked).item()


def test_roundtrip_random_distribution():
    mx.random.seed(42)
    for bits in [3, 5, 8]:
        max_val = (1 << bits) - 1
        x = mx.random.randint(0, max_val + 1, (10000,), dtype=mx.uint32)
        packed, n = BitPackedQuantizer.pack(x, bits)
        unpacked = BitPackedQuantizer.unpack(packed, n, bits)
        mx.eval(unpacked)
        assert mx.array_equal(x, unpacked).item()


# --- Stress tests ---

@pytest.mark.parametrize("bits", [3, 8])
def test_stress_one_million_codes(bits: int):
    mx.random.seed(123)
    max_val = (1 << bits) - 1
    x = mx.random.randint(0, max_val + 1, (1_000_000,), dtype=mx.uint32)
    packed, n = BitPackedQuantizer.pack(x, bits)
    assert n == 1_000_000
    unpacked = BitPackedQuantizer.unpack(packed, n, bits)
    mx.eval(unpacked)
    assert mx.array_equal(x, unpacked).item()


# --- Compression ratio ---

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_compression_ratio(bits: int):
    """Verify packed size is approximately bits/32 of original."""
    n = 10000
    x = mx.zeros((n,), dtype=mx.uint32)
    packed, _ = BitPackedQuantizer.pack(x, bits)
    codes_per_word = 32 // bits
    expected_words = (n + codes_per_word - 1) // codes_per_word
    assert int(packed.size) == expected_words


# --- Rejection edge cases ---

def test_reject_bits_0():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([0], dtype=mx.uint32), 0)


def test_reject_bits_1():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([0], dtype=mx.uint32), 1)


def test_reject_bits_9():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([0], dtype=mx.uint32), 9)


def test_reject_bits_16():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([0], dtype=mx.uint32), 16)


def test_reject_bits_negative():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([0], dtype=mx.uint32), -1)


def test_reject_empty_pack():
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(mx.array([], dtype=mx.uint32), 3)


def test_reject_fractional_float():
    with pytest.raises(ValueError, match="integer"):
        BitPackedQuantizer.pack(mx.array([0.0, 1.5, 2.0], dtype=mx.float32), 3)


def test_reject_negative():
    with pytest.raises(ValueError, match="negative"):
        BitPackedQuantizer.pack(mx.array([0, 1, -1], dtype=mx.int32), 3)


def test_reject_out_of_range():
    with pytest.raises(ValueError, match="exceed"):
        BitPackedQuantizer.pack(mx.array([0, 1, 8], dtype=mx.uint32), 3)


# --- Unpack rejections ---

def test_unpack_reject_empty_buffer():
    with pytest.raises(ValueError, match="empty"):
        BitPackedQuantizer.unpack(mx.array([], dtype=mx.uint32), n_values=10, bits=3)


def test_unpack_reject_bad_n_values():
    with pytest.raises(ValueError, match="n_values"):
        BitPackedQuantizer.unpack(mx.array([0], dtype=mx.uint32), n_values=0, bits=3)


def test_unpack_reject_too_small():
    with pytest.raises(ValueError, match="too small"):
        BitPackedQuantizer.unpack(mx.array([0], dtype=mx.uint32), n_values=100, bits=3)
