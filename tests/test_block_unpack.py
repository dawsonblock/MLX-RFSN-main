#!/usr/bin/env python3
"""
RFSN v10 — Block-level unpacking acceptance tests.

Verifies that unpack_blocks reconstructs selected token blocks correctly,
preserves caller ordering, deduplicates, rejects invalid inputs, and
produces shapes consistent with the selected blocks.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.block_unpack import dequantize_full, dequantize_kv_blocks, unpack_blocks

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_test_packed(
    shape: tuple[int, ...] = (1, 4, 256, 64),
    bits: int = 4,
    block_size: int = 64,
    group_size: int = 64,
):
    """Create dummy packed codes and scales arrays sized for the given params."""
    b, h, t, d = shape
    # Packed codes: one uint32 per token per head per batch per feature group
    # Simplified sizing for test fixtures
    n_groups = (t * d) // group_size
    codes_per_word = 32 // bits
    n_words = (n_groups + codes_per_word - 1) // codes_per_word
    packed = mx.zeros((b, h, n_words), dtype=mx.uint32)
    # Scales: one float per group
    scales = mx.ones((b, h, n_groups), dtype=mx.float32)
    return packed, scales


# ------------------------------------------------------------------
# BlockUnpack tests
# ------------------------------------------------------------------

class TestBlockUnpack:
    def test_full_block_unpack_equals_full_dequant(self):
        """Unpacking all blocks individually should equal full dequant sliced by block."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)

        full = dequantize_full(packed, scales, bits=bits, group_size=group_size, shape=shape)
        all_block_ids = list(range(shape[2] // block_size))  # [0, 1, 2, 3]
        blocks = unpack_blocks(
            packed, scales, block_ids=all_block_ids, block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )

        assert mx.allclose(blocks, full, atol=1e-4).item()

    def test_random_block_ids_preserve_order(self):
        """Random block IDs should preserve order in output."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)

        full = dequantize_full(packed, scales, bits=bits, group_size=group_size, shape=shape)
        # Request blocks out of order: 2, 0, 1 (skip 3)
        block_ids = [2, 0, 1]
        blocks = unpack_blocks(
            packed, scales, block_ids=block_ids, block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )

        # Verify each returned slice corresponds to the requested block in order
        for idx, blk in enumerate(block_ids):
            expected = full[:, :, blk * block_size:(blk + 1) * block_size, :]
            actual = blocks[:, :, idx * block_size:(idx + 1) * block_size, :]
            assert mx.allclose(actual, expected, atol=1e-4).item()

    def test_duplicate_block_ids_deduplicated(self):
        """Duplicate block IDs handled gracefully (deduplicated)."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)

        full = dequantize_full(packed, scales, bits=bits, group_size=group_size, shape=shape)
        # Pass duplicates: 0, 0, 1, 1
        blocks = unpack_blocks(
            packed, scales, block_ids=[0, 0, 1, 1], block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )

        # Expect deduplicated to two unique blocks
        assert blocks.shape == (1, 4, 128, 64)
        expected_0 = full[:, :, 0:block_size, :]
        expected_1 = full[:, :, block_size:2 * block_size, :]
        assert mx.allclose(blocks[:, :, 0:block_size, :], expected_0, atol=1e-4).item()
        assert mx.allclose(blocks[:, :, block_size:2 * block_size, :], expected_1, atol=1e-4).item()

    def test_invalid_block_ids_rejected(self):
        """Invalid block IDs raise ValueError."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)

        with pytest.raises(ValueError, match="block index"):
            unpack_blocks(
                packed, scales, block_ids=[10], block_size=block_size,
                bits=bits, group_size=group_size, shape=shape,
            )

        with pytest.raises(ValueError, match="non-negative"):
            unpack_blocks(
                packed, scales, block_ids=[-1], block_size=block_size,
                bits=bits, group_size=group_size, shape=shape,
            )

    def test_shape_preserved(self):
        """Output shape matches expected for selected blocks."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)

        # Single block
        single = unpack_blocks(
            packed, scales, block_ids=[1], block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )
        assert single.shape == (1, 4, 64, 64)

        # Two non-contiguous blocks
        pair = unpack_blocks(
            packed, scales, block_ids=[0, 3], block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )
        assert pair.shape == (1, 4, 128, 64)

        # All four blocks
        all_blocks = unpack_blocks(
            packed, scales, block_ids=[0, 1, 2, 3], block_size=block_size,
            bits=bits, group_size=group_size, shape=shape,
        )
        assert all_blocks.shape == (1, 4, 256, 64)

    def test_dequantize_kv_blocks_runs(self):
        """dequantize_kv_blocks should delegate to k and v block paths."""
        shape = (1, 4, 256, 64)
        bits = 4
        block_size = 64
        group_size = 64
        packed, scales = _make_test_packed(shape, bits, block_size, group_size)
        num_blocks = shape[2] // block_size

        # Build minimal packet objects with 1D per-block scales
        class FakePacket:
            def __init__(self, packed, bits, num_blocks):
                self.k_packed = packed
                # dequantize_k_blocks expects scalar or 1D scale slices
                self.k_scales = mx.ones((num_blocks,), dtype=mx.float32)
                self.k_bits = bits
                self.v_packed = packed
                self.v_scales = mx.ones((num_blocks,), dtype=mx.float32)
                self.v_bits = bits
                self.block_size = block_size
                self.num_blocks = num_blocks
                self.k_block_packed_offsets = [0] * num_blocks
                self.k_block_scale_offsets = list(range(num_blocks))
                self.k_block_n_values = [block_size * 4 * 64] * num_blocks
                self.v_block_packed_offsets = [0] * num_blocks
                self.v_block_scale_offsets = list(range(num_blocks))
                self.v_block_n_values = [block_size * 4 * 64] * num_blocks

        k_pkt = FakePacket(packed, bits, num_blocks)
        v_pkt = FakePacket(packed, bits, num_blocks)
        k_out, v_out = dequantize_kv_blocks(k_pkt, v_pkt, [0, 1])
        assert k_out is not None
        assert v_out is not None
