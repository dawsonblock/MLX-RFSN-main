"""RoPE (Rotary Position Encoding) offset validation tests.

Tests for verifying that RoPE offsets are correctly applied across
different generation scenarios, including:
- Single-turn generation
- Multi-turn generation with context continuation
- Long context generation
- Block boundary crossings
"""
from __future__ import annotations

import pytest

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.session import GenerationCacheSession

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_rope_block_position_monotonicity() -> None:
    """Verify block positions are monotonic for correct RoPE application.

    RoPE requires position IDs to be monotonically increasing.
    If blocks are out of order or have overlapping positions, RoPE will
    produce incorrect results.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Append 200 tokens (3 blocks + 8 tokens)
    keys = mx.random.normal(shape=(B, Hkv, 200, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 200, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Verify blocks are monotonically positioned
    blocks = list(layer_cache.iter_key_blocks())
    assert len(blocks) == 3

    positions = [b.logical_start for b in blocks]
    assert positions == [0, 64, 128], (
        f"Block positions should be [0, 64, 128], got {positions}"
    )

    # Verify no overlap
    for i in range(len(blocks) - 1):
        assert blocks[i].logical_start + blocks[i].token_count <= blocks[i + 1].logical_start, (
            f"Blocks {i} and {i+1} overlap"
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_rope_multi_turn_continuation() -> None:
    """Verify RoPE offsets are correct across multi-turn generation.

    In multi-turn scenarios, the context continues from previous turns.
    RoPE offsets must be correctly maintained across turns.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Turn 1: Append 100 tokens
    keys1 = mx.random.normal(shape=(B, Hkv, 100, D)).astype(mx.float32)
    values1 = mx.random.normal(shape=(B, Hkv, 100, D)).astype(mx.float32)
    layer_cache.append(keys1, values1)

    # Verify blocks are correctly positioned
    blocks_turn1 = list(layer_cache.iter_key_blocks())
    assert len(blocks_turn1) == 1
    assert blocks_turn1[0].logical_start == 0
    assert blocks_turn1[0].token_count == 64

    # Turn 2: Append 50 more tokens (continuation)
    keys2 = mx.random.normal(shape=(B, Hkv, 50, D)).astype(mx.float32)
    values2 = mx.random.normal(shape=(B, Hkv, 50, D)).astype(mx.float32)
    layer_cache.append(keys2, values2)

    # Verify new block starts at position 64
    blocks_turn2 = list(layer_cache.iter_key_blocks())
    assert len(blocks_turn2) == 2
    assert blocks_turn2[0].logical_start == 0
    assert blocks_turn2[1].logical_start == 64

    # Total tokens should be 150
    assert layer_cache.total_token_count() == 150


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_rope_long_context_positioning():
    """Verify RoPE positioning for long context generation.

    Long contexts (e.g., 2000+ tokens) stress the position encoding
    and block management. This test ensures positions remain correct.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Append 512 tokens (8 blocks)
    keys = mx.random.normal(shape=(B, Hkv, 512, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 512, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Verify all blocks are correctly positioned
    blocks = list(layer_cache.iter_key_blocks())
    assert len(blocks) == 8

    expected_positions = [i * 64 for i in range(8)]
    actual_positions = [b.logical_start for b in blocks]
    assert actual_positions == expected_positions, (
        f"Expected positions {expected_positions}, got {actual_positions}"
    )

    # Verify no gaps or overlaps
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1].logical_start - (blocks[i].logical_start + blocks[i].token_count)
        assert gap == 0, f"Gap between blocks {i} and {i+1}: {gap}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_rope_incremental_append_positioning():
    """Verify RoPE positioning with incremental appends.

    Real generation appends tokens incrementally. This test ensures
    block positions remain correct across multiple small appends.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Incremental appends of 10 tokens each
    for i in range(20):  # 20 * 10 = 200 tokens
        keys = mx.random.normal(shape=(B, Hkv, 10, D)).astype(mx.float32)
        values = mx.random.normal(shape=(B, Hkv, 10, D)).astype(mx.float32)
        layer_cache.append(keys, values)

    # Verify all blocks are correctly positioned
    blocks = list(layer_cache.iter_key_blocks())
    assert len(blocks) == 3  # 200 tokens / 64 = 3 blocks + 8 tokens

    expected_positions = [0, 64, 128]
    actual_positions = [b.logical_start for b in blocks]
    assert actual_positions == expected_positions
