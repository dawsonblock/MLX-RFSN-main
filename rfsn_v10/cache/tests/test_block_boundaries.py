"""Block boundary tests for staging capacity and block sealing.

Tests token counts at block boundaries (63, 64, 65, 127, 128, 129, etc.)
to verify correct block sealing behavior with staging_capacity=64.
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
def test_block_boundary_63_no_seal() -> None:
    """63 tokens: one short of block capacity, should not seal."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 63 tokens (one short of block capacity)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 63, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 63, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 63 tokens in staging, no sealed blocks
    assert layer_cache.total_token_count() == 63
    assert len(list(layer_cache.iter_key_blocks())) == 0
    # Note: runtime_counters not auto-incremented by layer cache append


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_boundary_64_exact_seal() -> None:
    """64 tokens: exactly block capacity, should seal one block."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 64 tokens (exactly block capacity)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 64, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 64, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 0 tokens in staging, one sealed block
    assert layer_cache.total_token_count() == 64
    assert len(list(layer_cache.iter_key_blocks())) == 1

    # Verify block properties
    block = list(layer_cache.iter_key_blocks())[0]
    assert block.token_count == 64
    assert block.logical_start == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_boundary_65_seal_plus_one() -> None:
    """65 tokens: one block sealed, one token in staging."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 65 tokens (one block + one token)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 65, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 65, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 1 token in staging, one sealed block
    assert layer_cache.total_token_count() == 65
    assert len(list(layer_cache.iter_key_blocks())) == 1

    # Verify block properties
    block = list(layer_cache.iter_key_blocks())[0]
    assert block.token_count == 64
    assert block.logical_start == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_boundary_127_no_second_seal() -> None:
    """127 tokens: one block sealed, 63 in staging (one short of second block)."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 127 tokens (one block + 63 tokens)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 127, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 127, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 63 tokens in staging, one sealed block
    assert layer_cache.total_token_count() == 127
    assert len(list(layer_cache.iter_key_blocks())) == 1


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_boundary_128_two_blocks() -> None:
    """128 tokens: exactly two blocks sealed."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 128 tokens (exactly two blocks)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 128, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 128, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 0 tokens in staging, two sealed blocks
    assert layer_cache.total_token_count() == 128
    assert len(list(layer_cache.iter_key_blocks())) == 2

    # Verify block properties
    blocks = list(layer_cache.iter_key_blocks())
    assert blocks[0].token_count == 64
    assert blocks[0].logical_start == 0
    assert blocks[1].token_count == 64
    assert blocks[1].logical_start == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_boundary_129_two_blocks_plus_one() -> None:
    """129 tokens: two blocks sealed, one token in staging."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)

    # Append 129 tokens (two blocks + one token)
    B, Hkv, D = 1, 2, 64
    keys = mx.random.normal(shape=(B, Hkv, 129, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 129, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    # Should have 1 token in staging, two sealed blocks
    assert layer_cache.total_token_count() == 129
    assert len(list(layer_cache.iter_key_blocks())) == 2


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_incremental_append_crosses_boundary() -> None:
    """Test incremental appends that cross block boundaries."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Append 60 tokens (no seal)
    keys = mx.random.normal(shape=(B, Hkv, 60, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 60, D)).astype(mx.float32)
    layer_cache.append(keys, values)
    assert len(list(layer_cache.iter_key_blocks())) == 0

    # Append 10 more (total 70, should seal one block)
    keys = mx.random.normal(shape=(B, Hkv, 10, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 10, D)).astype(mx.float32)
    layer_cache.append(keys, values)
    assert layer_cache.total_token_count() == 70
    assert len(list(layer_cache.iter_key_blocks())) == 1

    # Append 60 more (total 130, should seal second block)
    keys = mx.random.normal(shape=(B, Hkv, 60, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 60, D)).astype(mx.float32)
    layer_cache.append(keys, values)
    assert layer_cache.total_token_count() == 130
    assert len(list(layer_cache.iter_key_blocks())) == 2


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_block_positions_monotonic() -> None:
    """Verify block positions are monotonic and non-overlapping."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec, staging_capacity=64)

    layer_cache = session.get_layer_cache(0)
    B, Hkv, D = 1, 2, 64

    # Append 200 tokens (3 blocks + 8 tokens)
    keys = mx.random.normal(shape=(B, Hkv, 200, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, Hkv, 200, D)).astype(mx.float32)
    layer_cache.append(keys, values)

    blocks = list(layer_cache.iter_key_blocks())
    assert len(blocks) == 3

    # Verify monotonic positions
    positions = [b.logical_start for b in blocks]
    assert positions == [0, 64, 128]

    # Verify non-overlapping
    for i in range(len(blocks) - 1):
        assert blocks[i].logical_start + blocks[i].token_count <= blocks[i + 1].logical_start
