"""Tests for PagedPackedArena — fixed-capacity paged storage.

Phase 5 audit fix: Verify append, reset, get_block, iter_blocks,
capacity enforcement, instrumentation, and page reuse.
"""
from __future__ import annotations

import pytest

from rfsn_v10.cache.contracts import PackedBlock
from rfsn_v10.cache.paged_arena import PagedPackedArena


def _make_fake_block(
    codes_size: int = 100,
    scales_size: int = 10,
    logical_start: int = 0,
) -> PackedBlock:
    """Create a PackedBlock with minimal fake arrays."""
    try:
        import mlx.core as mx
        codes = mx.zeros((codes_size,), dtype=mx.uint32)
        scales = mx.zeros((scales_size,), dtype=mx.float32)
        return PackedBlock(
            packed_codes=codes,
            scales=scales,
            token_count=64,
            bits=8,
            group_size=64,
            n_values=codes_size,
            logical_start=logical_start,
            head_dim=64,
            num_elements=codes_size,
        )
    except ImportError:
        # Portable fallback: use simple objects with size/dtype
        class FakeArray:
            def __init__(self, size: int, dtype_name: str = "float32"):
                self.size = size
                self.dtype = FakeDtype(dtype_name)

        class FakeDtype:
            def __init__(self, name: str):
                self.name = name
                self.size = 4 if "float" in name else 1 if "uint8" in name else 4

        block = PackedBlock(
            packed_codes=FakeArray(codes_size, "uint32"),
            scales=FakeArray(scales_size, "float32"),
            token_count=64,
            bits=8,
            group_size=64,
            n_values=codes_size,
            logical_start=logical_start,
            head_dim=64,
            num_elements=codes_size,
        )
        return block


def test_arena_init() -> None:
    arena = PagedPackedArena(
        max_pages=4,
        block_tokens=64,
        head_dim=64,
        n_kv_heads=2,
        bits=8,
        group_size=64,
        name="test",
    )
    assert arena.num_blocks == 0
    assert arena.num_pages == 0
    assert arena.max_pages == 4
    assert arena.name == "test"


def test_append_and_get() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    block = _make_fake_block(logical_start=0)
    arena.append_block(block)

    assert arena.num_blocks == 1
    assert arena.num_pages == 1

    retrieved = arena.get_block(0)
    assert retrieved["meta"]["logical_index"] == 0
    assert retrieved["meta"]["logical_start"] == 0


def test_append_multiple_logical_order() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    for i in range(3):
        arena.append_block(_make_fake_block(logical_start=i * 64))

    assert arena.num_blocks == 3
    logical_starts = [arena.get_block(i)["meta"]["logical_start"] for i in range(3)]
    assert logical_starts == [0, 64, 128]


def test_iter_blocks() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    for i in range(3):
        arena.append_block(_make_fake_block(logical_start=i * 64))

    starts = [b["meta"]["logical_start"] for b in arena.iter_blocks()]
    assert starts == [0, 64, 128]


def test_capacity_exceeded() -> None:
    arena = PagedPackedArena(
        max_pages=2, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    arena.append_block(_make_fake_block(logical_start=0))
    arena.append_block(_make_fake_block(logical_start=64))

    with pytest.raises(RuntimeError, match="capacity exceeded"):
        arena.append_block(_make_fake_block(logical_start=128))


def test_reset_clears_all() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    for i in range(3):
        arena.append_block(_make_fake_block(logical_start=i * 64))

    assert arena.num_blocks == 3
    arena.reset()
    assert arena.num_blocks == 0
    # num_pages now counts allocated physical pages (including free pool)
    assert arena.num_pages == 3
    # _page_allocation_count is cumulative (not reset)
    assert arena._page_allocation_count == 3
    assert arena._page_reuse_count == 0
    assert arena._append_copy_bytes == 0


def test_reset_reuses_pages() -> None:
    """After reset, appending should reuse page slots (audit fix)."""
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    arena.append_block(_make_fake_block(logical_start=0))
    arena.append_block(_make_fake_block(logical_start=64))
    assert arena._page_allocation_count == 2

    arena.reset()
    # After reset, free pool should have 2 pages
    assert len(arena._free_pages) == 2
    # Re-append — should reuse page 0 slot
    arena.append_block(_make_fake_block(logical_start=128))
    # Audit fix: page_reuse_count should now be > 0
    assert arena._page_reuse_count > 0


def test_instrumentation_nonzero() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    arena.append_block(_make_fake_block(codes_size=100, scales_size=10))

    inst = arena.to_instrumentation()
    assert inst["num_blocks"] == 1
    assert inst["max_pages"] == 4
    assert inst["page_allocation_count"] == 1
    assert inst["packed_payload_bytes"] > 0
    assert inst["metadata_bytes"] > 0


def test_append_copy_bytes_accounting() -> None:
    """Verify that append_copy_bytes reflects actual new data size."""
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    block = _make_fake_block(codes_size=100, scales_size=10)
    arena.append_block(block)

    # Should be nonzero (the size of the referenced arrays)
    assert arena.append_copy_bytes > 0


def test_get_block_out_of_range() -> None:
    arena = PagedPackedArena(
        max_pages=4, block_tokens=64, head_dim=64,
        n_kv_heads=2, bits=8, group_size=64,
    )
    with pytest.raises(IndexError):
        arena.get_block(0)
