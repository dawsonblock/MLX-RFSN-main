"""Tests for PagedKVArena — GPU-resident fixed-capacity packed KV arena.

Invariants verified:
  * Appending one page is O(page size).
  * Historical pages are never copied during append.
  * Page metadata grows independently of payload.
  * Reset hides pages without zeroing the arena.
"""
from __future__ import annotations

import pytest

from rfsn_v10.cache.contracts import PackedBlock
from rfsn_v10.cache.paged_arena import PagedKVArena, PagedKVView, paged_view_from_blocks

HAS_MLX = False
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore


def _make_block(
    logical_start: int = 0,
    token_count: int = 64,
    n_kv_heads: int = 2,
    words_per_vector: int = 16,
    groups_per_vector: int = 1,
) -> PackedBlock:
    """Create a fake PackedBlock suitable for arena append."""
    if HAS_MLX:
        codes = mx.zeros(
            (1, n_kv_heads, token_count, words_per_vector), dtype=mx.uint32
        )
        scales = mx.zeros(
            (1, n_kv_heads, token_count, groups_per_vector), dtype=mx.float16
        )
    else:
        class FakeArray:
            def __init__(self, size: int, dtype_name: str = "float32"):
                self.size = size
                self.dtype = FakeDtype(dtype_name)
                self.shape = (1, n_kv_heads, token_count, words_per_vector)

        class FakeDtype:
            def __init__(self, name: str):
                self.name = name
                self.size = 4 if "float" in name else 1 if "uint8" in name else 4

        codes = FakeArray(1 * n_kv_heads * token_count * words_per_vector, "uint32")
        scales = FakeArray(1 * n_kv_heads * token_count * groups_per_vector, "float16")

    return PackedBlock(
        packed_codes=codes,
        scales=scales,
        token_count=token_count,
        bits=8,
        group_size=64,
        n_values=codes.size,
        logical_start=logical_start,
        head_dim=64,
        num_elements=codes.size,
        batch_size=1,
        n_kv_heads=n_kv_heads,
    )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_arena_init() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    assert arena.num_pages == 0
    assert arena.max_pages == 4
    assert arena.page_tokens == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_one_page() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    key_block = _make_block(logical_start=0, token_count=64)
    value_block = _make_block(logical_start=0, token_count=64)
    arena.append(key_block, value_block)

    assert arena.num_pages == 1
    assert arena.history_recopy_bytes == 0
    assert arena.page_write_bytes > 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_multiple_pages() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    for i in range(3):
        kb = _make_block(logical_start=i * 64, token_count=64)
        vb = _make_block(logical_start=i * 64, token_count=64)
        arena.append(kb, vb)

    assert arena.num_pages == 3
    view = arena.view()
    assert view.num_pages == 3
    assert view.page_table[0] == 0
    assert view.page_table[1] == 1
    assert view.page_table[2] == 2


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_page_table_maps_logical_to_physical() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    for i in range(3):
        kb = _make_block(logical_start=i * 64, token_count=64)
        vb = _make_block(logical_start=i * 64, token_count=64)
        arena.append(kb, vb)

    view = arena.view()
    for i in range(3):
        assert int(view.page_starts[i]) == i * 64
        assert int(view.page_counts[i]) == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_capacity_overflow_fails() -> None:
    arena = PagedKVArena(
        max_pages=2,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    arena.append(_make_block(0, 64), _make_block(0, 64))
    arena.append(_make_block(64, 64), _make_block(64, 64))

    with pytest.raises(RuntimeError, match="arena full"):
        arena.append(_make_block(128, 64), _make_block(128, 64))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reset_hides_pages() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    for i in range(3):
        arena.append(_make_block(i * 64, 64), _make_block(i * 64, 64))

    assert arena.num_pages == 3
    arena.reset()
    assert arena.num_pages == 0
    assert arena.page_write_bytes == 0
    assert arena.history_recopy_bytes == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reset_does_not_reallocate() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    arena.append(_make_block(0, 64), _make_block(0, 64))
    old_codes = arena.k_codes
    arena.reset()
    assert arena.k_codes is old_codes


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_malformed_pair_fails() -> None:
    arena = PagedKVArena(
        max_pages=4,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    kb = _make_block(logical_start=0, token_count=64)
    vb = _make_block(logical_start=1, token_count=64)
    with pytest.raises(ValueError, match="logical_start mismatch"):
        arena.append(kb, vb)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_history_recopy_invariant() -> None:
    """Appending page 255 must never touch pages 0..254."""
    arena = PagedKVArena(
        max_pages=256,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    expected_write = 0
    for i in range(256):
        kb = _make_block(logical_start=i * 64, token_count=64)
        vb = _make_block(logical_start=i * 64, token_count=64)
        arena.append(kb, vb)
        expected_write += kb.payload_bytes() + vb.payload_bytes()

    assert arena.history_recopy_bytes == 0
    assert arena.page_write_bytes == expected_write


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_slab_growth() -> None:
    """Arena grows in slabs rather than reserving max_pages at init."""
    arena = PagedKVArena(
        max_pages=256,
        page_tokens=64,
        n_kv_heads=2,
        k_words_per_vector=16,
        v_words_per_vector=16,
        k_groups_per_vector=1,
        v_groups_per_vector=1,
    )
    # Initial capacity should be one slab, not the full 256 pages.
    assert arena.k_codes.shape[1] == 16

    for i in range(17):
        kb = _make_block(logical_start=i * 64, token_count=64)
        vb = _make_block(logical_start=i * 64, token_count=64)
        arena.append(kb, vb)

    # After crossing the first slab boundary, capacity should have grown.
    assert arena.k_codes.shape[1] == 32
    assert arena.num_pages == 17


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_paged_view_from_blocks_helper() -> None:
    blocks = []
    for i in range(3):
        kb = _make_block(logical_start=i * 64, token_count=64)
        vb = _make_block(logical_start=i * 64, token_count=64)
        blocks.append((kb, vb))

    k_blocks = [b[0] for b in blocks]
    v_blocks = [b[1] for b in blocks]
    view = paged_view_from_blocks(k_blocks, v_blocks)
    assert isinstance(view, PagedKVView)
    assert view.num_pages == 3
