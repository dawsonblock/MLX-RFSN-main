"""PagedKVArena — fixed-capacity GPU-resident packed KV arena.

Design
------
* One combined arena for K and V (never inconsistent).
* Preallocated contiguous MLX arrays divided into logical pages.
* New pages are written in-place via indexed update; historical pages
  are never moved, copied, or reconstructed.
* Page table maps logical → physical pages at runtime.
* Kernel reads the arena arrays directly without Python concatenation.

Invariants
----------
* history_recopy_bytes == 0  (appending page N never touches pages 0..N-1)
* page_write_bytes grows linearly with total tokens
* reserved_capacity_bytes stays constant after init
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfsn_v10.compat import mx

from .contracts import PackedBlock


@dataclass(frozen=True)
class PagedKVView:
    """Read-only view into a PagedKVArena for kernel consumption."""

    k_codes: Any
    k_scales: Any
    v_codes: Any
    v_scales: Any

    page_table: Any
    page_starts: Any
    page_counts: Any

    num_pages: int
    max_pages: int
    page_tokens: int

    k_words_per_vector: int
    v_words_per_vector: int
    k_groups_per_vector: int
    v_groups_per_vector: int


def paged_view_from_blocks(
    key_blocks: list[PackedBlock],
    value_blocks: list[PackedBlock],
    *,
    max_pages: int | None = None,
) -> PagedKVView:
    """Build a temporary PagedKVView by writing blocks into a preallocated arena.

    Useful for tests and reference paths that still operate on block lists
    but need to exercise the paged kernel interface.
    """
    if not key_blocks or not value_blocks:
        raise ValueError("at least one key/value block pair required")
    if len(key_blocks) != len(value_blocks):
        raise ValueError("key/value block count mismatch")

    n_kv_heads = key_blocks[0].n_kv_heads
    head_dim = key_blocks[0].head_dim
    bits = key_blocks[0].bits
    group_size = key_blocks[0].group_size

    # PackedBlockV4 has words_per_vector/groups_per_vector; PackedBlock does not.
    if hasattr(key_blocks[0], "words_per_vector"):
        k_words = key_blocks[0].words_per_vector
        v_words = value_blocks[0].words_per_vector
        k_groups = key_blocks[0].groups_per_vector
        v_groups = value_blocks[0].groups_per_vector
    else:
        import math
        codes_per_word = 32 // bits
        k_words = math.ceil(head_dim / codes_per_word)
        v_words = k_words
        k_groups = head_dim // group_size
        v_groups = k_groups
    page_tokens = key_blocks[0].token_count

    num_pages = len(key_blocks)
    if max_pages is None:
        max_pages = num_pages
    if num_pages > max_pages:
        raise ValueError(f"too many blocks for max_pages: {num_pages} > {max_pages}")

    arena = PagedKVArena(
        max_pages=max_pages,
        page_tokens=page_tokens,
        n_kv_heads=n_kv_heads,
        k_words_per_vector=k_words,
        v_words_per_vector=v_words,
        k_groups_per_vector=k_groups,
        v_groups_per_vector=v_groups,
    )

    for kb, vb in zip(key_blocks, value_blocks):
        arena.append(kb, vb)

    return arena.view()


class PagedKVArena:
    """Fixed-capacity GPU-resident packed KV arena.

    Parameters
    ----------
    max_pages
        Maximum number of pages that can be stored.
    page_tokens
        Number of tokens per page (e.g. 64).
    n_kv_heads
        Number of KV heads.
    k_words_per_vector
        Number of uint32 words per key vector.
    v_words_per_vector
        Number of uint32 words per value vector.
    k_groups_per_vector
        Number of scale groups per key vector.
    v_groups_per_vector
        Number of scale groups per value vector.
    """

    def __init__(
        self,
        *,
        max_pages: int,
        page_tokens: int,
        n_kv_heads: int,
        k_words_per_vector: int,
        v_words_per_vector: int,
        k_groups_per_vector: int,
        v_groups_per_vector: int,
    ) -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if page_tokens <= 0:
            raise ValueError("page_tokens must be positive")

        self.max_pages = max_pages
        self.page_tokens = page_tokens
        self.n_kv_heads = n_kv_heads

        self.k_words_per_vector = k_words_per_vector
        self.v_words_per_vector = v_words_per_vector
        self.k_groups_per_vector = k_groups_per_vector
        self.v_groups_per_vector = v_groups_per_vector

        # Persistent GPU payload.
        self.k_codes = mx.zeros(
            (n_kv_heads, max_pages, page_tokens, k_words_per_vector),
            dtype=mx.uint32,
        )
        self.k_scales = mx.zeros(
            (n_kv_heads, max_pages, page_tokens, k_groups_per_vector),
            dtype=mx.float16,
        )
        self.v_codes = mx.zeros(
            (n_kv_heads, max_pages, page_tokens, v_words_per_vector),
            dtype=mx.uint32,
        )
        self.v_scales = mx.zeros(
            (n_kv_heads, max_pages, page_tokens, v_groups_per_vector),
            dtype=mx.float16,
        )

        # GPU-visible logical-to-physical mapping.
        self.page_table = mx.zeros((max_pages,), dtype=mx.int32)
        self.page_starts = mx.zeros((max_pages,), dtype=mx.int32)
        self.page_counts = mx.zeros((max_pages,), dtype=mx.int32)

        self._num_pages = 0
        self._next_physical_page = 0

        # Stored block references for fallback paths and reference tests.
        # The arena arrays are the canonical GPU storage; these Python objects
        # are lightweight metadata wrappers used only by backward-compatible
        # iterators.  They do not trigger extra device copies.
        self._blocks: list[tuple[PackedBlock, PackedBlock]] = []

        self.page_write_bytes = 0
        self.history_recopy_bytes = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def reserved_capacity_bytes(self) -> int:
        """Total bytes of the preallocated arena (active + inactive)."""
        return sum(
            int(a.size) * a.dtype.size
            for a in (
                self.k_codes,
                self.k_scales,
                self.v_codes,
                self.v_scales,
                self.page_table,
                self.page_starts,
                self.page_counts,
            )
        )

    @property
    def active_payload_bytes(self) -> int:
        """Bytes of the actually-written pages (excluding inactive arena)."""
        if self._num_pages == 0:
            return 0
        active_tokens = sum(
            int(self.page_counts[i]) for i in range(self._num_pages)
        )
        # Each token contributes: codes + scales for K and V
        bytes_per_token_kv = (
            self.k_words_per_vector * 4  # uint32
            + self.k_groups_per_vector * 2  # float16
            + self.v_words_per_vector * 4  # uint32
            + self.v_groups_per_vector * 2  # float16
        )
        return active_tokens * self.n_kv_heads * bytes_per_token_kv

    @property
    def page_metadata_bytes(self) -> int:
        """Bytes for page table, starts, and counts."""
        return (
            int(self.page_table.size) * self.page_table.dtype.size
            + int(self.page_starts.size) * self.page_starts.dtype.size
            + int(self.page_counts.size) * self.page_counts.dtype.size
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def append(
        self,
        key_block: PackedBlock,
        value_block: PackedBlock,
    ) -> None:
        """Append one sealed packed page without copying history.

        The page is written into the next physical slot and published
        atomically via the page table after ``mx.eval``.
        """
        self._validate_pair(key_block, value_block)

        if self._num_pages >= self.max_pages:
            raise RuntimeError(
                f"KV arena full: {self._num_pages}/{self.max_pages} pages"
            )

        logical_page = self._num_pages
        physical_page = self._next_physical_page
        count = int(key_block.token_count)

        # Write only the new page. Existing payload is untouched.
        # Blocks arrive as [B, Hkv, T, words] with B == 1.
        self.k_codes[:, physical_page, :count, :] = key_block.packed_codes[0]
        self.k_scales[:, physical_page, :count, :] = key_block.scales[0]
        self.v_codes[:, physical_page, :count, :] = value_block.packed_codes[0]
        self.v_scales[:, physical_page, :count, :] = value_block.scales[0]

        # Publish the page through metadata.
        self.page_table[logical_page] = physical_page
        self.page_starts[logical_page] = int(key_block.logical_start)
        self.page_counts[logical_page] = count

        # Resolve the page writes before publishing the host page count.
        mx.eval(
            self.k_codes,
            self.k_scales,
            self.v_codes,
            self.v_scales,
            self.page_table,
            self.page_starts,
            self.page_counts,
        )

        self._blocks.append((key_block, value_block))
        self._num_pages += 1
        self._next_physical_page += 1

        self.page_write_bytes += (
            key_block.payload_bytes() + value_block.payload_bytes()
        )

    def _validate_pair(
        self,
        key_block: PackedBlock,
        value_block: PackedBlock,
    ) -> None:
        if key_block.logical_start != value_block.logical_start:
            raise ValueError("K/V logical_start mismatch")

        if key_block.token_count != value_block.token_count:
            raise ValueError("K/V token_count mismatch")

        if key_block.token_count > self.page_tokens:
            raise ValueError("block exceeds page capacity")

        if key_block.batch_size != 1 or value_block.batch_size != 1:
            raise ValueError("paged arena currently requires batch size 1")

        if key_block.n_kv_heads != self.n_kv_heads:
            raise ValueError("key head count mismatch")

        if value_block.n_kv_heads != self.n_kv_heads:
            raise ValueError("value head count mismatch")

        expected_start = self._num_pages * self.page_tokens
        if key_block.logical_start != expected_start:
            raise ValueError(
                f"non-contiguous page: expected {expected_start}, "
                f"got {key_block.logical_start}"
            )

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def view(self) -> PagedKVView:
        return PagedKVView(
            k_codes=self.k_codes,
            k_scales=self.k_scales,
            v_codes=self.v_codes,
            v_scales=self.v_scales,
            page_table=self.page_table,
            page_starts=self.page_starts,
            page_counts=self.page_counts,
            num_pages=self._num_pages,
            max_pages=self.max_pages,
            page_tokens=self.page_tokens,
            k_words_per_vector=self.k_words_per_vector,
            v_words_per_vector=self.v_words_per_vector,
            k_groups_per_vector=self.k_groups_per_vector,
            v_groups_per_vector=self.v_groups_per_vector,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_key_blocks(self):
        """Yield each sealed key block in logical order.

        Backward-compatible iterator for fallback paths and tests.
        """
        for kb, _vb in self._blocks:
            yield kb

    def iter_value_blocks(self):
        """Yield each sealed value block in logical order.

        Backward-compatible iterator for fallback paths and tests.
        """
        for _kb, vb in self._blocks:
            yield vb

    def reset(self) -> None:
        """Hide all pages without zeroing the arena.

        Old page contents are ignored because ``num_pages`` becomes zero.
        Zeroing hundreds of megabytes on every reset is pointless.
        """
        self._num_pages = 0
        self._next_physical_page = 0
        self._blocks.clear()
        self.page_write_bytes = 0
        self.history_recopy_bytes = 0

    def to_instrumentation(self) -> dict:
        """Return instrumentation counters for memory reporting."""
        return {
            "num_pages": self._num_pages,
            "max_pages": self.max_pages,
            "page_tokens": self.page_tokens,
            "reserved_capacity_bytes": self.reserved_capacity_bytes,
            "active_payload_bytes": self.active_payload_bytes,
            "page_metadata_bytes": self.page_metadata_bytes,
            "page_write_bytes": self.page_write_bytes,
            "history_recopy_bytes": self.history_recopy_bytes,
        }
