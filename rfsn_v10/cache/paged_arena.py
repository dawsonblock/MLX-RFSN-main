"""PagedPackedArena — fixed-capacity paged storage for packed KV blocks.

Phase 5: Replace concatenated historical buffers with a paged arena where:
  * Appending one sealed block is O(block size).
  * Existing packed blocks are never copied during append.
  * Page metadata grows independently of packed payload.
  * The kernel can iterate pages directly.
  * Cache reset releases or reuses pages.
  * Multiple cache instances do not share mutable state.
  * Maximum capacity is explicit.

Design
------
Pages are stored as references to MLX arrays in a Python list.
No concatenation ever occurs.  Each append adds one reference.
The page table maps logical block indices → physical page indices.
"""
from __future__ import annotations

from typing import Any

from .contracts import PackedBlock


class PagedPackedArena:
    """Fixed-capacity paged arena for packed KV blocks.

    Parameters
    ----------
    max_pages
        Maximum number of blocks that can be stored.
    block_tokens
        Number of tokens per block (e.g. 64).
    head_dim
        Head dimension (e.g. 64 or 128).
    n_kv_heads
        Number of KV heads.
    bits
        Quantization bit width (8 for K8/V8).
    group_size
        Scale group size (64 for GS64).
    """

    def __init__(
        self,
        max_pages: int,
        block_tokens: int,
        head_dim: int,
        n_kv_heads: int,
        bits: int,
        group_size: int,
        name: str = "arena",
    ) -> None:
        self.max_pages = max_pages
        self.block_tokens = block_tokens
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.bits = bits
        self.group_size = group_size
        self.name = name

        # Scale tokens per group
        self._scales_per_block = block_tokens // group_size

        # Page table: maps logical block index → physical page index
        self._page_table: list[int] = []

        # Physical pages: list of dicts with 'codes', 'scales', 'signs'
        self._pages: list[dict[str, Any]] = []

        # Free page pool: physical page indices available for reuse
        self._free_pages: list[int] = []

        # Block metadata (logical position, layer_id, stream_id)
        self._block_meta: list[dict] = []

        # Instrumentation counters
        self._page_allocation_count: int = 0
        self._page_reuse_count: int = 0
        self._append_copy_bytes: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Number of logical blocks currently stored."""
        return len(self._page_table)

    @property
    def num_pages(self) -> int:
        """Number of physical pages currently allocated (in use + free)."""
        return len(self._pages)

    @property
    def packed_payload_bytes(self) -> int:
        """Logical payload size of stored blocks."""
        total = 0
        for page in self._pages:
            codes = page.get("codes")
            if codes is not None and hasattr(codes, "size"):
                from rfsn_v10.cache.contracts import _array_itemsize
                total += int(codes.size) * _array_itemsize(codes)
            scales = page.get("scales")
            if scales is not None and hasattr(scales, "size"):
                from rfsn_v10.cache.contracts import _array_itemsize
                total += int(scales.size) * _array_itemsize(scales)
        return total

    @property
    def metadata_bytes(self) -> int:
        """Page table + metadata overhead."""
        return (
            len(self._page_table) * 8  # int64 per entry
            + len(self._block_meta) * 64  # rough dict overhead
            + len(self._pages) * 48  # page dict overhead
        )

    @property
    def page_table_bytes(self) -> int:
        """Page table array bytes."""
        return self.max_pages * 8

    @property
    def append_copy_bytes(self) -> int:
        """Cumulative bytes copied during append operations."""
        return self._append_copy_bytes

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def append_block(self, block: PackedBlock) -> None:
        """Append one sealed packed block.

        Uses free-page pool if available, otherwise allocates a new page.
        Existing pages are never touched.
        """
        if self.num_blocks >= self.max_pages:
            raise RuntimeError(
                f"PagedPackedArena '{self.name}' capacity exceeded: "
                f"{self.num_blocks} >= {self.max_pages} pages"
            )

        # Reuse a free page slot if available, otherwise allocate new
        if self._free_pages:
            page_idx = self._free_pages.pop(0)
            self._page_reuse_count += 1
        else:
            page_idx = len(self._pages)
            self._pages.append({})  # placeholder, filled below
            self._page_allocation_count += 1

        # Store page data (reference, not copy — MLX arrays are immutable)
        page_data: dict[str, Any] = {
            "codes": block.packed_codes,
            "scales": block.scales,
        }
        if hasattr(block, "hash_signs") and block.hash_signs is not None:
            page_data["signs"] = block.hash_signs

        self._pages[page_idx] = page_data

        # Track bytes (the append "cost" is the size of the new block)
        if block.packed_codes is not None and hasattr(block.packed_codes, "size"):
            from rfsn_v10.cache.contracts import _array_itemsize
            self._append_copy_bytes += int(block.packed_codes.size) * _array_itemsize(block.packed_codes)
        if block.scales is not None and hasattr(block.scales, "size"):
            from rfsn_v10.cache.contracts import _array_itemsize
            self._append_copy_bytes += int(block.scales.size) * _array_itemsize(block.scales)

        # Update page table and metadata
        logical_idx = len(self._page_table)
        self._page_table.append(page_idx)
        self._block_meta.append(
            {
                "logical_index": logical_idx,
                "physical_page": page_idx,
                "logical_start": block.logical_start,
                "layer_id": getattr(block, "layer_id", 0),
                "stream_id": getattr(block, "stream_id", 0),
            }
        )

    def reset(self) -> None:
        """Clear all blocks. Physical pages are returned to the free pool for reuse."""
        # Return all allocated physical pages to the free pool
        for page_idx in self._page_table:
            self._free_pages.append(page_idx)
            # Dereference the actual data for GC
            self._pages[page_idx] = {}
        self._page_table.clear()
        self._block_meta.clear()
        self._append_copy_bytes = 0

    def get_block(self, logical_index: int) -> dict[str, Any]:
        """Return the block data at the given logical index."""
        if logical_index < 0 or logical_index >= len(self._page_table):
            raise IndexError(f"Logical block index {logical_index} out of range")
        page_idx = self._page_table[logical_index]
        return {
            **self._pages[page_idx],
            "meta": self._block_meta[logical_index],
        }

    def iter_blocks(self):
        """Yield each stored block in logical order."""
        for i in range(len(self._page_table)):
            yield self.get_block(i)

    def to_instrumentation(self) -> dict:
        """Return instrumentation counters for memory reporting."""
        return {
            "packed_payload_bytes": self.packed_payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "page_table_bytes": self.page_table_bytes,
            "dense_tail_bytes": 0,  # managed separately
            "scratch_peak_bytes": 0,
            "append_copy_bytes": self.append_copy_bytes,
            "page_allocation_count": self._page_allocation_count,
            "page_reuse_count": self._page_reuse_count,
            "max_pages": self.max_pages,
            "num_blocks": self.num_blocks,
        }
