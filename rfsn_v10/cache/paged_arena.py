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

from .contracts import (
    PackedBlock,
    PackedBlockV4,
    PackingLayout,
    Preconditioner,
    ScaleLayout,
    TensorLayout,
)


@dataclass(frozen=True)
class PagedKVFormat:
    """Immutable storage-format descriptor for a paged KV arena.

    Every append and every kernel dispatch must match the format frozen
    on the first page append.
    """

    format_version: int = 1
    key_bits: int = 8
    value_bits: int = 8
    key_group_size: int = 64
    value_group_size: int = 64
    head_dim: int = 64
    n_kv_heads: int = 1
    key_words_per_vector: int = 16
    value_words_per_vector: int = 16
    key_groups_per_vector: int = 1
    value_groups_per_vector: int = 1
    key_sign_seed: int = 42
    value_sign_seed: int = 42
    key_layer_id: int = 0
    value_layer_id: int = 0
    key_stream_id: str = "K"
    value_stream_id: str = "V"
    key_wht_applied: bool = True
    value_wht_applied: bool = True
    key_original_dtype: str = "float16"
    value_original_dtype: str = "float16"
    preconditioner: str = "WHT64_HASH_SIGN_V1"


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

    format: PagedKVFormat | None = None


def validate_direct_packed_format(
    key_codec: Any,
    value_codec: Any,
    *,
    label: str = "direct packed",
) -> None:
    """Fail fast if K/V codecs are not K8/V8 GS64.

    The canonical true-packed Metal kernel only supports this format in
    the current release.  Calling it with any other configuration silently
    decodes value vectors with the wrong bit width.
    """
    kb = getattr(key_codec, "bits", None)
    vb = getattr(value_codec, "bits", None)
    kgs = getattr(key_codec, "group_size", None)
    vgs = getattr(value_codec, "group_size", None)
    if kb != 8 or vb != 8 or kgs != 64 or vgs != 64:
        raise ValueError(
            f"{label} currently requires K8/V8 GS64; "
            f"got K{kb}/V{vb} GS{kgs}/{vgs}"
        )


def paged_view_from_blocks(
    key_blocks: list[PackedBlock],
    value_blocks: list[PackedBlock],
    *,
    max_pages: int | None = None,
    retain_source_blocks: bool = False,
) -> PagedKVView:
    """Build a temporary PagedKVView by writing blocks into a preallocated arena.

    Useful for tests and reference paths that still operate on block lists
    but need to exercise the paged kernel interface.

    Parameters
    ----------
    retain_source_blocks
        If ``True``, keep the original block objects alive inside the
        temporary arena.  Defaults to ``False`` so production-style callers
        do not double-store the payload.
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
        retain_source_blocks=retain_source_blocks,
    )

    for kb, vb in zip(key_blocks, value_blocks):
        arena.append(kb, vb)

    return arena.view()


class PagedKVArena:
    """GPU-resident packed KV arena with incremental slab growth.

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
    retain_source_blocks
        Debug option that keeps the original PackedBlock objects alive
        after copying their payload into the arena.  Must be ``False`` in
        benchmark/promotion mode.

    Notes
    -----
    The arena grows in fixed-size slabs rather than reserving the entire
    ``max_pages`` capacity at construction.  This avoids reserving the
    full maximum-context working set after the first page.  Slab growth
    copies existing pages into the new backing store; this is allocator
    overhead and is separate from per-attention-call history recopy.
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
        retain_source_blocks: bool = False,
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

        self._retain_source_blocks = retain_source_blocks

        # Slab allocator: start with one slab and grow by slab_size pages.
        # With 64 tokens/page, a 16-page slab holds 1,024 tokens.
        self._slab_size = min(16, max_pages)
        self._current_capacity = self._slab_size
        self._allocate_arrays(self._current_capacity)

        self._num_pages = 0
        self._next_physical_page = 0

        # Host-side metadata to avoid device-to-host reads on every query.
        self._active_tokens = 0
        self._page_counts_host: list[int] = []

        # Lightweight logical metadata for backward-compatible iterators.
        # Does NOT store packed_codes/scales; those live only in the arena.
        self._logical_metadata: list[dict[str, Any]] = []

        # Debug option: retain original blocks (doubles memory; off by default).
        self._source_blocks: list[tuple[PackedBlock, PackedBlock]] | None = (
            [] if retain_source_blocks else None
        )

        # Immutable format descriptor; frozen on first append.
        self._format: PagedKVFormat | None = None

        self.page_write_bytes = 0
        self.history_recopy_bytes = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def format(self) -> PagedKVFormat | None:
        return self._format

    @property
    def reserved_capacity_bytes(self) -> int:
        """Total bytes of the currently-allocated arena (active + inactive)."""
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
        # Use host metadata to avoid repeated device synchronisation.
        active_tokens = self._active_tokens
        # Each token contributes: codes + scales for K and V
        bytes_per_token_kv = (
            self.k_words_per_vector * 4  # uint32
            + self.k_groups_per_vector * 2  # float16
            + self.v_words_per_vector * 4  # uint32
            + self.v_groups_per_vector * 2  # float16
        )
        return active_tokens * self.n_kv_heads * bytes_per_token_kv

    @property
    def allocator_overhead_bytes(self) -> int:
        """Reserved capacity minus active payload."""
        return max(0, self.reserved_capacity_bytes - self.active_payload_bytes)

    @property
    def page_metadata_bytes(self) -> int:
        """Bytes for page table, starts, and counts."""
        return (
            int(self.page_table.size) * self.page_table.dtype.size
            + int(self.page_starts.size) * self.page_starts.dtype.size
            + int(self.page_counts.size) * self.page_counts.dtype.size
        )

    @property
    def source_block_bytes(self) -> int:
        """Bytes retained from original PackedBlock objects, if debugging."""
        if not self._retain_source_blocks or self._source_blocks is None:
            return 0
        total = 0
        for kb, vb in self._source_blocks:
            total += kb.payload_bytes() + vb.payload_bytes()
        return total

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _allocate_arrays(self, capacity: int) -> None:
        """Allocate backing arrays for the given page capacity."""
        self.k_codes = mx.zeros(
            (self.n_kv_heads, capacity, self.page_tokens, self.k_words_per_vector),
            dtype=mx.uint32,
        )
        self.k_scales = mx.zeros(
            (self.n_kv_heads, capacity, self.page_tokens, self.k_groups_per_vector),
            dtype=mx.float16,
        )
        self.v_codes = mx.zeros(
            (self.n_kv_heads, capacity, self.page_tokens, self.v_words_per_vector),
            dtype=mx.uint32,
        )
        self.v_scales = mx.zeros(
            (self.n_kv_heads, capacity, self.page_tokens, self.v_groups_per_vector),
            dtype=mx.float16,
        )
        self.page_table = mx.zeros((capacity,), dtype=mx.int32)
        self.page_starts = mx.zeros((capacity,), dtype=mx.int32)
        self.page_counts = mx.zeros((capacity,), dtype=mx.int32)

    def _grow_if_needed(self) -> None:
        """Grow the backing store by one slab when the current slab fills."""
        if self._next_physical_page < self._current_capacity:
            return
        if self._current_capacity >= self.max_pages:
            raise RuntimeError(
                f"KV arena cannot exceed max_pages={self.max_pages}"
            )

        new_capacity = min(self._current_capacity + self._slab_size, self.max_pages)

        # Allocate new arrays and copy existing payload.
        new_k_codes = mx.zeros(
            (self.n_kv_heads, new_capacity, self.page_tokens, self.k_words_per_vector),
            dtype=mx.uint32,
        )
        new_k_codes[:, : self._current_capacity, :, :] = self.k_codes
        new_k_scales = mx.zeros(
            (self.n_kv_heads, new_capacity, self.page_tokens, self.k_groups_per_vector),
            dtype=mx.float16,
        )
        new_k_scales[:, : self._current_capacity, :, :] = self.k_scales
        new_v_codes = mx.zeros(
            (self.n_kv_heads, new_capacity, self.page_tokens, self.v_words_per_vector),
            dtype=mx.uint32,
        )
        new_v_codes[:, : self._current_capacity, :, :] = self.v_codes
        new_v_scales = mx.zeros(
            (self.n_kv_heads, new_capacity, self.page_tokens, self.v_groups_per_vector),
            dtype=mx.float16,
        )
        new_v_scales[:, : self._current_capacity, :, :] = self.v_scales

        new_page_table = mx.zeros((new_capacity,), dtype=mx.int32)
        new_page_table[: self._current_capacity] = self.page_table
        new_page_starts = mx.zeros((new_capacity,), dtype=mx.int32)
        new_page_starts[: self._current_capacity] = self.page_starts
        new_page_counts = mx.zeros((new_capacity,), dtype=mx.int32)
        new_page_counts[: self._current_capacity] = self.page_counts

        # Publish the new arrays atomically.
        self.k_codes = new_k_codes
        self.k_scales = new_k_scales
        self.v_codes = new_v_codes
        self.v_scales = new_v_scales
        self.page_table = new_page_table
        self.page_starts = new_page_starts
        self.page_counts = new_page_counts
        self._current_capacity = new_capacity

        mx.eval(
            self.k_codes,
            self.k_scales,
            self.v_codes,
            self.v_scales,
            self.page_table,
            self.page_starts,
            self.page_counts,
        )

    def _freeze_format(
        self,
        key_block: PackedBlock,
        value_block: PackedBlock,
    ) -> None:
        """Freeze the immutable format descriptor from the first append."""
        key_codec = getattr(key_block, "codec", None)
        value_codec = getattr(value_block, "codec", None)
        self._format = PagedKVFormat(
            format_version=1,
            key_bits=getattr(key_block, "bits", 8),
            value_bits=getattr(value_block, "bits", 8),
            key_group_size=getattr(key_block, "group_size", 64),
            value_group_size=getattr(value_block, "group_size", 64),
            head_dim=getattr(key_block, "head_dim", 64),
            n_kv_heads=self.n_kv_heads,
            key_words_per_vector=self.k_words_per_vector,
            value_words_per_vector=self.v_words_per_vector,
            key_groups_per_vector=self.k_groups_per_vector,
            value_groups_per_vector=self.v_groups_per_vector,
            key_sign_seed=getattr(key_codec, "sign_seed", 42),
            value_sign_seed=getattr(value_codec, "sign_seed", 42),
            key_layer_id=getattr(key_block, "layer_id", 0),
            value_layer_id=getattr(value_block, "layer_id", 0),
            key_stream_id=getattr(key_block, "stream_id", "K"),
            value_stream_id=getattr(value_block, "stream_id", "V"),
            key_wht_applied=getattr(key_block, "wht_applied", True),
            value_wht_applied=getattr(value_block, "wht_applied", True),
            key_original_dtype=getattr(key_block, "original_dtype", "float16"),
            value_original_dtype=getattr(value_block, "original_dtype", "float16"),
            preconditioner=(
                "WHT64_HASH_SIGN_V1"
                if getattr(key_codec, "use_wht", True)
                else "NONE"
            ),
        )

    def _validate_format(self, key_block: PackedBlock, value_block: PackedBlock) -> None:
        """Ensure every append matches the frozen arena format."""
        if self._format is None:
            return
        fmt = self._format
        errors: list[str] = []
        if getattr(key_block, "bits", None) != fmt.key_bits:
            errors.append(
                f"key bits {getattr(key_block, 'bits', None)} != {fmt.key_bits}"
            )
        if getattr(value_block, "bits", None) != fmt.value_bits:
            errors.append(
                f"value bits {getattr(value_block, 'bits', None)} != {fmt.value_bits}"
            )
        if getattr(key_block, "group_size", None) != fmt.key_group_size:
            errors.append(
                f"key group_size {getattr(key_block, 'group_size', None)} != {fmt.key_group_size}"
            )
        if getattr(value_block, "group_size", None) != fmt.value_group_size:
            errors.append(
                f"value group_size {getattr(value_block, 'group_size', None)} != {fmt.value_group_size}"
            )
        if getattr(key_block, "head_dim", None) != fmt.head_dim:
            errors.append(
                f"head_dim {getattr(key_block, 'head_dim', None)} != {fmt.head_dim}"
            )
        if errors:
            raise ValueError(f"PagedKVFormat mismatch: {', '.join(errors)}")

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
        if self._format is None:
            self._freeze_format(key_block, value_block)
        self._validate_format(key_block, value_block)

        if self._num_pages >= self.max_pages:
            raise RuntimeError(
                f"KV arena full: {self._num_pages}/{self.max_pages} pages"
            )

        # P1-1: Grow the backing store by one slab if necessary.
        self._grow_if_needed()

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

        # Resolve only the written slices and metadata entries before
        # publishing the host page count.  This is cheaper than evaluating
        # the entire backing store on every 64-token append.
        mx.eval(
            self.k_codes[:, physical_page, :count, :],
            self.k_scales[:, physical_page, :count, :],
            self.v_codes[:, physical_page, :count, :],
            self.v_scales[:, physical_page, :count, :],
            self.page_table[logical_page],
            self.page_starts[logical_page],
            self.page_counts[logical_page],
        )

        # Host-side metadata update (avoids device-to-host reads later).
        self._active_tokens += count
        self._page_counts_host.append(count)

        # Lightweight logical metadata for iterators/views.
        self._logical_metadata.append(
            {
                "logical_start": int(key_block.logical_start),
                "token_count": count,
                "physical_page": physical_page,
            }
        )

        if self._source_blocks is not None:
            self._source_blocks.append((key_block, value_block))

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
        # MAX_PAGES in the Metal kernel is used as a stride for per-head
        # indexing.  It MUST match the actual backing array capacity, not the
        # logical maximum, or the kernel reads from the wrong offsets for
        # kv_head >= 1.  When the arena grows via slab allocation, a new
        # PagedKVView is created with the updated capacity, and the kernel
        # cache naturally recompiles with the new stride.
        return PagedKVView(
            k_codes=self.k_codes,
            k_scales=self.k_scales,
            v_codes=self.v_codes,
            v_scales=self.v_scales,
            page_table=self.page_table,
            page_starts=self.page_starts,
            page_counts=self.page_counts,
            num_pages=self._num_pages,
            max_pages=self._current_capacity,
            page_tokens=self.page_tokens,
            k_words_per_vector=self.k_words_per_vector,
            v_words_per_vector=self.v_words_per_vector,
            k_groups_per_vector=self.k_groups_per_vector,
            v_groups_per_vector=self.v_groups_per_vector,
            format=self._format,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _key_block_from_arena(self, meta: dict[str, Any]) -> PackedBlockV4:
        """Reconstruct a lightweight key PackedBlockV4 from an arena slice."""
        physical_page = meta["physical_page"]
        count = meta["token_count"]
        fmt = self._format
        bits = fmt.key_bits if fmt is not None else 8
        codes_per_word = 32 // bits
        head_dim = self.k_words_per_vector * codes_per_word
        group_size = fmt.key_group_size if fmt is not None else 64
        groups_per_vector = head_dim // group_size
        codes_view = self.k_codes[None, :, physical_page, :count, :]
        scales_view = self.k_scales[None, :, physical_page, :count, :]
        value_count = self.n_kv_heads * count * head_dim
        preconditioner = (
            Preconditioner.WHT64_HASH_SIGN_V1
            if (fmt is not None and fmt.preconditioner == "WHT64_HASH_SIGN_V1")
            else Preconditioner.NONE
        )
        return PackedBlockV4(
            packed_codes=codes_view,
            scales=scales_view,
            token_count=count,
            bits=bits,
            group_size=group_size,
            groups_per_vector=groups_per_vector,
            codes_per_word=codes_per_word,
            words_per_vector=self.k_words_per_vector,
            logical_start=meta["logical_start"],
            logical_end=meta["logical_start"] + count,
            head_dim=head_dim,
            original_value_count=value_count,
            padded_value_count=value_count,
            original_dtype=fmt.key_original_dtype if fmt is not None else "float16",
            batch_size=1,
            n_kv_heads=self.n_kv_heads,
            sign_seed=fmt.key_sign_seed if fmt is not None else 42,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=fmt.key_layer_id if fmt is not None else 0,
            stream_id=fmt.key_stream_id if fmt is not None else "K",
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=preconditioner,
            codec_signature="",
        )

    def _value_block_from_arena(self, meta: dict[str, Any]) -> PackedBlockV4:
        """Reconstruct a lightweight value PackedBlockV4 from an arena slice."""
        physical_page = meta["physical_page"]
        count = meta["token_count"]
        fmt = self._format
        bits = fmt.value_bits if fmt is not None else 8
        codes_per_word = 32 // bits
        head_dim = self.v_words_per_vector * codes_per_word
        group_size = fmt.value_group_size if fmt is not None else 64
        groups_per_vector = head_dim // group_size
        codes_view = self.v_codes[None, :, physical_page, :count, :]
        scales_view = self.v_scales[None, :, physical_page, :count, :]
        value_count = self.n_kv_heads * count * head_dim
        preconditioner = (
            Preconditioner.WHT64_HASH_SIGN_V1
            if (fmt is not None and fmt.preconditioner == "WHT64_HASH_SIGN_V1")
            else Preconditioner.NONE
        )
        return PackedBlockV4(
            packed_codes=codes_view,
            scales=scales_view,
            token_count=count,
            bits=bits,
            group_size=group_size,
            groups_per_vector=groups_per_vector,
            codes_per_word=codes_per_word,
            words_per_vector=self.v_words_per_vector,
            logical_start=meta["logical_start"],
            logical_end=meta["logical_start"] + count,
            head_dim=head_dim,
            original_value_count=value_count,
            padded_value_count=value_count,
            original_dtype=fmt.value_original_dtype if fmt is not None else "float16",
            batch_size=1,
            n_kv_heads=self.n_kv_heads,
            sign_seed=fmt.value_sign_seed if fmt is not None else 42,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=fmt.value_layer_id if fmt is not None else 0,
            stream_id=fmt.value_stream_id if fmt is not None else "V",
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=preconditioner,
            codec_signature="",
        )

    def iter_key_blocks(self):
        """Yield each sealed key block in logical order.

        Backward-compatible iterator for fallback paths and tests.  In
        production paged mode this reconstructs a temporary view from the
        arena slice; the original PackedBlock objects are NOT retained.
        """
        if self._source_blocks is not None:
            for kb, _vb in self._source_blocks:
                yield kb
            return

        # Reconstruct lightweight views from arena slices.  This keeps the
        # historical cache iterable without keeping duplicate payloads alive.
        for meta in self._logical_metadata:
            yield self._key_block_from_arena(meta)

    def iter_value_blocks(self):
        """Yield each sealed value block in logical order.

        See ``iter_key_blocks`` for the production-vs-debug distinction.
        """
        if self._source_blocks is not None:
            for _kb, vb in self._source_blocks:
                yield vb
            return

        for meta in self._logical_metadata:
            yield self._value_block_from_arena(meta)

    def reset(self) -> None:
        """Hide all pages without zeroing the arena.

        Old page contents are ignored because ``num_pages`` becomes zero.
        Zeroing hundreds of megabytes on every reset is pointless.
        """
        self._num_pages = 0
        self._next_physical_page = 0
        self._logical_metadata.clear()
        if self._source_blocks is not None:
            self._source_blocks.clear()
        self._active_tokens = 0
        self._page_counts_host.clear()
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
            "allocator_overhead_bytes": self.allocator_overhead_bytes,
            "page_metadata_bytes": self.page_metadata_bytes,
            "source_block_bytes": self.source_block_bytes,
            "page_write_bytes": self.page_write_bytes,
            "history_recopy_bytes": self.history_recopy_bytes,
        }
