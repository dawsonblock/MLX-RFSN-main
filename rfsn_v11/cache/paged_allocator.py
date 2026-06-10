"""Simplified paged KV allocator with block-level prefix caching.

64-token blocks, reference counting, LRU eviction.

Step 20: Block allocator
Step 21: Chain-hash prefix cache
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheBlock:
    """A single 64-token KV cache block."""
    block_id: int
    ref_count: int = 0
    block_hash: bytes = b""
    compression_signature: str = ""  # e.g. "k8v4_gs64_wht"
    last_used: float = field(default_factory=time.time)
    compressed_kv: Any = None  # actual compressed KV data


class PagedKVAllocator:
    """Simplified paged KV allocator.

    Parameters
    ----------
    block_size : int
        Tokens per block. Default 64.
    max_blocks : int
        Maximum number of blocks before LRU eviction triggers.
    """

    def __init__(self, block_size: int = 64, max_blocks: int = 10_000) -> None:
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._blocks: Dict[int, CacheBlock] = {}
        self._free_list: List[int] = []
        self._next_block_id: int = 0
        # prefix cache: hash → block_id
        self._hash_to_block: Dict[bytes, int] = {}
        self._lock = threading.Lock()
        self._evictions: int = 0
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Block allocation
    # ------------------------------------------------------------------

    def allocate_block(self, compression_signature: str = "") -> CacheBlock:
        """Allocate a new block from the free list or by growing the pool."""
        with self._lock:
            if self._free_list:
                bid = self._free_list.pop()
                block = self._blocks[bid]
                block.ref_count = 1
                block.compression_signature = compression_signature
                block.last_used = time.time()
                block.compressed_kv = None
                return block

            # Grow pool if under limit
            if len(self._blocks) < self.max_blocks:
                bid = self._next_block_id
                self._next_block_id += 1
                block = CacheBlock(
                    block_id=bid,
                    ref_count=1,
                    compression_signature=compression_signature,
                )
                self._blocks[bid] = block
                return block

            # Evict LRU block with ref_count == 0
            evicted = self._evict_lru()
            if evicted is None:
                raise MemoryError(
                    "PagedKVAllocator: no blocks available"
                )
            evicted.ref_count = 1
            evicted.compression_signature = compression_signature
            evicted.last_used = time.time()
            evicted.compressed_kv = None
            evicted.block_hash = b""
            return evicted

    def free_block(self, block_id: int) -> None:
        """Decrement ref_count; if zero, return to free list."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                return
            block.ref_count = max(block.ref_count - 1, 0)
            if block.ref_count == 0:
                self._free_list.append(block_id)
                # Remove from hash cache if present
                if (
                    block.block_hash
                    and block.block_hash in self._hash_to_block
                ):
                    del self._hash_to_block[block.block_hash]

    def _evict_lru(self) -> Optional[CacheBlock]:
        """Find and evict the least-recently-used block with ref_count == 0."""
        candidates = [b for b in self._blocks.values() if b.ref_count == 0]
        if not candidates:
            return None
        lru = min(candidates, key=lambda b: b.last_used)
        self._evictions += 1
        return lru

    # ------------------------------------------------------------------
    # Prefix cache (chain hash)
    # ------------------------------------------------------------------

    def compute_hash(
        self,
        parent_hash: bytes,
        token_ids: Tuple[int, ...],
        model_id: str,
        compression_signature: str,
    ) -> bytes:
        """Compute content-based chain hash for a block."""
        hasher = hashlib.sha256()
        hasher.update(parent_hash)
        hasher.update(bytes(token_ids))
        hasher.update(model_id.encode("utf-8"))
        hasher.update(compression_signature.encode("utf-8"))
        return hasher.digest()

    def lookup_prefix(self, block_hash: bytes) -> Optional[CacheBlock]:
        """Look up a block by its content hash."""
        with self._lock:
            bid = self._hash_to_block.get(block_hash)
            if bid is not None:
                block = self._blocks[bid]
                block.ref_count += 1
                block.last_used = time.time()
                self._hits += 1
                return block
            self._misses += 1
            return None

    def store_prefix(self, block: CacheBlock, block_hash: bytes) -> None:
        """Store a block in the prefix cache."""
        with self._lock:
            block.block_hash = block_hash
            self._hash_to_block[block_hash] = block.block_id
            block.last_used = time.time()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "prefix_cache_hit_rate": self._hits / max(total, 1),
                "blocks_reused": self._hits,
                "blocks_evicted": self._evictions,
                "total_blocks": len(self._blocks),
                "free_blocks": len(self._free_list),
                # placeholder — measured at call site
                "allocator_overhead_ms": 0.0,
            }
