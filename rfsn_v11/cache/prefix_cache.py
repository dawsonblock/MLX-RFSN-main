"""Prefix Cache Manager for rfsn_v11.

Ported from vmlx-main/vmlx_engine/prefix_cache.py.

This module provides BlockAwarePrefixCache: a block-based cache backed by
PagedCacheManager.  Model-family dispatch logic and vmlx-specific helper
classes (PrefixCacheManager trie, smelt/TQ loader flags etc.) are omitted;
only the block-level infrastructure consumed by rfsn_v11 is retained.

Reference model-cache extraction helpers that depend on vmlx-internal cache
class names (DeepseekV4Cache, ZayaNoStateCache, …) are preserved in the
reconstruction logic but guarded with try/except so they gracefully degrade
when the corresponding packages are absent.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .paged_cache import BlockTable, PagedCacheManager, compute_block_hash

logger = logging.getLogger(__name__)

# ---- Cache-type priority for LRU eviction (assistant evicted first) ----------
_CACHE_TYPE_PRIORITY: Tuple[str, ...] = ("assistant", "user", "system")


# =============================================================================
# Module-level helper predicates and utilities
# =============================================================================


def _is_dsv4_cache_class(class_name: str) -> bool:
    """Return True for DeepseekV4 cache variant class names."""
    return (class_name or "") in {"DeepseekV4Cache", "PoolQuantizedV4Cache"}


def _cache_data_has_dsv4(cache_data: Any) -> bool:
    """Return True when extracted cache states include DSV4 composite layers."""
    try:
        for layer_state in cache_data or []:
            if not isinstance(layer_state, dict):
                continue
            if _is_dsv4_cache_class(layer_state.get("class_name", "")):
                return True
    except Exception:
        return False
    return False


def _to_numpy_tree(obj: Any) -> Any:
    """Convert nested MLX-array state to numpy, preserving tuple/list shape."""
    if obj is None:
        return None
    if hasattr(obj, "__array__"):
        import numpy as np
        return np.array(obj)
    if isinstance(obj, tuple):
        return tuple(_to_numpy_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_to_numpy_tree(x) for x in obj]
    return obj


def _copy_mlx_tree(obj: Any) -> Any:
    """Materialise independent MLX-array copies in a nested cache state tree."""
    if obj is None or not HAS_MLX:
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        copied = obj * 1
        mx.eval(copied)
        return copied
    if isinstance(obj, tuple):
        return tuple(_copy_mlx_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_copy_mlx_tree(x) for x in obj]
    return obj


def _is_zaya_cca_cache_list_state(layer_state: Dict[str, Any]) -> bool:
    """Return True for ZAYA's CacheList(KVCache, ArraysCache) CCA layer."""
    if layer_state.get("class_name") != "CacheList":
        return False
    subs = layer_state.get("sub_caches")
    if not isinstance(subs, (list, tuple)) or len(subs) < 2:
        return False
    first = subs[0] if isinstance(subs[0], dict) else {}
    second = subs[1] if isinstance(subs[1], dict) else {}
    first_cls = str(first.get("class_name", ""))
    second_cls = str(second.get("class_name", ""))
    if first_cls not in {"KVCache", "QuantizedKVCache"}:
        return False
    if second_cls != "ArraysCache":
        return False
    second_state = second.get("state")
    return isinstance(second_state, (list, tuple)) and len(second_state) >= 2


def _cache_data_has_zaya_cca(cache_data: Any) -> bool:
    """Return True when extracted cache states include typed ZAYA CCA layers."""
    try:
        for layer_state in cache_data or []:
            if isinstance(layer_state, dict) and _is_zaya_cca_cache_list_state(
                layer_state
            ):
                return True
    except Exception:
        return False
    return False


def _cache_data_has_rotating_kv(cache_data: Any) -> bool:
    """Return True when extracted cache states include rotating-window KV."""
    try:
        for layer_state in cache_data or []:
            if not isinstance(layer_state, dict):
                continue
            if "Rotating" in str(layer_state.get("class_name", "")):
                return True
    except Exception:
        return False
    return False


def _dsv4_cache_meta(layer_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract DSV4-specific metadata from a layer state dict."""
    meta: Dict[str, Any] = {}
    cr = layer_state.get("compress_ratio")
    sw = layer_state.get("sliding_window")
    if cr is not None:
        try:
            meta["compress_ratio"] = int(cr)
        except Exception:
            meta["compress_ratio"] = cr
    if sw is not None:
        try:
            meta["sliding_window"] = int(sw)
        except Exception:
            meta["sliding_window"] = sw
    if "pool_quant" in layer_state:
        meta["pool_quant"] = layer_state.get("pool_quant")
    if "local_quant_meta" in layer_state:
        meta["local_quant_meta"] = layer_state.get("local_quant_meta")
    return meta


def _block_needs_cumulative_update(cache_data: Any) -> bool:
    """Check if a block's cache_data is missing cumulative SSM state.

    Returns True if the block has "skip" entries (SSM layers stored as
    non-last) but no "cumulative" entries.
    """
    if not cache_data:
        return False
    has_skip = False
    has_cumulative = False
    for entry in cache_data:
        if not isinstance(entry, (tuple, list)) or len(entry) == 0:
            continue
        tag = entry[0]
        if tag == "skip":
            has_skip = True
        elif tag == "deepseek_v4_pending":
            has_skip = True
        elif tag in ("cumulative", "deepseek_v4"):
            has_cumulative = True
        elif tag == "zaya_cca":
            if len(entry) > 2 and entry[2] is not None:
                has_cumulative = True
            else:
                has_skip = True
        elif tag == "cache_list" and len(entry) > 1:
            for sub in entry[1]:
                if isinstance(sub, (tuple, list)) and len(sub) > 0:
                    if sub[0] == "skip":
                        has_skip = True
                    elif sub[0] == "deepseek_v4_pending":
                        has_skip = True
                    elif sub[0] in ("cumulative", "deepseek_v4"):
                        has_cumulative = True
    return has_skip and not has_cumulative


# =============================================================================
# BlockCacheEntry
# =============================================================================


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: Optional[List[Any]]  # Legacy full-cache reference, if needed
    last_access: float
    cache_type: str = "assistant"  # "system" | "user" | "assistant"


# =============================================================================
# BlockAwarePrefixCache
# =============================================================================


class BlockAwarePrefixCache:
    """Prefix cache backed by PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (configurable tokens per block).
    - Copy-on-Write for efficient forking.
    - Hash-based deduplication across requests.
    - Reference counting for memory efficiency.
    - Priority-bucketed release (assistant < user < system).

    Usage::

        paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager)

        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)
        cache.store_cache(request_id, tokens, kv_cache_data)
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
        model_path: Optional[str] = None,
        # loader fingerprint params — kept for API parity but simplified
        smelt_enabled: bool = False,
        smelt_pct: Optional[float] = None,
        tq_enabled: bool = False,
        kv_quant_bits: int = 0,
    ) -> None:
        """Initialise block-aware prefix cache.

        Args:
            model: The loaded model object.  Used only for deriving a stable
                model key (architecture + path + mtime).  May be None.
            paged_cache_manager: The PagedCacheManager instance for block
                management.
            model_path: Optional filesystem path to the model directory.
            smelt_enabled: Whether SMELT quantisation was applied.
            smelt_pct: SMELT quantisation percentage (if enabled).
            tq_enabled: Whether TurboQuant KV compression is active.
            kv_quant_bits: Integer KV quantisation bit-width.
        """
        self.model = model
        self.model_key = self._compute_model_key(
            model,
            model_path=model_path,
            smelt_enabled=smelt_enabled,
            smelt_pct=smelt_pct,
            tq_enabled=tq_enabled,
            kv_quant_bits=kv_quant_bits,
        )
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size

        # Hash table for quick prefix lookup
        self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Per-cache-type LRU buckets
        self._entries_by_type: Dict[str, OrderedDict] = {
            t: OrderedDict() for t in _CACHE_TYPE_PRIORITY
        }

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

        # Lazy-cached expected KV head count for validation
        self._n_kv_heads: Optional[int] = None
        self._allowed_n_kv_heads: Optional[set] = None

        # Expected layer count derived from model
        self._expected_num_layers: Optional[int] = None
        if model is not None and hasattr(model, "make_cache"):
            try:
                _cache = model.make_cache() or []
                if len(_cache) > 0:
                    self._expected_num_layers = len(_cache)
            except Exception:
                self._expected_num_layers = None
        if self._expected_num_layers is None:
            for _attr in ("args", "config"):
                _cfg = getattr(model, _attr, None) if model is not None else None
                if _cfg is not None:
                    _ln = getattr(_cfg, "num_hidden_layers", 0)
                    if _ln:
                        self._expected_num_layers = int(_ln)
                        break

    # =========================================================================
    # Model key computation (simplified — no vmlx schema version embedding)
    # =========================================================================

    @staticmethod
    def _compute_model_key(
        model: Any,
        model_path: Optional[str] = None,
        smelt_enabled: bool = False,
        smelt_pct: Optional[float] = None,
        tq_enabled: bool = False,
        kv_quant_bits: int = 0,
    ) -> str:
        """Compute a stable content-derived cache key for a loaded model.

        Returns the first 32 hex chars of SHA-256 over architecture identity,
        path+mtime, and loader flags.  Falls back to ``id(model)`` on error.
        """
        parts: List[str] = []

        try:
            if model is not None:
                parts.append(type(model).__module__ + "." + type(model).__name__)
                for attr in ("args", "config"):
                    cfg = getattr(model, attr, None)
                    if cfg is None:
                        continue
                    for f in (
                        "model_type",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "hidden_size",
                        "vocab_size",
                        "kv_lora_rank",
                    ):
                        v = getattr(cfg, f, None)
                        if v is not None and not callable(v):
                            parts.append(f"{f}={v}")
                    break
        except Exception:
            pass

        if model_path:
            parts.append(f"path={model_path}")
            try:
                for fname in ("config.json", "jang_config.json"):
                    p = os.path.join(model_path, fname)
                    if os.path.exists(p):
                        parts.append(f"{fname}_mtime={int(os.path.getmtime(p))}")
            except Exception:
                pass

        parts.append(f"smelt={1 if smelt_enabled else 0}")
        if smelt_enabled and smelt_pct is not None:
            parts.append(f"smelt_pct={float(smelt_pct):.4f}")
        parts.append(f"tq={1 if tq_enabled else 0}")
        parts.append(f"kvq={int(kv_quant_bits or 0)}")

        if not parts:
            return f"id_{id(model):x}"

        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
        return digest

    # =========================================================================
    # KV head count helpers (for cache validation)
    # =========================================================================

    def _get_n_kv_heads(self) -> int:
        """Return the expected KV head count from model config (cached).

        Returns 0 if unknown (validation skipped).  For MLA models
        (kv_lora_rank > 0) returns 1 (compressed latent single head).
        """
        if self._n_kv_heads is not None:
            return self._n_kv_heads
        n_kv = 0
        try:
            candidates = [self.model]
            if self.model is not None:
                lm = getattr(self.model, "language_model", None)
                if lm is not None:
                    candidates.append(lm)
                mm = getattr(self.model, "model", None)
                if mm is not None and mm is not self.model:
                    candidates.append(mm)
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    model_type = str(getattr(cfg, "model_type", "") or "").lower()
                    kv_lora_rank = getattr(cfg, "kv_lora_rank", 0)
                    if kv_lora_rank and kv_lora_rank > 0:
                        if model_type in ("bailing_hybrid", "bailing_moe_v2_5"):
                            n_kv = int(getattr(cfg, "num_attention_heads", 0) or 0)
                        else:
                            n_kv = 1
                        break
                if n_kv:
                    break
            if not n_kv:
                for model_obj in candidates:
                    for attr in ("args", "config", "text_config"):
                        cfg = getattr(model_obj, attr, None)
                        if cfg is None:
                            continue
                        n_kv = int(
                            getattr(cfg, "num_key_value_heads", 0)
                            or getattr(cfg, "num_kv_heads", 0)
                            or 0
                        )
                        if n_kv:
                            break
                    if n_kv:
                        break
        except Exception:
            pass
        self._n_kv_heads = n_kv
        return n_kv

    def _get_allowed_n_kv_heads(self) -> set:
        """Return the set of valid KV head counts (for mixed-head models)."""
        if self._allowed_n_kv_heads is not None:
            return self._allowed_n_kv_heads
        primary = self._get_n_kv_heads()
        allowed: set = {primary} if primary else set()
        try:
            candidates = [self.model]
            if self.model is not None:
                lm = getattr(self.model, "language_model", None)
                if lm is not None:
                    candidates.append(lm)
                mm = getattr(self.model, "model", None)
                if mm is not None and mm is not self.model:
                    candidates.append(mm)
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    for field in (
                        "num_global_key_value_heads",
                        "swa_num_key_value_heads",
                        "num_key_value_heads",
                    ):
                        v = getattr(cfg, field, None)
                        if v:
                            allowed.add(int(v))
        except Exception:
            pass
        self._allowed_n_kv_heads = allowed
        return allowed

    @staticmethod
    def _is_positional_cache(state: Any, class_name: str) -> bool:
        """Return True for positional KVCache (keys, values pair)."""
        if state is None:
            return False
        if isinstance(state, (list, tuple)) and len(state) == 2:
            k, v = state
            if isinstance(k, (list, tuple)) and isinstance(v, (list, tuple)):
                return True
            if hasattr(k, "shape") and hasattr(v, "shape"):
                return True
        return False

    # =========================================================================
    # Prefix index helpers
    # =========================================================================

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Register a token/block-ids pair in the prefix index."""
        key = hashlib.sha256(
            bytes(str(tuple(tokens)), "utf-8")
        ).hexdigest()[:32]
        self._prefix_index[key] = (tokens, block_ids)

    def _find_best_prefix_match(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Search the prefix index for the longest matching prefix.

        Returns (matched_tokens, block_ids) or None.
        """
        best: Optional[Tuple[List[int], List[int]]] = None
        for _key, (idx_tokens, idx_blocks) in list(self._prefix_index.items()):
            n = len(idx_tokens)
            if n > len(tokens):
                continue
            if tokens[:n] == idx_tokens:
                if best is None or n > len(best[0]):
                    best = (idx_tokens, idx_blocks)
        return best

    # =========================================================================
    # Public API — fetch / store / release
    # =========================================================================

    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier.
            tokens: Input token sequence.
            cache_extra_keys: Optional extra keys mixed into the block hash.

        Returns:
            Tuple of (block_table, remaining_tokens).
            - block_table: BlockTable if prefix found, None on miss.
            - remaining_tokens: Tokens that still need processing.
        """
        if not tokens:
            return None, tokens

        # Primary path: chain-hash lookup.
        cached_blocks, num_cached = self.paged_cache.get_computed_blocks(
            tokens,
            extra_keys=cache_extra_keys,
        )

        # Also try N-1 lookup for caches stored with truncated final token.
        if len(tokens) > 1:
            alt_blocks, alt_num_cached = self.paged_cache.get_computed_blocks(
                tokens[:-1],
                extra_keys=cache_extra_keys,
            )
            if alt_num_cached > num_cached:
                if cached_blocks:
                    try:
                        self.paged_cache.release_request_refs(
                            BlockTable(
                                request_id=request_id,
                                block_ids=[b.block_id for b in cached_blocks],
                                num_tokens=num_cached,
                            )
                        )
                    except Exception:
                        pass
                cached_blocks, num_cached = alt_blocks, alt_num_cached
            elif alt_blocks:
                try:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[b.block_id for b in alt_blocks],
                            num_tokens=alt_num_cached,
                        )
                    )
                except Exception:
                    pass

        # Check prefix index for longer exact-partial matches.
        best_match = self._find_best_prefix_match(
            tokens,
            cache_extra_keys=cache_extra_keys,
        )
        if cached_blocks and best_match and len(best_match[0]) > num_cached:
            try:
                self.paged_cache.release_request_refs(
                    BlockTable(
                        request_id=request_id,
                        block_ids=[b.block_id for b in cached_blocks],
                        num_tokens=num_cached,
                    )
                )
            except Exception:
                pass
            cached_blocks = []
            num_cached = 0

        if cached_blocks:
            _disk_store = getattr(self.paged_cache, "_disk_store", None)
            if self._dsv4_l2_chain_missing_terminal_state(cached_blocks, _disk_store):
                _reject_table = BlockTable(
                    request_id=request_id,
                    block_ids=[cb.block_id for cb in cached_blocks],
                    num_tokens=sum(
                        getattr(cb, "token_count", 0) for cb in cached_blocks
                    ),
                )
                self.paged_cache.release_request_refs(_reject_table)
                logger.warning(
                    "Ignoring DSV4 paged prefix hit for %s: no terminal "
                    "deepseek_v4 composite state in restored blocks.",
                    request_id,
                )
                self._misses += 1
                return None, tokens

            if self._zaya_l2_chain_missing_terminal_state(cached_blocks, _disk_store):
                _reject_table = BlockTable(
                    request_id=request_id,
                    block_ids=[cb.block_id for cb in cached_blocks],
                    num_tokens=sum(
                        getattr(cb, "token_count", 0) for cb in cached_blocks
                    ),
                )
                self.paged_cache.release_request_refs(_reject_table)
                logger.warning(
                    "Ignoring ZAYA paged prefix hit for %s: no terminal CCA "
                    "state in restored blocks.",
                    request_id,
                )
                self._misses += 1
                return None, tokens

            block_table = self.paged_cache.create_block_table(request_id)
            for cb in cached_blocks:
                block_table.block_ids.append(cb.block_id)
                block_table.num_tokens += cb.token_count

            remaining = tokens[block_table.num_tokens:]
            self._hits += 1
            self._tokens_saved += block_table.num_tokens
            logger.info(
                "Paged cache hit for %s: %d blocks, %d tokens",
                request_id,
                len(cached_blocks),
                block_table.num_tokens,
            )
            return block_table, remaining

        # Fallback: prefix index.
        if best_match:
            matched_tokens, matched_block_ids = best_match
            matched_blocks = [
                self.paged_cache.allocated_blocks.get(bid)
                for bid in matched_block_ids
            ]
            matched_blocks = [b for b in matched_blocks if b is not None]
            _disk_store = getattr(self.paged_cache, "_disk_store", None)
            if self._dsv4_l2_chain_missing_terminal_state(matched_blocks, _disk_store):
                logger.warning(
                    "Ignoring DSV4 prefix-index hit for %s.", request_id
                )
                self._misses += 1
                return None, tokens
            if self._zaya_l2_chain_missing_terminal_state(matched_blocks, _disk_store):
                logger.warning(
                    "Ignoring ZAYA prefix-index hit for %s.", request_id
                )
                self._misses += 1
                return None, tokens

            block_table = self.paged_cache.create_block_table(request_id)
            for block_id in matched_block_ids:
                if self.paged_cache.increment_ref(block_id):
                    block_table.block_ids.append(block_id)
                    b = self.paged_cache.allocated_blocks.get(block_id)
                    if b:
                        block_table.num_tokens += b.token_count

            if block_table.block_ids:
                remaining = tokens[len(matched_tokens):]
                self._hits += 1
                self._tokens_saved += len(matched_tokens)
                logger.info(
                    "Prefix index hit for %s: %d tokens",
                    request_id,
                    len(matched_tokens),
                )
                return block_table, remaining

        self._misses += 1
        return None, tokens

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
        cache_type: str = "assistant",
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[BlockTable]:
        """Store computed cache for future reuse.

        Args:
            request_id: Unique request identifier.
            tokens: Token sequence that was processed.
            cache_data: The computed KV cache to store.  Either a list of
                KVCache objects or a list of dicts with a ``"state"`` key
                holding ``(keys, values)`` tensors.
            cache_type: One of ``"system"`` | ``"user"`` | ``"assistant"``.
            cache_extra_keys: Optional extra keys mixed into the block hash.

        Returns:
            BlockTable for the stored cache, or None on failure.
        """
        if not tokens:
            return None

        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )
        has_dsv4_cache_data = _cache_data_has_dsv4(cache_data) if is_tensor_data else False
        has_zaya_cca_cache_data = (
            _cache_data_has_zaya_cca(cache_data) if is_tensor_data else False
        )
        has_rotating_kv_cache_data = (
            _cache_data_has_rotating_kv(cache_data) if is_tensor_data else False
        )

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            return block_table

        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size

        # For disk write-through, compute chain hashes over the full sequence.
        from .paged_cache import compute_block_hash as _compute_chain_hash
        parent_hash = None
        if existing_tokens > 0:
            num_existing_full = existing_tokens // self.block_size
            for eb_idx in range(num_existing_full):
                eb_start = eb_idx * self.block_size
                eb_end = eb_start + self.block_size
                parent_hash = _compute_chain_hash(
                    parent_hash,
                    tokens[eb_start:eb_end],
                    extra_keys=cache_extra_keys,
                )

        disk_store = self.paged_cache._disk_store

        # Frugal mode: skip in-RAM tensor mirror when disk store is present.
        _frugal_env = os.environ.get("RFSN_PAGED_FRUGAL", "").strip()
        if _frugal_env == "":
            _paged_frugal = disk_store is not None
        else:
            _paged_frugal = _frugal_env not in ("0", "false", "False", "no")

        logger.info(
            "Block disk write-through: disk_store=%s, is_tensor_data=%s, "
            "new_tokens=%d, num_new_blocks=%d, frugal=%s",
            "present" if disk_store else "None",
            is_tensor_data,
            len(new_tokens),
            num_new_blocks,
            _paged_frugal,
        )

        # Pre-convert source KV arrays to numpy for safe per-block slicing.
        np_sources: dict = {}
        if is_tensor_data and HAS_MLX:
            import numpy as np
            mx.synchronize()
            for idx, layer_state in enumerate(cache_data):
                cls = layer_state.get("class_name", "")
                if _is_zaya_cca_cache_list_state(layer_state):
                    # ZAYA CCA path omitted for brevity — stub as skip
                    continue
                state = layer_state.get("state")
                if state is None:
                    continue
                if self._is_positional_cache(state, cls):
                    try:
                        keys, values = state
                        if isinstance(keys, (tuple, list)):
                            # Quantized KV: dequantize first
                            if len(keys) < 2 or len(values) < 2:
                                continue
                            meta = layer_state.get("meta_state", ())
                            g_size, q_bits = 64, 8
                            if isinstance(meta, (tuple, list)) and len(meta) >= 2:
                                try:
                                    g_size = int(meta[-2])
                                    q_bits = int(meta[-1])
                                except (ValueError, TypeError):
                                    pass
                            k_w, k_s = keys[0], keys[1]
                            k_b = keys[2] if len(keys) >= 3 else mx.zeros_like(k_s)
                            v_w, v_s = values[0], values[1]
                            v_b = values[2] if len(values) >= 3 else mx.zeros_like(v_s)
                            k_dq = mx.dequantize(k_w, k_s, k_b, group_size=g_size, bits=q_bits)
                            v_dq = mx.dequantize(v_w, v_s, v_b, group_size=g_size, bits=q_bits)
                            orig_dtype = k_dq.dtype
                            if "bfloat16" in str(k_dq.dtype):
                                k_dq = k_dq.astype(mx.float32)
                                v_dq = v_dq.astype(mx.float32)
                                orig_dtype = mx.bfloat16
                            mx.eval(k_dq, v_dq)
                            np_sources[idx] = (
                                np.array(k_dq),
                                np.array(v_dq),
                                orig_dtype,
                            )
                        elif hasattr(keys, "shape"):
                            orig_dtype = keys.dtype
                            k_np = keys
                            v_np = values
                            if "bfloat16" in str(keys.dtype):
                                k_np = keys.astype(mx.float32)
                                v_np = values.astype(mx.float32)
                                orig_dtype = mx.bfloat16
                            mx.eval(k_np, v_np)
                            np_sources[idx] = (
                                np.array(k_np),
                                np.array(v_np),
                                orig_dtype,
                            )
                    except Exception as _e:
                        logger.debug("np_sources skip layer %d: %s", idx, _e)

        # Allocate and populate blocks
        for block_num in range(num_new_blocks):
            block = self.paged_cache.allocate_block()
            if block is None:
                logger.warning(
                    "store_cache: out of blocks after %d/%d for %s",
                    block_num,
                    num_new_blocks,
                    request_id,
                )
                break

            token_start = existing_tokens + block_num * self.block_size
            token_end = min(token_start + self.block_size, len(tokens))
            block_tokens = tokens[token_start:token_end]
            is_last_block = (block_num == num_new_blocks - 1)

            # Populate block cache_data from tensor states
            if is_tensor_data and not _paged_frugal:
                block_data = self._extract_block_tensor_slice(
                    cache_data,
                    token_start - existing_tokens,
                    token_end - existing_tokens,
                    is_last_block,
                    np_sources=np_sources,
                    existing_tokens=existing_tokens,
                )
                block.cache_data = block_data
            elif is_tensor_data:
                # frugal mode: data only written to disk (below)
                block.cache_data = None
            else:
                # Legacy KVCache object list
                block.cache_data = cache_data if is_last_block else None

            block.token_count = len(block_tokens)

            # Compute chain hash and register in prefix cache
            block_hash = _compute_chain_hash(
                parent_hash,
                block_tokens,
                extra_keys=cache_extra_keys,
            )
            block.block_hash = block_hash
            self.paged_cache.cached_block_hash_to_block.insert(block_hash, block)

            legacy_hash = self.paged_cache.compute_block_hash(block_tokens)
            block.hash_value = legacy_hash
            self.paged_cache.hash_to_block[legacy_hash] = block.block_id
            if len(block_tokens) < self.block_size:
                self.paged_cache._partial_block_sizes.add(len(block_tokens))

            # Async disk write-through
            if disk_store is not None and is_tensor_data:
                # Build numpy slices for disk store
                disk_data = self._extract_block_tensor_slice(
                    cache_data,
                    token_start - existing_tokens,
                    token_end - existing_tokens,
                    is_last_block,
                    np_sources=np_sources,
                    existing_tokens=existing_tokens,
                )
                if disk_data is not None:
                    try:
                        disk_store.write_block_async(
                            block_hash, disk_data, block.token_count
                        )
                    except Exception as _de:
                        logger.debug("Disk write-through failed for block %d: %s", block.block_id, _de)

            parent_hash = block_hash
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

        # Update prefix index for later exact-partial lookups
        if block_table.block_ids:
            stored_tokens = tokens[: block_table.num_tokens]
            self._update_prefix_index(stored_tokens, list(block_table.block_ids))

        # Register in request_tables
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data if not is_tensor_data else None,
            last_access=time.time(),
            cache_type=cache_type if cache_type in _CACHE_TYPE_PRIORITY else "assistant",
        )

        # Track in priority bucket
        bucket = self._entries_by_type.get(cache_type, self._entries_by_type["assistant"])
        bucket[request_id] = True

        logger.info(
            "Stored cache for %s: %d blocks, %d tokens",
            request_id,
            len(block_table.block_ids),
            block_table.num_tokens,
        )

        return block_table

    def release_cache(self, request_id: str) -> None:
        """Release all block references for a request.

        Args:
            request_id: Request whose cache should be released.
        """
        entry = self._request_tables.pop(request_id, None)
        if entry is not None:
            self.paged_cache.release_request_refs(entry.block_table)
            # Remove from priority buckets
            for bucket in self._entries_by_type.values():
                bucket.pop(request_id, None)

        self.paged_cache.delete_block_table(request_id)

    # =========================================================================
    # Tensor extraction helpers
    # =========================================================================

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Any],
        start_idx: int,
        end_idx: int,
        is_last_block: bool,
        np_sources: Optional[dict] = None,
        existing_tokens: int = 0,
    ) -> Optional[List[Any]]:
        """Extract per-block cache slices from full KV state.

        Args:
            cache_data: Full layer-state dicts.
            start_idx: Start token index within new tokens.
            end_idx: End token index within new tokens.
            is_last_block: Whether this is the last block in the sequence.
            np_sources: Optional pre-converted numpy arrays keyed by layer idx.
            existing_tokens: Number of tokens already cached (for offset calc).

        Returns:
            List of tagged tuples per layer, or None if unavailable.
        """
        if not HAS_MLX or not cache_data:
            return None

        if np_sources:
            return self._numpy_block_slice(
                cache_data, np_sources, start_idx, end_idx, is_last_block, existing_tokens
            )

        block_slices = []
        for layer_idx, layer_state in enumerate(cache_data):
            if "state" not in layer_state:
                continue

            class_name = layer_state.get("class_name", "")

            if _is_dsv4_cache_class(class_name):
                state = layer_state["state"]
                if is_last_block and state is not None:
                    meta = layer_state.get("meta_state", "")
                    block_slices.append((
                        "deepseek_v4",
                        state,
                        meta,
                        class_name,
                        _dsv4_cache_meta(layer_state),
                    ))
                else:
                    block_slices.append((
                        "deepseek_v4_pending",
                        class_name,
                        _dsv4_cache_meta(layer_state),
                    ))
                continue

            if class_name == "ZayaNoStateCache" or layer_state.get("no_state"):
                block_slices.append(("no_state", class_name or "ZayaNoStateCache"))
                continue

            if _is_zaya_cca_cache_list_state(layer_state):
                # ZAYA CCA: handle KV + CCA state
                subs = layer_state["sub_caches"]
                kv_sub = subs[0]
                state_sub = subs[1]
                kv_entry: Any = ("skip",)
                try:
                    keys, values = kv_sub.get("state")
                    if hasattr(keys, "shape"):
                        ndim = len(keys.shape)
                        seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                        if seq_dim >= 0:
                            seq_len = keys.shape[seq_dim]
                            actual_end = min(end_idx, seq_len)
                            if start_idx < actual_end:
                                if ndim == 4:
                                    ks = keys[:, :, start_idx:actual_end, :]
                                    vs = values[:, :, start_idx:actual_end, :]
                                else:
                                    ks = keys[:, start_idx:actual_end, :]
                                    vs = values[:, start_idx:actual_end, :]
                                kv_entry = ("kv", ks, vs)
                except Exception:
                    kv_entry = ("skip",)

                cca_state = (
                    _copy_mlx_tree(state_sub.get("state")) if is_last_block else None
                )
                block_slices.append((
                    "zaya_cca",
                    kv_entry,
                    cca_state,
                    state_sub.get("meta_state", ""),
                    {
                        "schema": "zaya_cca_v1",
                        "cca_class_name": state_sub.get("class_name", "ArraysCache"),
                    },
                ))
                continue

            # CacheList (MoE models)
            if class_name == "CacheList" and "sub_caches" in layer_state:
                sub_slices = []
                for sub in layer_state["sub_caches"]:
                    sub_state = sub.get("state")
                    sub_cls = sub.get("class_name", "")
                    if sub_state is None:
                        sub_slices.append(("skip",))
                        continue
                    if self._is_positional_cache(sub_state, sub_cls):
                        try:
                            keys, values = sub_state
                            if isinstance(keys, (tuple, list)):
                                first_k = keys[0]
                                seq_len = first_k.shape[-2]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                keys_slice = tuple(
                                    t[..., start_idx:actual_end, :] for t in keys
                                )
                                values_slice = tuple(
                                    t[..., start_idx:actual_end, :] for t in values
                                )
                                meta = sub.get("meta_state", ())
                                sub_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                            elif hasattr(keys, "shape"):
                                ndim = len(keys.shape)
                                seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                                if seq_dim < 0:
                                    sub_slices.append(("skip",))
                                    continue
                                seq_len = keys.shape[seq_dim]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                if ndim == 4:
                                    ks = keys[:, :, start_idx:actual_end, :]
                                    vs = values[:, :, start_idx:actual_end, :]
                                else:
                                    ks = keys[:, start_idx:actual_end, :]
                                    vs = values[:, start_idx:actual_end, :]
                                sub_slices.append(("kv", ks, vs))
                        except Exception:
                            sub_slices.append(("skip",))
                    else:
                        if is_last_block:
                            meta = sub.get("meta_state", "")
                            sub_slices.append((
                                "cumulative",
                                _copy_mlx_tree(sub_state),
                                meta,
                                sub_cls,
                            ))
                        else:
                            sub_slices.append(("skip",))
                block_slices.append(("cache_list", sub_slices))
                continue

            state = layer_state["state"]

            if self._is_positional_cache(state, class_name):
                try:
                    keys, values = state

                    if isinstance(keys, (tuple, list)):
                        first_k = keys[0]
                        seq_len = first_k.shape[-2]
                        actual_end = min(end_idx, seq_len)
                        if start_idx >= actual_end:
                            block_slices.append(("skip",))
                            continue
                        keys_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in keys
                        )
                        values_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in values
                        )
                        meta = layer_state.get("meta_state", ())
                        block_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                        continue

                    if not hasattr(keys, "shape"):
                        block_slices.append(("skip",))
                        continue

                    ndim = len(keys.shape)
                    seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                    if seq_dim < 0:
                        block_slices.append(("skip",))
                        continue

                    seq_len = keys.shape[seq_dim]
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        block_slices.append(("skip",))
                        continue

                    if ndim == 4:
                        ks = keys[:, :, start_idx:actual_end, :]
                        vs = values[:, :, start_idx:actual_end, :]
                    else:
                        ks = keys[:, start_idx:actual_end, :]
                        vs = values[:, start_idx:actual_end, :]

                    # Detect RotatingKVCache by class name
                    if "Rotating" in class_name:
                        meta = layer_state.get("meta_state", ())
                        max_size = seq_len
                        keep = 0
                        offset = None
                        idx_state = None
                        if meta and len(meta) >= 2:
                            try:
                                keep = int(meta[0])
                                max_size = int(meta[1])
                                if len(meta) >= 3:
                                    offset = int(meta[2])
                                if len(meta) >= 4:
                                    idx_state = int(meta[3])
                            except (ValueError, TypeError):
                                pass
                        block_slices.append((
                            "rotating_kv",
                            ks,
                            vs,
                            max_size,
                            keep,
                            offset,
                            idx_state,
                        ))
                    else:
                        block_slices.append(("kv", ks, vs))
                except Exception:
                    block_slices.append(("skip",))
            else:
                # Cumulative / SSM layer
                state = layer_state.get("state")
                if is_last_block and state is not None:
                    meta = layer_state.get("meta_state", "")
                    if isinstance(state, (list, tuple)):
                        block_slices.append((
                            "cumulative",
                            _copy_mlx_tree(state),
                            meta,
                            class_name,
                        ))
                    else:
                        block_slices.append(("skip",))
                else:
                    block_slices.append(("skip",))

        return block_slices if block_slices else None

    def _numpy_block_slice(
        self,
        cache_data: List[Any],
        np_sources: dict,
        start_idx: int,
        end_idx: int,
        is_last_block: bool,
        existing_tokens: int = 0,
    ) -> Optional[List[Any]]:
        """Create per-block cache_data using numpy slicing (no MLX/Metal ops).

        Args:
            cache_data: Full layer-state dicts from _extract_cache_states.
            np_sources: dict mapping layer_idx -> (np_keys, np_values, orig_dtype).
            start_idx, end_idx: Token range for this block.
            is_last_block: Whether this is the last block in the sequence.
            existing_tokens: Number of previously cached tokens.

        Returns:
            List of tagged tuples per layer, same format as
            _extract_block_tensor_slice but with numpy arrays.
        """
        import numpy as np

        block_slices = []
        for idx, layer_state in enumerate(cache_data):
            if "state" not in layer_state:
                continue
            cls = layer_state.get("class_name", "")

            if _is_dsv4_cache_class(cls):
                state = layer_state.get("state")
                if is_last_block and state is not None:
                    block_slices.append((
                        "deepseek_v4",
                        _to_numpy_tree(state),
                        layer_state.get("meta_state", ""),
                        cls,
                        _dsv4_cache_meta(layer_state),
                    ))
                else:
                    block_slices.append((
                        "deepseek_v4_pending",
                        cls,
                        _dsv4_cache_meta(layer_state),
                    ))
                continue

            if cls == "ZayaNoStateCache" or layer_state.get("no_state"):
                block_slices.append(("no_state", cls or "ZayaNoStateCache"))
                continue

            if _is_zaya_cca_cache_list_state(layer_state):
                # ZAYA CCA numpy path — not yet implemented; emit skip
                block_slices.append(("skip",))
                continue

            if cls == "CacheList":
                block_slices.append(("skip",))
                continue

            if idx in np_sources:
                np_k, np_v, orig_dtype = np_sources[idx]
                ndim = np_k.ndim
                if ndim == 4:
                    seq_len = np_k.shape[2]
                elif ndim == 3:
                    seq_len = np_k.shape[1]
                else:
                    block_slices.append(("skip",))
                    continue

                actual_end = min(end_idx, seq_len)
                if actual_end <= 0 or start_idx >= actual_end:
                    block_slices.append(("skip",))
                    continue

                if ndim == 4:
                    ks = np_k[:, :, start_idx:actual_end, :]
                    vs = np_v[:, :, start_idx:actual_end, :]
                else:
                    ks = np_k[:, start_idx:actual_end, :]
                    vs = np_v[:, start_idx:actual_end, :]

                ks_mx = mx.array(ks).astype(orig_dtype) if HAS_MLX else ks
                vs_mx = mx.array(vs).astype(orig_dtype) if HAS_MLX else vs

                if "Rotating" in cls:
                    meta = layer_state.get("meta_state", ())
                    max_size = seq_len
                    keep = 0
                    offset = None
                    idx_state = None
                    if meta and len(meta) >= 2:
                        try:
                            keep = int(meta[0])
                            max_size = int(meta[1])
                            if len(meta) >= 3:
                                offset = int(meta[2])
                            if len(meta) >= 4:
                                idx_state = int(meta[3])
                        except (ValueError, TypeError):
                            pass
                    block_slices.append((
                        "rotating_kv", ks_mx, vs_mx, max_size, keep, offset, idx_state
                    ))
                else:
                    block_slices.append(("kv", ks_mx, vs_mx))
            else:
                # Cumulative layer with no numpy source
                state = layer_state.get("state")
                if is_last_block and state is not None:
                    meta = layer_state.get("meta_state", "")
                    if isinstance(state, (list, tuple)):
                        np_state = []
                        for s in state:
                            if hasattr(s, "__array__"):
                                np_state.append(np.array(s))
                            else:
                                np_state.append(s)
                        block_slices.append(("cumulative", np_state, meta, cls))
                    else:
                        block_slices.append(("skip",))
                else:
                    block_slices.append(("skip",))

        return block_slices if block_slices else None

    # =========================================================================
    # Cache reconstruction
    # =========================================================================

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """Reconstruct cache objects from stored block data.

        Handles positional (KVCache) and cumulative (MambaCache/ArraysCache)
        layers:
        - KVCache: concatenates tensor slices from all blocks along seq axis.
        - MambaCache/ArraysCache: restores full cumulative state from last block.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from.

        Returns:
            List of reconstructed cache objects (one per layer), or None.
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning("Block %d not found in allocated blocks", block_id)
                    return None

                if block.cache_data is None:
                    _disk = getattr(self.paged_cache, "_disk_store", None)
                    if _disk is not None and block.block_hash is not None:
                        try:
                            _disk_data = _disk.read_block(block.block_hash)
                        except Exception as _re:
                            logger.warning("Block %d disk read failed: %s", block_id, _re)
                            _disk_data = None
                        if _disk_data is not None:
                            block.cache_data = _disk_data
                            block.cache_data_from_disk = True
                    if block.cache_data is None:
                        logger.debug("Block %d has no tensor data stored", block_id)
                        return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # Validate blocks before reconstructing tensors
            try:
                from ..cache.cache_record_validator import reject_or_warn as _reject_or_warn
            except Exception:
                _reject_or_warn = None
            if _reject_or_warn is not None:
                _expected_layers = getattr(self, "_expected_num_layers", None)
                if _expected_layers is None:
                    _expected_layers = num_layers
                for _bi, _block_data in enumerate(all_block_data):
                    if not _reject_or_warn(
                        _block_data,
                        expected_num_layers=_expected_layers,
                        source=f"reconstruct[block={_bi}]",
                    ):
                        return None

            # Import cache classes
            try:
                from mlx_lm.models.cache import KVCache, RotatingKVCache
            except ImportError:
                try:
                    from mlx_lm.models.cache import KVCache
                except ImportError:
                    logger.warning("mlx_lm not available — cannot reconstruct cache")
                    return None
                RotatingKVCache = None
            try:
                from mlx_lm.models.cache import MambaCache
                has_mamba = True
            except ImportError:
                MambaCache = None
                has_mamba = False
            has_rotating = RotatingKVCache is not None

            reconstructed_caches: List[Any] = []
            reconstructed_indices: set = set()

            for layer_idx in range(num_layers):
                layer_entries = []
                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        layer_entries.append(block_data[layer_idx])

                if not layer_entries:
                    continue

                best_cumulative = None
                best_dsv4 = None
                kv_slices_keys: List[Any] = []
                kv_slices_values: List[Any] = []
                rotating_kv_slices_keys: List[Any] = []
                rotating_kv_slices_values: List[Any] = []
                rotating_params = None
                quantized_kv_slices_keys: List[Any] = []
                quantized_kv_slices_values: List[Any] = []
                quantized_meta = None
                cache_list_entries: List[Any] = []
                zaya_cca_entries: List[Any] = []

                for entry in layer_entries:
                    if not isinstance(entry, (tuple, list)):
                        continue
                    tag = entry[0]
                    if tag == "kv":
                        kv_slices_keys.append(entry[1])
                        kv_slices_values.append(entry[2])
                    elif tag == "quantized_kv":
                        quantized_kv_slices_keys.append(entry[1])
                        quantized_kv_slices_values.append(entry[2])
                        if len(entry) > 3 and quantized_meta is None:
                            quantized_meta = entry[3]
                    elif tag == "rotating_kv":
                        rotating_kv_slices_keys.append(entry[1])
                        rotating_kv_slices_values.append(entry[2])
                        if len(entry) > 3 and rotating_params is None:
                            rotating_params = (
                                entry[3],
                                entry[4] if len(entry) > 4 else 0,
                                entry[5] if len(entry) > 5 else None,
                                entry[6] if len(entry) > 6 else None,
                            )
                    elif tag == "cumulative":
                        best_cumulative = entry
                    elif tag == "deepseek_v4":
                        best_dsv4 = entry
                    elif tag == "deepseek_v4_pending":
                        pass
                    elif tag == "no_state":
                        best_cumulative = entry
                    elif tag == "zaya_cca":
                        zaya_cca_entries.append(entry)
                    elif tag == "cache_list":
                        cache_list_entries.append(entry[1])

                # Reconstruct based on collected entry types
                if quantized_kv_slices_keys:
                    try:
                        from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                        n_comp = len(quantized_kv_slices_keys[0])
                        ck = tuple(
                            mx.concatenate([s[c] for s in quantized_kv_slices_keys], axis=-2)
                            for c in range(n_comp)
                        )
                        cv = tuple(
                            mx.concatenate([s[c] for s in quantized_kv_slices_values], axis=-2)
                            for c in range(n_comp)
                        )
                        mx.eval(*ck, *cv)
                        g_size, q_bits = 64, 8
                        if quantized_meta and len(quantized_meta) >= 3:
                            try:
                                _, g_size, q_bits = map(int, quantized_meta[:3])
                            except (ValueError, TypeError):
                                pass
                        cache = QKVCache(group_size=g_size, bits=q_bits)
                        cache.keys = ck
                        cache.values = cv
                        cache.offset = ck[0].shape[-2]
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                    except Exception as e:
                        logger.warning(
                            "Cannot reconstruct quantized layer %d: %s", layer_idx, e
                        )
                        return None

                elif kv_slices_keys:
                    ndim = len(kv_slices_keys[0].shape) if hasattr(kv_slices_keys[0], "shape") else 4
                    seq_axis = 1 if ndim == 3 else 2
                    try:
                        ck = mx.concatenate(kv_slices_keys, axis=seq_axis)
                        cv = mx.concatenate(kv_slices_values, axis=seq_axis)
                        mx.eval(ck, cv)
                        cache = KVCache()
                        cache.keys = ck
                        cache.values = cv
                        cache.offset = ck.shape[seq_axis]
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                    except Exception as e:
                        logger.warning(
                            "Cannot reconstruct KV layer %d: %s", layer_idx, e
                        )
                        return None

                elif rotating_kv_slices_keys and has_rotating:
                    try:
                        ndim = len(rotating_kv_slices_keys[0].shape) if hasattr(rotating_kv_slices_keys[0], "shape") else 4
                        seq_axis = 1 if ndim == 3 else 2
                        ck = mx.concatenate(rotating_kv_slices_keys, axis=seq_axis)
                        cv = mx.concatenate(rotating_kv_slices_values, axis=seq_axis)
                        mx.eval(ck, cv)
                        max_size, keep, offset, _idx = rotating_params or (ck.shape[seq_axis], 0, None, None)
                        cache = RotatingKVCache(max_size=max_size, keep=keep)
                        cache.keys = ck
                        cache.values = cv
                        if offset is not None:
                            cache.offset = offset
                        else:
                            cache.offset = ck.shape[seq_axis]
                        if _idx is not None and hasattr(cache, "_idx"):
                            cache._idx = _idx
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                    except Exception as e:
                        logger.warning(
                            "Cannot reconstruct RotatingKV layer %d: %s", layer_idx, e
                        )
                        return None

                elif best_dsv4 is not None:
                    # DSV4 composite cache reconstruction
                    try:
                        _, state, meta, class_name, _dsv4_meta = best_dsv4
                        # Attempt to import and reconstruct DSV4 cache
                        cache = None
                        try:
                            import mlx_lm.models.cache as _cm
                            cls_obj = getattr(_cm, class_name, None)
                            if cls_obj is not None and hasattr(cls_obj, "from_state"):
                                cache = cls_obj.from_state(state, meta)
                        except Exception:
                            pass
                        if cache is None:
                            cache = KVCache()
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                    except Exception as e:
                        logger.warning(
                            "Cannot reconstruct DSV4 layer %d: %s", layer_idx, e
                        )
                        return None

                elif best_cumulative is not None:
                    if best_cumulative[0] == "no_state":
                        class_name = (
                            best_cumulative[1] if len(best_cumulative) > 1
                            else "ZayaNoStateCache"
                        )
                        # Attempt ZayaNoStateCache reconstruction
                        try:
                            from rfsn_v11.models.zaya import ZayaNoStateCache  # type: ignore
                            cache = ZayaNoStateCache()
                        except ImportError:
                            logger.warning(
                                "Cannot reconstruct no_state layer %d: unknown class %s",
                                layer_idx, class_name,
                            )
                            return None
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                        continue

                    _, state, meta, class_name = best_cumulative
                    state = _copy_mlx_tree(state)
                    cache = None
                    try:
                        import mlx_lm.models.cache as cache_mod
                        cache_cls = getattr(cache_mod, class_name, None)
                        if cache_cls is None and class_name == "ArraysCache":
                            try:
                                from rfsn_v11.utils.mamba_cache import ArraysCache  # type: ignore
                                cache_cls = ArraysCache
                            except ImportError:
                                pass
                        if cache_cls and hasattr(cache_cls, "from_state"):
                            try:
                                cache = cache_cls.from_state(state, meta)
                            except (ValueError, TypeError):
                                cache = cache_cls.from_state(state, None)
                        elif has_mamba and "Mamba" in class_name:
                            cache = MambaCache.from_state(state, meta)
                        elif has_mamba:
                            cache = MambaCache.from_state(state, meta)
                        else:
                            cache = KVCache.from_state(state, meta)
                    except Exception:
                        try:
                            if has_mamba:
                                cache = MambaCache.from_state(state, meta)
                        except Exception:
                            pass
                    if cache is None:
                        logger.warning(
                            "Cannot reconstruct layer %d (%s): no suitable cache class",
                            layer_idx, class_name,
                        )
                        return None
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)

                elif cache_list_entries:
                    # CacheList (MoE models)
                    try:
                        from mlx_lm.models.cache import CacheList as CLCache
                    except ImportError:
                        logger.warning("Cannot reconstruct CacheList: import failed")
                        return None

                    num_subs = len(cache_list_entries[0])
                    sub_caches_rebuilt: List[Any] = []
                    for sub_idx in range(num_subs):
                        sub_kv_keys: List[Any] = []
                        sub_kv_vals: List[Any] = []
                        sub_qkv_keys: List[Any] = []
                        sub_qkv_vals: List[Any] = []
                        sub_qkv_meta = None
                        sub_cumulative = None
                        for block_entry in cache_list_entries:
                            if sub_idx >= len(block_entry):
                                continue
                            sub = block_entry[sub_idx]
                            if not isinstance(sub, (tuple, list)):
                                continue
                            st = sub[0]
                            if st == "kv":
                                sub_kv_keys.append(sub[1])
                                sub_kv_vals.append(sub[2])
                            elif st == "quantized_kv":
                                sub_qkv_keys.append(sub[1])
                                sub_qkv_vals.append(sub[2])
                                if len(sub) > 3 and sub_qkv_meta is None:
                                    sub_qkv_meta = sub[3]
                            elif st == "cumulative":
                                sub_cumulative = sub

                        if sub_qkv_keys:
                            n_comp = len(sub_qkv_keys[0])
                            ck = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_keys], axis=-2)
                                for c in range(n_comp)
                            )
                            cv = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_vals], axis=-2)
                                for c in range(n_comp)
                            )
                            mx.eval(*ck, *cv)
                            try:
                                from mlx_lm.models.cache import QuantizedKVCache as _QKV
                                g_size, q_bits = 64, 8
                                if sub_qkv_meta and len(sub_qkv_meta) >= 3:
                                    try:
                                        _, g_size, q_bits = map(int, sub_qkv_meta[:3])
                                    except (ValueError, TypeError):
                                        pass
                                sc = _QKV(group_size=g_size, bits=q_bits)
                                sc.keys = ck
                                sc.values = cv
                                sc.offset = ck[0].shape[-2]
                                sub_caches_rebuilt.append(sc)
                            except ImportError:
                                return None
                        elif sub_kv_keys:
                            ndim_s = len(sub_kv_keys[0].shape) if hasattr(sub_kv_keys[0], "shape") else 4
                            seq_ax = 1 if ndim_s == 3 else 2
                            ck = mx.concatenate(sub_kv_keys, axis=seq_ax)
                            cv = mx.concatenate(sub_kv_vals, axis=seq_ax)
                            mx.eval(ck, cv)
                            sc = KVCache()
                            sc.keys = ck
                            sc.values = cv
                            sc.offset = ck.shape[seq_ax]
                            sub_caches_rebuilt.append(sc)
                        elif sub_cumulative is not None:
                            _, s_state, s_meta, s_cls = sub_cumulative
                            s_state = _copy_mlx_tree(s_state)
                            try:
                                import mlx_lm.models.cache as _cm2
                                s_cls_obj = getattr(_cm2, s_cls, None)
                                if s_cls_obj and hasattr(s_cls_obj, "from_state"):
                                    sc = s_cls_obj.from_state(s_state, s_meta)
                                elif has_mamba:
                                    sc = MambaCache.from_state(s_state, s_meta)
                                else:
                                    sc = KVCache()
                            except Exception:
                                sc = KVCache()
                            sub_caches_rebuilt.append(sc)
                        else:
                            sub_caches_rebuilt.append(KVCache())

                    try:
                        cache = CLCache(sub_caches_rebuilt)
                    except Exception:
                        cache = sub_caches_rebuilt[0] if sub_caches_rebuilt else KVCache()
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)

                else:
                    # No usable data — skip layer with empty KVCache
                    reconstructed_caches.append(KVCache())
                    reconstructed_indices.add(layer_idx)

            logger.info(
                "Reconstructed cache: %d layers from %d blocks",
                len(reconstructed_caches),
                len(block_table.block_ids),
            )
            return reconstructed_caches if reconstructed_caches else None

        except Exception as e:
            logger.warning("reconstruct_cache failed: %s", e)
            return None

    # =========================================================================
    # Trim helpers
    # =========================================================================

    def trim_block_table(
        self,
        request_id: str,
        target_tokens: int,
    ) -> Optional[BlockTable]:
        """Shrink the block table to at most target_tokens (block-aligned).

        Args:
            request_id: Request whose block table to trim.
            target_tokens: Maximum number of tokens to retain.

        Returns:
            Trimmed BlockTable, or None if not possible.
        """
        block_table = self.paged_cache.get_block_table(request_id)
        if block_table is None or not block_table.block_ids:
            return None
        if target_tokens <= 0:
            return None

        kept_blocks_count = target_tokens // self.block_size
        if kept_blocks_count <= 0:
            return None
        kept_blocks_count = min(kept_blocks_count, len(block_table.block_ids))
        if kept_blocks_count == len(block_table.block_ids):
            return block_table

        to_release = block_table.block_ids[kept_blocks_count:]
        for block_id in to_release:
            self.paged_cache.decrement_ref(block_id)

        block_table.block_ids = block_table.block_ids[:kept_blocks_count]
        block_table.num_tokens = kept_blocks_count * self.block_size

        logger.debug(
            "Trimmed block_table for %s to %d blocks (%d tokens); "
            "released %d trailing blocks",
            request_id,
            kept_blocks_count,
            block_table.num_tokens,
            len(to_release),
        )
        return block_table

    @staticmethod
    def _block_has_terminal_state(block: Any, tag: str) -> bool:
        """Return True if a block carries a terminal path-dependent payload."""
        for entry in getattr(block, "cache_data", None) or []:
            if not isinstance(entry, (tuple, list)) or not entry:
                continue
            if entry[0] != tag:
                continue
            if tag == "zaya_cca":
                return len(entry) > 2 and entry[2] is not None
            return True
        return False

    # =========================================================================
    # DSV4 / ZAYA terminal state guards
    # =========================================================================

    @staticmethod
    def _iter_terminal_check_entries(
        block: Any,
        disk_store: Optional[Any] = None,
    ):
        """Yield cache_data entries from a block (loading from disk if needed)."""
        cache_data = getattr(block, "cache_data", None)
        if cache_data is None and disk_store is not None:
            bh = getattr(block, "block_hash", None)
            if bh is not None:
                try:
                    cache_data = disk_store.read_block(bh)
                except Exception:
                    cache_data = None
        for entry in (cache_data or []):
            yield entry

    @classmethod
    def _dsv4_l2_chain_missing_terminal_state(
        cls,
        cached_blocks: List[Any],
        disk_store: Optional[Any] = None,
    ) -> bool:
        """True when an L2 hit restored only non-terminal DSV4 blocks."""
        saw_pending = False
        saw_terminal = False
        for block in cached_blocks or []:
            for entry in cls._iter_terminal_check_entries(block, disk_store):
                if not isinstance(entry, (tuple, list)) or not entry:
                    continue
                tag = entry[0]
                if tag == "deepseek_v4_pending":
                    saw_pending = True
                elif tag == "deepseek_v4":
                    saw_terminal = True
        return saw_pending and not saw_terminal

    @classmethod
    def _zaya_l2_chain_missing_terminal_state(
        cls,
        cached_blocks: List[Any],
        disk_store: Optional[Any] = None,
    ) -> bool:
        """True when a ZAYA hit has KV pages but lacks terminal CCA state."""
        saw_zaya = False
        saw_terminal = False
        for block in cached_blocks or []:
            for entry in cls._iter_terminal_check_entries(block, disk_store):
                if not isinstance(entry, (tuple, list)) or not entry:
                    continue
                if entry[0] != "zaya_cca":
                    continue
                saw_zaya = True
                if len(entry) > 2 and entry[2] is not None:
                    saw_terminal = True
        return saw_zaya and not saw_terminal

    # =========================================================================
    # Priority-based release
    # =========================================================================

    def release_low_priority(self, n_entries: int = 1) -> int:
        """Release up to n_entries from the lowest-priority non-empty bucket.

        Eviction order: assistant -> user -> system.

        Args:
            n_entries: Maximum number of entries to release.

        Returns:
            Number of entries actually released.
        """
        released = 0
        for t in _CACHE_TYPE_PRIORITY:
            d = self._entries_by_type[t]
            while d and released < n_entries:
                rid, _ = d.popitem(last=False)
                if rid in self._request_tables:
                    self.release_cache(rid)
                    released += 1
            if released >= n_entries:
                break
        return released

    # =========================================================================
    # Fork
    # =========================================================================

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID.
            new_request_id: New request ID.

        Returns:
            Forked BlockTable, or None if source not found.
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,
            last_access=time.time(),
        )

        logger.debug("Forked cache: %s -> %s", source_request_id, new_request_id)
        return forked_table

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "tokens_saved": self._tokens_saved,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
            "paged_cache": self.paged_cache.get_memory_usage(),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        for request_id in list(self._request_tables.keys()):
            self.release_cache(request_id)
        self._prefix_index.clear()
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
