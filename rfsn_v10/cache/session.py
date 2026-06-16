"""Request-local generation cache sessions.

Each generation owns isolated cache state keyed by:
    session_id | model_id | layer_id | codec_signature

No process-global request cache sharing.  Cache is reliably destroyed on
completion, error, cancellation, or disconnect.
"""
from __future__ import annotations

import uuid
from typing import Any

from .cartesian_codec import CartesianCodec
from .contracts import RuntimeCounters
from .incremental_layer_cache import QuantizedLayerCache
from .memory import MemoryReport, measure_process_rss
from .paged_arena import validate_direct_packed_format


class GenerationCacheSession:
    """Per-generation cache session.

    Creates one QuantizedLayerCache per model layer.
    All state is isolated; no sharing across sessions.
    """

    def __init__(
        self,
        model_id: str,
        num_layers: int,
        key_codec: CartesianCodec,
        value_codec: CartesianCodec,
        staging_capacity: int = 64,
        dense_residual_window: int = 0,
        use_paged_arena: bool = False,
        max_pages: int = 256,
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.model_id = model_id
        self.num_layers = num_layers
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window
        self.use_paged_arena = use_paged_arena
        self.max_pages = max_pages

        # Direct packed paging is only valid for K8/V8 GS64.
        if use_paged_arena:
            validate_direct_packed_format(key_codec, value_codec, label="session")

        # One layer cache per layer
        self._layer_caches: dict[int, QuantizedLayerCache] = {
            i: QuantizedLayerCache(
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
                layer_id=i,
                session=self,
                use_paged_arena=use_paged_arena,
                max_pages=max_pages,
            )
            for i in range(num_layers)
        }

        # Proof counters (aggregated across all layers)
        self._counters: dict[str, int] = {
            "new_tokens_received": 0,
            "new_tokens_encoded": 0,
            "packed_blocks_created": 0,
            "sealed_blocks_read": 0,
            "fallback_attention_calls": 0,
            "dense_shadow_bytes": 0,
            "requantized_tokens": 0,
        }

        # Typed runtime counters (Phase 10)
        self.runtime_counters = RuntimeCounters()
        # Fix #1: Track strict mode in runtime counters
        self.runtime_counters.requested_strict_mode = False
        self.runtime_counters.effective_strict_mode = False

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_layer_cache(self, layer_id: int) -> QuantizedLayerCache:
        if layer_id not in self._layer_caches:
            raise KeyError(f"Layer {layer_id} not in session")
        return self._layer_caches[layer_id]

    def all_layer_caches(self) -> dict[int, QuantizedLayerCache]:
        return dict(self._layer_caches)

    # ------------------------------------------------------------------
    # Proof counters
    # ------------------------------------------------------------------

    def increment(self, counter: str, delta: int = 1) -> None:
        """Legacy string-based counter increment.

        Fix #2: This method is deprecated. Use typed methods on runtime_counters directly.
        This method is kept for backward compatibility but will be removed in future.
        """
        self._counters[counter] = self._counters.get(counter, 0) + delta
        # Sync typed runtime counters with new unified schema using typed methods.
        # tokens_appended is owned by layer_cache.append() via record_token_appended();
        # do NOT also increment it here from new_tokens_received to avoid double-counting.
        if counter == "new_tokens_received":
            # Legacy dict-counter only; tokens_appended is handled by layer_cache.append()
            pass
        elif counter == "new_tokens_encoded":
            # Don't double-count - already counted by new_tokens_received
            pass
        elif counter == "packed_blocks_created":
            self.runtime_counters.record_block_created(delta)
        elif counter == "sealed_blocks_read":
            self.runtime_counters.record_block_read(delta)
        elif counter == "fallback_attention_calls":
            self.runtime_counters.record_fallback(delta)
        elif counter == "dense_shadow_bytes":
            self.runtime_counters.record_scratch_allocation(delta)

    def track_layer_divergence(self, layer_id: int, has_divergence: bool) -> None:
        """Track layer-by-layer divergence for debugging (Phase 4.14)."""
        self.runtime_counters.layers_processed += 1
        if has_divergence:
            self.runtime_counters.layer_divergence_count += 1

    def track_payload_bytes(self, bytes_count: int) -> None:
        """Track actual compressed KV payload bytes (Phase 5.20)."""
        self.runtime_counters.logical_payload_bytes += bytes_count

    def get_counter(self, counter: str) -> int:
        return self._counters.get(counter, 0)

    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def total_payload_bytes(self) -> int:
        return sum(lc.payload_bytes() for lc in self._layer_caches.values())

    def total_dense_residual_bytes(self) -> int:
        return sum(lc.dense_residual_bytes() for lc in self._layer_caches.values())

    def total_staging_bytes(self) -> int:
        return sum(lc.staging_bytes() for lc in self._layer_caches.values())

    def total_memory_bytes(self) -> int:
        return sum(lc.total_memory_bytes() for lc in self._layer_caches.values())

    def memory_report(self) -> MemoryReport:
        """Return a detailed memory report for this session."""
        report = MemoryReport(
            key_bits=self.key_codec.bits,
            value_bits=self.value_codec.bits,
            group_size=self.key_codec.group_size,
            num_layers=self.num_layers,
            process_rss_bytes=measure_process_rss(),
        )

        for lc in self._layer_caches.values():
            stats = lc.stats()
            report.total_tokens += (
                stats.tokens_encoded
                + stats.staged_tokens
                + stats.dense_residual_tokens
            )

            # Payload: sealed blocks or paged arena
            from rfsn_v10.cache.contracts import _array_itemsize
            arena_stats = lc.get_paged_arena_stats()
            if arena_stats is not None:
                report.packed_key_codes_bytes += arena_stats.get(
                    "active_payload_bytes", 0
                )
                report.block_metadata_bytes += arena_stats.get(
                    "page_metadata_bytes", 0
                )
                report.allocator_overhead_bytes += arena_stats.get(
                    "allocator_overhead_bytes", 0
                )
            else:
                for kb in lc.iter_key_blocks():
                    report.packed_key_codes_bytes += int(kb.packed_codes.size) * _array_itemsize(kb.packed_codes)
                    report.key_scales_bytes += int(kb.scales.size) * _array_itemsize(kb.scales)
                    report.block_metadata_bytes += 32  # approximate per-block header

                for vb in lc.iter_value_blocks():
                    report.packed_value_codes_bytes += int(vb.packed_codes.size) * _array_itemsize(vb.packed_codes)
                    report.value_scales_bytes += int(vb.scales.size) * _array_itemsize(vb.scales)
                    report.block_metadata_bytes += 32

            # Staging
            sk, sv, sn = lc.get_staging()
            if sk is not None:
                report.staging_keys_bytes += int(sk.size) * _array_itemsize(sk)
            if sv is not None:
                report.staging_values_bytes += int(sv.size) * _array_itemsize(sv)

            # Dense residual
            dk, dv = lc.get_dense_residual()
            if dk is not None:
                report.dense_residual_keys_bytes += int(dk.size) * 2  # FP16
            if dv is not None:
                report.dense_residual_values_bytes += int(dv.size) * 2

        # Dense shadow from counters
        report.dense_shadow_bytes = self._counters.get("dense_shadow_bytes", 0)

        return report

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all layer caches and release state."""
        for lc in self._layer_caches.values():
            lc.reset()
        self._layer_caches.clear()

    def __enter__(self) -> GenerationCacheSession:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.destroy()
