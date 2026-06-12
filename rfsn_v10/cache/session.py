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
from .incremental_layer_cache import QuantizedLayerCache


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
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.model_id = model_id
        self.num_layers = num_layers
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window

        # One layer cache per layer
        self._layer_caches: dict[int, QuantizedLayerCache] = {
            i: QuantizedLayerCache(
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
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
        self._counters[counter] = self._counters.get(counter, 0) + delta

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all layer caches and release state."""
        for lc in self._layer_caches.values():
            lc.reset()
        self._layer_caches.clear()

    def __enter__(self) -> "GenerationCacheSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.destroy()
