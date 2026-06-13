"""rfsn_v10.cache — append-only compressed KV cache with bounded-memory attention.

This is the repaired incremental KV runtime.  It replaces the previous
monolithic kv_manager with:

1. Stateless Cartesian codec (K8/V5, WHT-64, group size 64)
2. Append-only per-layer cache (never recompresses sealed history)
3. Request-local generation sessions (isolated cache per request)
4. Bounded-memory blockwise attention (never reconstructs full K/V)
"""
from __future__ import annotations

# Phase 2 — Codec
from .cartesian_codec import CartesianCodec, PackedBlock

# Phase 3 — Layer cache
from .incremental_layer_cache import QuantizedLayerCache

# Phase 5 — Attention
from .memory import MemoryReport, measure_metal_peak_memory, measure_process_rss
from .reference_attention import BlockwiseReferenceAttention

# Phase 4 — Sessions
from .session import GenerationCacheSession

__all__ = [
    "CartesianCodec",
    "PackedBlock",
    "QuantizedLayerCache",
    "GenerationCacheSession",
    "MemoryReport",
    "measure_process_rss",
    "measure_metal_peak_memory",
    "BlockwiseReferenceAttention",
]
