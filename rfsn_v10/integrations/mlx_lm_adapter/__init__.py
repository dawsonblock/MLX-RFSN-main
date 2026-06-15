"""Explicit MLX-LM integration adapter for rfsn_v10.

No monkeypatching.  The adapter creates custom cache objects that implement
the MLX-LM KVCache interface, then passes them to MLX-LM's standard
generation functions.

Usage::

    from rfsn_v10.integrations.mlx_lm import RfsnMLXModelAdapter

    adapter = RfsnMLXModelAdapter(model, tokenizer, num_layers=24)
    text = adapter.generate("Hello", max_tokens=32)

Proof counters are tracked in the session and can be inspected after generation.
"""
from __future__ import annotations

from .adapter import (
    RfsnDenseReconstructionReferenceCache,
    RfsnMLXModelAdapter,
    RfsnMLXReferenceAdapter,
    RfsnQuantizedKVCache,
)
from .compatibility import (
    PINNED_MLX_LM_VERSION,
    PINNED_MLX_VERSION,
    check_mlx_lm_version,
    require_pinned_versions,
)

__all__ = [
    "RfsnMLXReferenceAdapter",
    "RfsnDenseReconstructionReferenceCache",
    # Deprecated aliases
    "RfsnMLXModelAdapter",
    "RfsnQuantizedKVCache",
    "check_mlx_lm_version",
    "require_pinned_versions",
    "PINNED_MLX_VERSION",
    "PINNED_MLX_LM_VERSION",
]

