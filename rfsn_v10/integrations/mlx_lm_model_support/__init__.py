"""MLX-LM model support for direct packed attention.

Provides model inspection, architecture validation, and an explicit
attention wrapper that replaces the model's standard attention with
packed blockwise attention over an RFSN quantized cache.
"""
from __future__ import annotations

from .model_support import (
    ModelArchitecture,
    inspect_model_architecture,
    is_supported_architecture,
)
from .attention_wrapper import (
    RfsnDirectPackedKVCache,
    wrap_model_attention,
    unwrap_model_attention,
)

__all__ = [
    "ModelArchitecture",
    "inspect_model_architecture",
    "is_supported_architecture",
    "RfsnDirectPackedKVCache",
    "wrap_model_attention",
    "unwrap_model_attention",
]
