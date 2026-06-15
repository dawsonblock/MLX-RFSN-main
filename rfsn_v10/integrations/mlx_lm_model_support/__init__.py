"""MLX-LM model support for direct packed attention.

Provides model inspection, architecture validation, and an explicit
attention wrapper that replaces the model's standard attention with
packed blockwise attention over an RFSN quantized cache.
"""
from __future__ import annotations

from .attention_wrapper import (
    RfsnDirectPackedKVCache,
    install_packed_attention,
    is_model_wrapped,
    packed_attention_context,
    uninstall_packed_attention,
    unwrap_model_attention,
    wrap_model_attention,
)
from .model_support import (
    ModelArchitecture,
    inspect_model_architecture,
    is_supported_architecture,
)

__all__ = [
    "ModelArchitecture",
    "inspect_model_architecture",
    "is_supported_architecture",
    "RfsnDirectPackedKVCache",
    "install_packed_attention",
    "is_model_wrapped",
    "packed_attention_context",
    "uninstall_packed_attention",
    "wrap_model_attention",
    "unwrap_model_attention",
]
