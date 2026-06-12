"""rfsn_polar_fused — clean-room PolarQuant reimplementation for RFSN."""
from __future__ import annotations

from .config import PolarFusedConfig
from .contracts import (
    AttentionOutputResult,
    AttentionScoreResult,
    PolarCacheState,
    QuantizedVectors,
)
from .cache import PolarCache
from .codebooks import CodebookRegistry, get_default_codebook_registry
from .fallback import StandardMLXAttention
from .quantize import PolarQuantizer
from .rotations import RotationRegistry, get_default_rotation_registry

__all__ = [
    "PolarFusedConfig",
    "PolarCache",
    "PolarCacheState",
    "PolarQuantizer",
    "QuantizedVectors",
    "AttentionScoreResult",
    "AttentionOutputResult",
    "RotationRegistry",
    "get_default_rotation_registry",
    "CodebookRegistry",
    "get_default_codebook_registry",
    "StandardMLXAttention",
]
