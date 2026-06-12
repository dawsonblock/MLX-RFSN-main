"""rfsn_polar_fused — clean-room PolarQuant reimplementation for RFSN."""
from __future__ import annotations

from .attention import NaivePolarAttention
from .attention_backend import PolarFusedAttentionBackend
from .cache import PolarCache
from .codebooks import CodebookRegistry, get_default_codebook_registry
from .config import PolarFusedConfig
from .contracts import (
    AttentionOutputResult,
    AttentionScoreResult,
    PolarCacheState,
    QuantizedVectors,
)
from .fallback import StandardMLXAttention
from .incremental_cache import IncrementalPolarCache
from .lazy_convert import CacheState, LazyPolarCache
from .promotion import PromotionEngine, PromotionLevel, PromotionStatus
from .quality_gates import PolarQualityGates, QualityGateResult
from .quantize import PolarQuantizer
from .rotations import RotationRegistry, get_default_rotation_registry
from .telemetry import PolarTelemetry, PolarTelemetryCollector

__all__ = [
    "PolarFusedConfig",
    "PolarCache",
    "PolarCacheState",
    "PolarQuantizer",
    "QuantizedVectors",
    "AttentionScoreResult",
    "AttentionOutputResult",
    "NaivePolarAttention",
    "PolarFusedAttentionBackend",
    "IncrementalPolarCache",
    "LazyPolarCache",
    "CacheState",
    "RotationRegistry",
    "get_default_rotation_registry",
    "CodebookRegistry",
    "get_default_codebook_registry",
    "StandardMLXAttention",
    "PolarQualityGates",
    "QualityGateResult",
    "PromotionEngine",
    "PromotionLevel",
    "PromotionStatus",
    "PolarTelemetry",
    "PolarTelemetryCollector",
]
