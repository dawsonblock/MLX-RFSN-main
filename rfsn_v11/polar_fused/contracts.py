"""Structured data contracts for rfsn_polar_fused.

All public interfaces return structured objects instead of anonymous tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


@dataclass(frozen=True)
class QuantizedVectors:
    """Result of polar quantization.

    Attributes
    ----------
    indices
        Packed or unpacked codebook indices. Shape depends on packing state.
    norms
        Per-vector L2 norms before rotation and quantization.
    original_dim
        Original head dimension (before rotation).
    bits
        Bit width used for quantization.
    rotation_id
        Identifier of the rotation matrix used.
    codebook_id
        Identifier of the codebook used.
    """

    indices: Any  # mx.array (uint8 when unpacked, uint32 when packed)
    norms: Any    # mx.array (float32)
    original_dim: int
    bits: int
    rotation_id: str
    codebook_id: str


@dataclass(frozen=False)
class PolarCacheState:
    """Compressed cache state for a single attention head.

    Attributes
    ----------
    key_indices
        Packed key codebook indices.
    key_norms
        Key norms per token.
    value_indices
        Packed value codebook indices.
    value_norms
        Value norms per token.
    offset
        Number of valid tokens currently stored.
    capacity
        Allocated token capacity (multiple of block size).
    """

    key_indices: Any     # mx.array
    key_norms: Any       # mx.array
    value_indices: Any   # mx.array
    value_norms: Any     # mx.array
    offset: int
    capacity: int


@dataclass(frozen=True)
class AttentionScoreResult:
    """Result of fused packed QK computation.

    Attributes
    ----------
    scores
        Attention scores before softmax.
    backend
        Which backend produced the result ("polar_fused", "fallback").
    """

    scores: Any  # mx.array
    backend: str


@dataclass(frozen=True)
class AttentionOutputResult:
    """Result of complete attention.

    Attributes
    ----------
    output
        Attention output tensor.
    backend
        Which backend produced the result.
    metrics
        Optional telemetry metrics.
    """

    output: Any  # mx.array
    backend: str
    metrics: dict[str, float] | None = None
