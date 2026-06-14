"""Immutable configuration for rfsn_polar_fused backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PolarFusedConfig:
    """Strict immutable configuration for the Polar fused attention backend.

    Parameters
    ----------
    key_bits
        Bits per key coordinate (2, 3, or 4).
    value_bits
        Bits per value coordinate (2, 3, or 4).
    head_dim
        Attention head dimension (64 or 128 in the initial stable profile).
    key_rotation_seed
        Seed for deterministic key rotation matrix generation.
    value_rotation_seed
        Seed for deterministic value rotation matrix generation.
    allocation_block_tokens
        Cache capacity grows in multiples of this block size.
    lazy_quantization_tokens
        Keep FP16 cache until this token count, then bulk-convert.
    boundary_layers
        Number of first/last layers kept in FP16 for quality protection.
    enable_qk_fused
        Use fused packed QK kernel when available.
    enable_sv_fused
        Use fused packed SV kernel when available.
    enable_sparse_v
        Sparse V path (disabled by default).
    enable_rigidity_reuse
        Rigidity reuse optimization (disabled by default).
    allow_fallback
        Fall back to standard MLX attention when unsupported.
    require_batch_size_one
        Enforce batch_size == 1 (initial restriction).
    """

    key_bits: int = 4
    value_bits: int = 4
    head_dim: int = 128
    key_rotation_seed: int = 42
    value_rotation_seed: int = 43
    allocation_block_tokens: int = 256
    lazy_quantization_tokens: int = 1024
    boundary_layers: int = 2
    enable_qk_fused: bool = True
    enable_sv_fused: bool = True
    enable_sparse_v: bool = False
    enable_rigidity_reuse: bool = False
    allow_fallback: bool = True
    require_batch_size_one: bool = True

    def __post_init__(self) -> None:
        # Bits must be 2, 3, or 4 (Polar codebook limitation)
        if self.key_bits not in (2, 3, 4):
            raise ValueError(f"key_bits must be 2, 3, or 4; got {self.key_bits}")
        if self.value_bits not in (2, 3, 4):
            raise ValueError(f"value_bits must be 2, 3, or 4; got {self.value_bits}")

        # Head dimension must initially be 64 or 128
        if self.head_dim not in (64, 128):
            raise ValueError(f"head_dim must be 64 or 128; got {self.head_dim}")

        # Allocation block must be positive
        if self.allocation_block_tokens <= 0:
            raise ValueError(
                f"allocation_block_tokens must be positive; got {self.allocation_block_tokens}"
            )

        # Lazy threshold cannot be negative
        if self.lazy_quantization_tokens < 0:
            raise ValueError(
                f"lazy_quantization_tokens cannot be negative; got {self.lazy_quantization_tokens}"
            )

        # Sparse V cannot be enabled in the initial stable profile
        if self.enable_sparse_v:
            raise ValueError(
                "enable_sparse_v is not supported in the initial stable profile"
            )

        # Rigidity reuse cannot be enabled in the initial stable profile
        if self.enable_rigidity_reuse:
            raise ValueError(
                "enable_rigidity_reuse is not supported in the initial stable profile"
            )

    # ------------------------------------------------------------------
    # Predefined profiles
    # ------------------------------------------------------------------

    @classmethod
    def polar_safe(cls) -> PolarFusedConfig:
        """Conservative K4/V4 profile."""
        return cls(
            key_bits=4,
            value_bits=4,
            boundary_layers=2,
            lazy_quantization_tokens=1024,
            enable_sparse_v=False,
        )

    @classmethod
    def polar_balanced(cls) -> PolarFusedConfig:
        """Balanced K4/V3 profile."""
        return cls(
            key_bits=4,
            value_bits=3,
            boundary_layers=2,
            lazy_quantization_tokens=1024,
            enable_sparse_v=False,
        )

    @classmethod
    def polar_aggressive(cls) -> PolarFusedConfig:
        """Aggressive K3/V3 profile (experimental only)."""
        return cls(
            key_bits=3,
            value_bits=3,
            boundary_layers=4,
            lazy_quantization_tokens=1024,
            enable_sparse_v=False,
        )
