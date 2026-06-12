"""Data contracts for the incremental KV cache.

All public structures are immutable dataclasses.  No anonymous tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


@dataclass(frozen=True)
class PackedBlock:
    """One immutable sealed block of packed quantized data.

    A block corresponds to a fixed number of tokens (the group_size or
    a multiple thereof).  Once sealed, a block never changes.
    """
    packed_codes: Any       # mx.array uint32, shape (n_words,)
    scales: Any             # mx.array float32, shape (n_groups,)
    token_count: int        # Number of tokens represented
    bits: int               # Bit width per coordinate
    group_size: int         # Elements per scale
    n_values: int           # Original number of quantized values

    def payload_bytes(self) -> int:
        """Exact bytes from the stored arrays."""
        if HAS_MLX and self.packed_codes is not None:
            code_bytes = int(self.packed_codes.size) * 4
            scale_bytes = int(self.scales.size) * 4
            return code_bytes + scale_bytes
        return 0


@dataclass(frozen=True)
class CacheStats:
    """Runtime statistics for a layer cache."""
    tokens_encoded: int = 0
    tokens_requantized: int = 0
    sealed_blocks: int = 0
    staged_tokens: int = 0
    dense_residual_tokens: int = 0
    payload_bytes: int = 0


@dataclass(frozen=True)
class AttentionScratch:
    """Per-attention-call scratch memory accounting."""
    max_reconstructed_block_tokens: int = 0
    score_vector_bytes: int = 0
    output_accumulator_bytes: int = 0
