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
    """Immutable sealed block (format version 3).

    V3 changes over V2:
      * ``format_version`` defaults to 3.
      * ``vector_alignment`` — required SIMD alignment (64 for K8/V5).
      * ``validate()`` — strict range checks on every field.
    """
    packed_codes: Any       # mx.array uint32
    scales: Any             # mx.array float32
    token_count: int
    bits: int
    group_size: int
    n_values: int

    batch_size: int = 1
    n_kv_heads: int = 0
    head_dim: int = 0
    logical_start: int = 0
    original_dtype: str = "float16"
    format_version: int = 3
    num_elements: int = 0
    wht_applied: bool = False
    sign_seed: int = 0
    vector_alignment: int = 64  # NEW in V3

    def payload_bytes(self) -> int:
        if HAS_MLX and self.packed_codes is not None:
            code_bytes = int(self.packed_codes.size) * 4
            scale_bytes = int(self.scales.size) * 4
            return code_bytes + scale_bytes
        return 0

    def validate(self) -> None:
        """Fail-fast validation. Call immediately after construction."""
        if self.bits not in (2, 3, 4, 5, 6, 7, 8, 16):
            raise ValueError(f"Unsupported bits: {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"Invalid group_size: {self.group_size}")
        if self.token_count < 0:
            raise ValueError(f"Invalid token_count: {self.token_count}")
        if self.n_values < 0:
            raise ValueError(f"Invalid n_values: {self.n_values}")
        if self.num_elements < 0:
            raise ValueError(f"Invalid num_elements: {self.num_elements}")
        if self.logical_start < 0:
            raise ValueError(f"Invalid logical_start: {self.logical_start}")
        if self.vector_alignment <= 0:
            raise ValueError(f"Invalid vector_alignment: {self.vector_alignment}")
        # Geometry self-consistency: if geometry is set, scales must match BHTG
        if self.n_kv_heads > 0 and self.head_dim > 0 and self.token_count > 0:
            groups_per_head = self.head_dim // self.group_size
            expected_scale_elements = self.batch_size * self.n_kv_heads * self.token_count * groups_per_head
            if self.scales is not None and int(self.scales.size) != expected_scale_elements:
                raise ValueError(
                    f"scales size ({int(self.scales.size)}) != expected BHTG "
                    f"({expected_scale_elements}) for "
                    f"batch={self.batch_size}, heads={self.n_kv_heads}, "
                    f"tokens={self.token_count}, groups_per_head={groups_per_head}"
                )
        # Payload sanity: n_values should be close to num_elements when no WHT
        if self.num_elements > 0 and self.n_values > 0:
            padded = self.num_elements + (
                (self.group_size - (self.num_elements % self.group_size)) % self.group_size
            )
            if self.n_values != padded and not self.wht_applied:
                raise ValueError(
                    f"n_values ({self.n_values}) != padded ({padded}) for "
                    f"num_elements={self.num_elements}, group_size={self.group_size}"
                )


def validate_block_positions(blocks: list[PackedBlock]) -> None:
    """Validate that blocks are ordered and non-overlapping."""
    for i in range(1, len(blocks)):
        prev = blocks[i - 1]
        curr = blocks[i]
        prev_end = prev.logical_start + prev.token_count
        if curr.logical_start != prev_end:
            raise ValueError(
                f"Block position gap/overlap at index {i}: "
                f"prev ends at {prev_end}, curr starts at {curr.logical_start}"
            )


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
