"""Data contracts for the incremental KV cache.

All public structures are immutable dataclasses.  No anonymous tuples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


class TensorLayout(StrEnum):
    BHTD = "BHTD"


class PackingLayout(StrEnum):
    GLOBAL_FLAT_V3 = "GLOBAL_FLAT_V3"
    VECTOR_ALIGNED_UINT32_V4 = "VECTOR_ALIGNED_UINT32_V4"


class ScaleLayout(StrEnum):
    FLAT_GROUPS_V3 = "FLAT_GROUPS_V3"
    BHTG_V4 = "BHTG_V4"


class Preconditioner(StrEnum):
    NONE = "NONE"
    WHT64_HASH_SIGN_V1 = "WHT64_HASH_SIGN_V1"


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


@dataclass(frozen=True, slots=True)
class PackedBlockV4:
    """Self-describing immutable sealed block (format version 4).

    V4 guarantees that metadata matches physical buffers exactly.
    A V4 block can be decoded without external shape guessing or
    mutable codec state.
    """
    packed_codes: Any
    scales: Any

    format_version: int
    tensor_layout: TensorLayout
    packing_layout: PackingLayout
    scale_layout: ScaleLayout
    preconditioner: Preconditioner

    batch_size: int
    n_kv_heads: int
    token_count: int
    head_dim: int

    logical_start: int
    logical_end: int

    bits: int
    group_size: int
    groups_per_vector: int
    codes_per_word: int
    words_per_vector: int

    original_value_count: int
    padded_value_count: int
    original_dtype: str

    sign_seed: int
    sign_algorithm: str
    layer_id: int
    stream_id: str

    codec_signature: str = ""

    # Legacy aliases for backward compatibility with V3 decoders
    @property
    def n_values(self) -> int:
        return self.padded_value_count

    @property
    def num_elements(self) -> int:
        return self.original_value_count

    @property
    def wht_applied(self) -> bool:
        return self.preconditioner == Preconditioner.WHT64_HASH_SIGN_V1

    @property
    def vector_alignment(self) -> int:
        return 64

    def payload_bytes(self) -> int:
        if HAS_MLX and self.packed_codes is not None:
            code_bytes = int(self.packed_codes.size) * 4
            scale_bytes = int(self.scales.size) * 4
            return code_bytes + scale_bytes
        return 0

    def validate(self) -> None:
        if self.format_version != 4:
            raise ValueError("unsupported PackedBlock format")
        if self.logical_start < 0:
            raise ValueError("logical_start must be nonnegative")
        if self.logical_end - self.logical_start != self.token_count:
            raise ValueError("logical range does not match token count")
        expected_values = (
            self.batch_size
            * self.n_kv_heads
            * self.token_count
            * self.head_dim
        )
        if self.original_value_count != expected_values:
            raise ValueError("original value count does not match geometry")
        if self.padded_value_count < self.original_value_count:
            raise ValueError("padded value count is too small")
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")
        if self.groups_per_vector != self.head_dim // self.group_size:
            raise ValueError("groups_per_vector mismatch")
        expected_words = math.ceil(self.head_dim / self.codes_per_word)
        if self.words_per_vector != expected_words:
            raise ValueError("words_per_vector mismatch")
        expected_code_shape = (
            self.batch_size,
            self.n_kv_heads,
            self.token_count,
            self.words_per_vector,
        )
        if tuple(self.packed_codes.shape) != expected_code_shape:
            raise ValueError("packed_codes shape mismatch")
        expected_scale_shape = (
            self.batch_size,
            self.n_kv_heads,
            self.token_count,
            self.groups_per_vector,
        )
        if tuple(self.scales.shape) != expected_scale_shape:
            raise ValueError("scales shape mismatch")


def validate_block_positions(blocks: list) -> None:
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


@dataclass(slots=True)
class RuntimeCounters:
    """Typed runtime counters that reconcile with physical cache state.

    These counters describe actual operations, not estimates.
    """
    tokens_received: int = 0
    tokens_staged: int = 0
    tokens_packed: int = 0
    tokens_dense_tail: int = 0
    tokens_reencoded_intentionally: int = 0

    blocks_created: int = 0
    blocks_read_reference: int = 0
    blocks_read_metal: int = 0

    reference_dense_calls: int = 0
    packed_reference_calls: int = 0
    packed_metal_calls: int = 0
    fallback_calls: int = 0

    current_scratch_bytes: int = 0
    peak_scratch_bytes: int = 0
    cumulative_scratch_traffic_bytes: int = 0
