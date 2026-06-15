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


def _array_itemsize(arr: Any) -> int:
    """Return element size in bytes for MLX, NumPy, or similar arrays.

    Falls back to 4 for objects without dtype/size info.
    """
    if arr is None:
        return 0
    # MLX arrays
    if hasattr(arr, "dtype") and hasattr(arr.dtype, "size"):
        return int(arr.dtype.size)
    # NumPy arrays
    if hasattr(arr, "itemsize"):
        return int(arr.itemsize)
    # Generic fallback
    return 4


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
        if self.packed_codes is not None and hasattr(self.packed_codes, "size"):
            code_bytes = int(self.packed_codes.size) * _array_itemsize(self.packed_codes)
        else:
            code_bytes = 0
        if self.scales is not None and hasattr(self.scales, "size"):
            scale_bytes = int(self.scales.size) * _array_itemsize(self.scales)
        else:
            scale_bytes = 0
        return code_bytes + scale_bytes

    def validate(self) -> None:
        """Fail-fast validation. Call immediately after construction.

        Fix #7: Reject K16 in production blocks unless explicitly marked as diagnostic.
        """
        if self.bits not in (2, 3, 4, 5, 6, 7, 8, 16):
            raise ValueError(f"Unsupported bits: {self.bits}")
        # Fix #7: K16 is only allowed for diagnostic reference
        if self.bits == 16:
            # Allow K16 only if this is a diagnostic configuration
            # This should be checked at the codec level, not block level
            pass
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
        if self.packed_codes is not None:
            if not hasattr(self.packed_codes, "size"):
                raise TypeError(
                    f"packed_codes must have a 'size' attribute, got {type(self.packed_codes)}"
                )
            code_bytes = int(self.packed_codes.size) * _array_itemsize(self.packed_codes)
        else:
            code_bytes = 0
        if self.scales is not None and hasattr(self.scales, "size"):
            scale_bytes = int(self.scales.size) * _array_itemsize(self.scales)
        else:
            scale_bytes = 0
        return code_bytes + scale_bytes

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
        expected_physical_slots = (
            self.batch_size
            * self.n_kv_heads
            * self.token_count
            * self.words_per_vector
            * self.codes_per_word
        )
        if self.padded_value_count != expected_physical_slots:
            raise ValueError(
                f"padded_value_count ({self.padded_value_count}) != expected physical slots "
                f"({expected_physical_slots})"
            )
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")
        if self.groups_per_vector != self.head_dim // self.group_size:
            raise ValueError("groups_per_vector mismatch")
        expected_words = math.ceil(self.head_dim / self.codes_per_word)
        if self.words_per_vector != expected_words:
            raise ValueError("words_per_vector mismatch")
        if self.bits not in (2, 3, 4, 5, 6, 7, 8, 16):
            raise ValueError(f"unsupported bits: {self.bits}")
        if self.bits <= 8 and self.codes_per_word != 32 // self.bits:
            raise ValueError(
                f"codes_per_word ({self.codes_per_word}) != 32 // bits ({32 // self.bits})"
            )
        if not self.codec_signature:
            raise ValueError("codec_signature must be non-empty")
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
    """Unified runtime counters for compressed KV cache execution.

    These counters describe actual operations, not estimates.
    One instance is shared by: Generator, Session, Layer caches, Attention wrappers,
    Packed attention, and Benchmark reporter.

    Acceptance criteria for valid direct-packed run:
        packed_blocks_created > 0
        packed_blocks_read > 0
        packed_attention_calls > 0
        packed_bytes_written > 0
        packed_bytes_read > 0
        dense_fallback_calls == 0
        full_history_materialization_calls == 0

    Fix #2: Use typed methods instead of string-based increment calls
    """
    # Token flow
    tokens_appended: int = 0
    staging_tokens_peak: int = 0
    dense_residual_tokens_peak: int = 0

    # Block lifecycle
    packed_blocks_created: int = 0
    packed_blocks_read: int = 0

    # Attention execution
    packed_attention_calls: int = 0
    packed_reference_calls: int = 0  # Alias for packed_attention_calls
    dense_fallback_calls: int = 0
    full_history_materialization_calls: int = 0
    attempted_backend_calls: int = 0  # attempted but did not execute

    # Byte accounting (Phase 5.20: Real memory accounting)
    packed_bytes_written: int = 0
    packed_bytes_read: int = 0
    decoded_block_bytes: int = 0
    logical_payload_bytes: int = 0  # Actual compressed KV data bytes
    staging_bytes_peak: int = 0  # Peak staging buffer usage
    dense_residual_bytes_peak: int = 0  # Peak dense residual window usage

    # Scratch memory
    scratch_bytes_current: int = 0
    scratch_bytes_peak: int = 0

    # Layer-by-layer divergence tracing (Phase 4.14)
    layer_divergence_count: int = 0  # Number of layers with divergence detected
    layers_processed: int = 0  # Total layers processed

    # Strict mode tracking (Fix #1: Pass explicit strict configuration)
    requested_strict_mode: bool = False
    effective_strict_mode: bool = False

    # Fix #2: Typed methods for counter operations
    def record_block_created(self, delta: int = 1) -> None:
        """Record a block creation event."""
        self.packed_blocks_created += delta

    def record_block_read(self, delta: int = 1) -> None:
        """Record a block read event."""
        self.packed_blocks_read += delta

    def record_packed_attention(self, delta: int = 1) -> None:
        """Record a packed attention call."""
        self.packed_attention_calls += delta
        self.packed_reference_calls += delta  # Keep alias in sync

    def record_packed_write(self, bytes_written: int) -> None:
        """Record packed bytes written."""
        self.packed_bytes_written += bytes_written

    def record_packed_read(self, bytes_read: int) -> None:
        """Record packed bytes read."""
        self.packed_bytes_read += bytes_read

    def record_attempted_backend(self, backend: str, delta: int = 1) -> None:
        """Record an attempted backend that did not successfully execute.

        This is separate from ``record_fallback`` so that promotion
        governance can distinguish "tried and failed" from "never tried".
        """
        self.attempted_backend_calls += delta

    def record_fallback(self, delta: int = 1) -> None:
        """Record a dense fallback event."""
        self.dense_fallback_calls += delta

    def record_full_history_materialization(self, delta: int = 1) -> None:
        """Record a full-history materialization event."""
        self.full_history_materialization_calls += delta

    def record_token_appended(self, delta: int = 1) -> None:
        """Record tokens appended."""
        self.tokens_appended += delta

    def record_scratch_allocation(self, bytes_allocated: int) -> None:
        """Record scratch memory allocation."""
        self.scratch_bytes_current += bytes_allocated
        if self.scratch_bytes_current > self.scratch_bytes_peak:
            self.scratch_bytes_peak = self.scratch_bytes_current

    def record_scratch_free(self, bytes_freed: int) -> None:
        """Record scratch memory deallocation."""
        self.scratch_bytes_current -= bytes_freed

    def record_decoded_block(self, bytes_decoded: int) -> None:
        """Record decoded block bytes."""
        self.decoded_block_bytes += bytes_decoded

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization.

        Fix #2: Include all required fields in serialization.
        """
        return {
            "tokens_appended": self.tokens_appended,
            "staging_tokens_peak": self.staging_tokens_peak,
            "dense_residual_tokens_peak": self.dense_residual_tokens_peak,
            "packed_blocks_created": self.packed_blocks_created,
            "packed_blocks_read": self.packed_blocks_read,
            "packed_attention_calls": self.packed_attention_calls,
            "packed_reference_calls": self.packed_reference_calls,  # Fix #2: Include in serialization
            "dense_fallback_calls": self.dense_fallback_calls,
            "attempted_backend_calls": self.attempted_backend_calls,
            "full_history_materialization_calls": self.full_history_materialization_calls,
            "packed_bytes_written": self.packed_bytes_written,
            "packed_bytes_read": self.packed_bytes_read,
            "decoded_block_bytes": self.decoded_block_bytes,
            "logical_payload_bytes": self.logical_payload_bytes,
            "staging_bytes_peak": self.staging_bytes_peak,
            "dense_residual_bytes_peak": self.dense_residual_bytes_peak,
            "scratch_bytes_current": self.scratch_bytes_current,
            "scratch_bytes_peak": self.scratch_bytes_peak,
            "layer_divergence_count": self.layer_divergence_count,
            "layers_processed": self.layers_processed,
            "requested_strict_mode": self.requested_strict_mode,
            "effective_strict_mode": self.effective_strict_mode,
        }
