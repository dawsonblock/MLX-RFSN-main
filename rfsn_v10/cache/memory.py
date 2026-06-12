"""Detailed memory measurement for the incremental KV cache.

Separates:
  * Packed codes (actual stored quantized data)
  * Scales (per-group scale factors)
  * Metadata (block headers, codec signatures)
  * Staging arrays (pending quantization buffers)
  * Dense residual (optional bounded FP16 window)
  * Attention scratch (temporary reconstruction buffers)
  * Allocator overhead (pre-allocated capacity)
  * Process RSS (system-level memory)

All measurements use actual array sizes, not estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]


@dataclass
class MemoryReport:
    """Complete memory breakdown for a cache session.

    All fields in bytes.  Zero means "not present" or "not measured".
    """
    # Core quantized payload
    packed_key_codes_bytes: int = 0
    packed_value_codes_bytes: int = 0
    key_scales_bytes: int = 0
    value_scales_bytes: int = 0

    # Metadata
    block_metadata_bytes: int = 0

    # Staging (pre-quantization)
    staging_keys_bytes: int = 0
    staging_values_bytes: int = 0

    # Dense residual (optional bounded window)
    dense_residual_keys_bytes: int = 0
    dense_residual_values_bytes: int = 0

    # Attention scratch (temporary, per-call)
    attention_scratch_bytes: int = 0

    # Allocator overhead (capacity minus actual usage)
    allocator_overhead_bytes: int = 0

    # System
    process_rss_bytes: int = 0

    # Dense shadow (temporary reconstructions from adapter)
    dense_shadow_bytes: int = 0

    # Token counts for ratio computation
    total_tokens: int = 0
    key_bits: int = 0
    value_bits: int = 0
    group_size: int = 0
    num_layers: int = 0

    # Computed properties
    @property
    def payload_bytes(self) -> int:
        """Actual stored quantized data (codes + scales)."""
        return (
            self.packed_key_codes_bytes
            + self.packed_value_codes_bytes
            + self.key_scales_bytes
            + self.value_scales_bytes
        )

    @property
    def staging_bytes(self) -> int:
        return self.staging_keys_bytes + self.staging_values_bytes

    @property
    def dense_residual_bytes(self) -> int:
        return self.dense_residual_keys_bytes + self.dense_residual_values_bytes

    @property
    def scratch_bytes(self) -> int:
        """All temporary/transient memory."""
        return self.attention_scratch_bytes + self.dense_shadow_bytes

    @property
    def total_accounted_bytes(self) -> int:
        """Everything we can account for."""
        return (
            self.payload_bytes
            + self.block_metadata_bytes
            + self.staging_bytes
            + self.dense_residual_bytes
            + self.scratch_bytes
            + self.allocator_overhead_bytes
        )

    @property
    def compression_ratio(self) -> float:
        """FP16 reference / compressed payload."""
        if self.payload_bytes == 0 or self.total_tokens == 0:
            return 1.0
        # FP16: 2 bytes per element, K+V = 2 tensors
        # Per token: num_layers * head_dim * 2 (kv heads implicit in shape)
        # We don't know head_dim here, so use a simple ratio
        fp16_ref = self.dense_shadow_bytes  # best proxy we have
        if fp16_ref == 0:
            return 1.0
        return fp16_ref / self.payload_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "packed_key_codes_bytes": self.packed_key_codes_bytes,
            "packed_value_codes_bytes": self.packed_value_codes_bytes,
            "key_scales_bytes": self.key_scales_bytes,
            "value_scales_bytes": self.value_scales_bytes,
            "block_metadata_bytes": self.block_metadata_bytes,
            "staging_keys_bytes": self.staging_keys_bytes,
            "staging_values_bytes": self.staging_values_bytes,
            "dense_residual_keys_bytes": self.dense_residual_keys_bytes,
            "dense_residual_values_bytes": self.dense_residual_values_bytes,
            "attention_scratch_bytes": self.attention_scratch_bytes,
            "allocator_overhead_bytes": self.allocator_overhead_bytes,
            "process_rss_bytes": self.process_rss_bytes,
            "dense_shadow_bytes": self.dense_shadow_bytes,
            "total_tokens": self.total_tokens,
            "payload_bytes": self.payload_bytes,
            "staging_bytes": self.staging_bytes,
            "dense_residual_bytes": self.dense_residual_bytes,
            "scratch_bytes": self.scratch_bytes,
            "total_accounted_bytes": self.total_accounted_bytes,
            "compression_ratio": self.compression_ratio,
        }


def measure_process_rss() -> int:
    """Return current process RSS in bytes."""
    try:
        import os
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss
    except Exception:
        return 0


def measure_metal_peak_memory() -> int:
    """Return peak Metal GPU memory in bytes."""
    if not HAS_MLX:
        return 0
    try:
        return int(mx.metal.get_peak_memory())
    except Exception:
        return 0


def measure_metal_active_memory() -> int:
    """Return currently active Metal GPU memory in bytes."""
    if not HAS_MLX:
        return 0
    try:
        return int(mx.metal.get_active_memory())
    except Exception:
        return 0
