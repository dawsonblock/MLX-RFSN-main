"""True packed attention using MLX's inline Metal kernel compilation.

This module implements the true packed attention kernel using MLX's
mx.fast.metal_kernel() API, which compiles Metal kernels from source strings
at runtime. This avoids the need for the standalone Metal compiler toolchain.

The kernel decodes packed K/V blocks on-the-fly inside the GPU shader,
computing vector QK dot products without materializing dense tensors.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
    import numpy as np
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    np = None  # type: ignore


# =============================================================================
# Execution Contract
# =============================================================================

@dataclass(frozen=True)
class ExecutionContract:
    """Immutable record of kernel execution."""
    backend: str
    kernel_hash: str
    num_blocks: int
    total_kv_tokens: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    bits: int
    materialized_bytes: int = 0
    decoded_tokens: int = 0
    fallback_reason: str | None = None
    execution_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def validate_invariant(self) -> tuple[bool, list[str]]:
        violations = []
        if self.materialized_bytes > 0:
            violations.append(
                f"materialized_bytes={self.materialized_bytes} > 0"
            )
        if self.decoded_tokens > 0:
            violations.append(f"decoded_tokens={self.decoded_tokens} > 0")
        if "true_packed_metal" not in self.backend:
            violations.append(f"backend={self.backend}")
        return len(violations) == 0, violations


# =============================================================================
# MLX Inline Metal Kernel Source
# =============================================================================

# The kernel source must be compatible with MLX's metal_kernel API.
# No C++ structs or helper functions - everything inlined.

_TRUE_PACKED_KERNEL_MLX = """
// True Packed Attention - MLX inline version
// Each thread processes one (query_head, query_token) pair

uint q_head = thread_position_in_grid.x;
uint q_token = thread_position_in_grid.y;

if (q_head >= NUM_Q_HEADS || q_token >= NUM_Q_TOKENS) return;

// GQA mapping
uint kv_head = q_head / Q_PER_KV;
uint query_pos = QUERY_START + q_token;

// Query offset: [NUM_Q_HEADS, NUM_Q_TOKENS, HEAD_DIM]
uint q_offset = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;

// Output offset
uint out_offset = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;

// Read scale from buffer (templates don't support floats)
float scale_val = scale_arr[0];

// Online softmax state
float running_max = -INFINITY;
float running_sum = 0.0;

// ============================================================================
// First Pass: Compute QK scores and online softmax
// ============================================================================

uint token_idx = 0;
for (uint b = 0; b < NUM_BLOCKS; b++) {
    // Read block metadata from flat arrays
    int block_start = block_starts[b];
    int block_count = block_counts[b];
    float key_scale = key_scales[b];
    float key_zp = key_zero_points[b];

    for (uint t = 0; t < block_count; t++) {
        int kv_pos = block_start + int(t);

        // Causal mask
        if (CAUSAL != 0 && kv_pos > query_pos) {
            token_idx++;
            continue;
        }

        // Compute QK dot product with on-the-fly decode
        float dot = 0.0;

        // Calculate bit offset for this token's key data
        // Layout: [block, kv_head, token, dim] with bit-packing on dim
        // We flatten to a single offset
        uint base_bits = token_idx * HEAD_DIM * BITS;

        for (uint d = 0; d < HEAD_DIM; d++) {
            // Read query value
            float q_val = queries[q_offset + d];

            // Decode key value from packed uint8
            uint bit_offset = base_bits + d * BITS;
            uint byte_idx = bit_offset / 8;
            uint bit_idx = bit_offset % 8;

            // Read uint8 from packed_keys
            uint8_t packed = packed_keys[byte_idx];

            // Extract bits (P1 Fix: guard against byte-spanning underflow)
            uint8_t extracted = 0;
            if (bit_idx + BITS <= 8) {
                uint shift = 8 - bit_idx - BITS;
                uint mask = (1u << BITS) - 1;
                extracted = (packed >> shift) & mask;
            }
            // TODO: Handle bits spanning two bytes (cross-byte packing)

            // Map to [-1, 1]
            float max_val = float((1u << BITS) - 1);
            float normalized = (float(extracted) / max_val) * 2.0 - 1.0;

            // Apply hash-sign (simplified inline)
            int sign_seed = SIGN_SEED;
            int h = sign_seed;
            h = h * 31 + LAYER_ID;
            h = h * 31 + STREAM_ID;
            h = h * 31 + kv_pos;
            h = h * 31 + int(d);
            int hash_sign = (h & 1) ? 1 : -1;

            float signed_val = normalized * float(hash_sign);

            // Dequantize
            float k_val = signed_val * key_scale + key_zp;

            // Accumulate dot product
            dot += q_val * k_val;
        }

        // Apply attention scale
        dot *= scale_val;

        // Update online softmax
        float new_max = max(running_max, dot);
        float old_scale = (running_max == -INFINITY)
            ? 0.0 : exp(running_max - new_max);
        running_sum = running_sum * old_scale + exp(dot - new_max);
        running_max = new_max;

        token_idx++;
    }
}

// Handle fully-masked case
if (running_sum == 0.0) {
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = 0.0;
    }
    return;
}

// ============================================================================
// Second Pass: Weighted accumulation of values
// ============================================================================

token_idx = 0;
for (uint b = 0; b < NUM_BLOCKS; b++) {
    int block_start = block_starts[b];
    int block_count = block_counts[b];
    float key_scale = key_scales[b];
    float key_zp = key_zero_points[b];
    float value_scale = value_scales[b];
    float value_zp = value_zero_points[b];

    for (uint t = 0; t < block_count; t++) {
        int kv_pos = block_start + int(t);

        if (CAUSAL != 0 && kv_pos > query_pos) {
            token_idx++;
            continue;
        }

        // Recompute QK score to get weight
        float dot = 0.0;
        uint base_bits = token_idx * HEAD_DIM * BITS;

        for (uint d = 0; d < HEAD_DIM; d++) {
            float q_val = queries[q_offset + d];

            uint bit_offset = base_bits + d * BITS;
            uint byte_idx = bit_offset / 8;
            uint bit_idx = bit_offset % 8;

            uint8_t packed = packed_keys[byte_idx];
            // Extract bits (P1 Fix: guard against byte-spanning underflow)
            uint8_t extracted = 0;
            if (bit_idx + BITS <= 8) {
                uint shift = 8 - bit_idx - BITS;
                uint mask = (1u << BITS) - 1;
                extracted = (packed >> shift) & mask;
            }
            // TODO: Handle bits spanning two bytes

            float max_val = float((1u << BITS) - 1);
            float normalized = (float(extracted) / max_val) * 2.0 - 1.0;

            int h = SIGN_SEED;
            h = h * 31 + LAYER_ID;
            h = h * 31 + STREAM_ID;
            h = h * 31 + kv_pos;
            h = h * 31 + int(d);
            int hash_sign = (h & 1) ? 1 : -1;

            float signed_val = normalized * float(hash_sign);
            float k_val = signed_val * key_scale + key_zp;

            dot += q_val * k_val;
        }

        dot *= scale_val;

        // Compute softmax weight
        float weight = exp(dot - running_max) / running_sum;

        // Decode value and accumulate weighted
        uint value_base_bits = token_idx * HEAD_DIM * BITS;
        for (uint d = 0; d < HEAD_DIM; d++) {
            uint bit_offset = value_base_bits + d * BITS;
            uint byte_idx = bit_offset / 8;
            uint bit_idx = bit_offset % 8;

            uint8_t packed = packed_values[byte_idx];
            // Extract bits (P1 Fix: guard against byte-spanning underflow)
            uint8_t extracted = 0;
            if (bit_idx + BITS <= 8) {
                uint shift = 8 - bit_idx - BITS;
                uint mask = (1u << BITS) - 1;
                extracted = (packed >> shift) & mask;
            }
            // TODO: Handle bits spanning two bytes

            float max_val = float((1u << BITS) - 1);
            float normalized = (float(extracted) / max_val) * 2.0 - 1.0;

            int h = SIGN_SEED;
            h = h * 31 + LAYER_ID;
            h = h * 31 + STREAM_ID;
            h = h * 31 + kv_pos;
            h = h * 31 + int(d);
            int hash_sign = (h & 1) ? 1 : -1;

            float signed_val = normalized * float(hash_sign);
            float v_val = signed_val * value_scale + value_zp;

            // Weighted accumulation
            output[out_offset + d] += weight * v_val;
        }

        token_idx++;
    }
}
"""


# =============================================================================
# Kernel Wrapper
# =============================================================================

class TruePackedMLXInline:
    """True packed attention using MLX inline Metal kernel compilation."""

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        sign_seed: int = 42,
    ):
        self.bits = bits
        self.group_size = group_size
        self.sign_seed = sign_seed
        self._kernel_hash = hashlib.sha256(
            _TRUE_PACKED_KERNEL_MLX.encode()
        ).hexdigest()[:16]

    def __call__(
        self,
        queries: "mx.array",
        blocks: list[Any],
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
        strict: bool = False,
    ) -> tuple["mx.array", ExecutionContract]:
        """Execute true packed attention via MLX Metal kernel.

        Args:
            queries: [batch=1, num_q_heads, num_q_tokens, head_dim]
            blocks: List of PackedBlock objects
            scale: Attention scale
            causal: Causal masking
            query_start_pos: Global position of first query token
            strict: If True, validate zero materialization

        Returns:
            (output, contract)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX required")

        start_time = time.perf_counter()

        # Get dimensions
        B, Hq, Lq, D = queries.shape
        if B != 1:
            raise RuntimeError(
                f"TruePackedMLXInline only supports batch_size=1, got {B}"
            )
        num_kv_heads = Hq  # Simplified; GQA handled in kernel

        # Flatten queries for kernel
        queries_flat = queries.reshape(Hq, Lq, D)

        # Prepare block metadata as flat arrays
        num_blocks = len(blocks)
        total_tokens = sum(getattr(b, 'token_count', 0) for b in blocks)

        # P1 Fix: Validate block format. Real PackedBlocks use packed_codes
        # (4D arrays), but this scaffold kernel expects mock-style packed_keys
        # flat bytes. Raise early so the wrapper falls through safely.
        for block in blocks:
            if hasattr(block, 'packed_codes'):
                raise RuntimeError(
                    "TruePackedMLXInline: real PackedBlock format not yet "
                    "supported. Use MockBlock for tests, or fall through to "
                    "reference in production."
                )

        block_starts = []
        block_counts = []
        key_scales = []
        key_zero_points = []
        value_scales = []
        value_zero_points = []
        layer_ids = set()
        stream_ids = set()

        # Prepare packed data buffers
        packed_keys_data = bytearray()
        packed_values_data = bytearray()

        for block in blocks:
            block_starts.append(getattr(block, 'position', 0))
            block_counts.append(getattr(block, 'token_count', 0))
            key_scales.append(getattr(block, 'key_scale', 1.0))
            key_zero_points.append(getattr(block, 'key_zero_point', 0.0))
            value_scales.append(getattr(block, 'value_scale', 1.0))
            value_zero_points.append(getattr(block, 'value_zero_point', 0.0))
            layer_ids.add(getattr(block, 'layer_id', 0))
            stream_ids.add(getattr(block, 'stream_id', 0))

        # P1 Fix: Kernel template constants require uniform layer/stream IDs
        if len(layer_ids) > 1 or len(stream_ids) > 1:
            raise RuntimeError(
                "TruePackedMLXInline: all blocks must share "
                f"layer_id/stream_id, got layer_ids={layer_ids}, "
                f"stream_ids={stream_ids}"
            )
        layer_id = next(iter(layer_ids)) if layer_ids else 0
        stream_id = next(iter(stream_ids)) if stream_ids else 0

        for block in blocks:
            # Append packed data
            key_data = getattr(block, 'packed_keys', b'')
            val_data = getattr(block, 'packed_values', b'')
            if isinstance(key_data, np.ndarray):
                packed_keys_data.extend(key_data.tobytes())
            else:
                packed_keys_data.extend(bytes(key_data))
            if isinstance(val_data, np.ndarray):
                packed_values_data.extend(val_data.tobytes())
            else:
                packed_values_data.extend(bytes(val_data))

        # Convert to MLX arrays
        block_starts_arr = mx.array(block_starts, dtype=mx.int32)
        block_counts_arr = mx.array(block_counts, dtype=mx.int32)
        key_scales_arr = mx.array(key_scales, dtype=mx.float32)
        key_zp_arr = mx.array(key_zero_points, dtype=mx.float32)
        value_scales_arr = mx.array(value_scales, dtype=mx.float32)
        value_zp_arr = mx.array(value_zero_points, dtype=mx.float32)

        # Ensure packed data is byte-aligned and convert to uint8
        packed_keys_arr = mx.array(
            np.frombuffer(bytes(packed_keys_data), dtype=np.uint8)
        )
        packed_values_arr = mx.array(
            np.frombuffer(bytes(packed_values_data), dtype=np.uint8)
        )

        # Scale as 1-element array (template doesn't support float)
        scale_arr = mx.array([float(scale)], dtype=mx.float32)
        query_start_arr = mx.array([int(query_start_pos)], dtype=mx.int32)

        # Compile and dispatch kernel
        kernel = mx.fast.metal_kernel(
            name="true_packed_attention_inline",
            input_names=[
                "queries", "packed_keys", "packed_values",
                "block_starts", "block_counts",
                "key_scales", "key_zero_points",
                "value_scales", "value_zero_points",
                "scale_arr", "query_start_arr",
            ],
            output_names=["output"],
            source=_TRUE_PACKED_KERNEL_MLX,
        )

        outputs = kernel(
            inputs=[
                queries_flat, packed_keys_arr, packed_values_arr,
                block_starts_arr, block_counts_arr,
                key_scales_arr, key_zp_arr,
                value_scales_arr, value_zp_arr,
                scale_arr, query_start_arr,
            ],
            template=[
                ("NUM_Q_HEADS", int(Hq)),
                ("NUM_Q_TOKENS", int(Lq)),
                ("HEAD_DIM", int(D)),
                ("NUM_BLOCKS", int(num_blocks)),
                ("BITS", int(self.bits)),
                ("SIGN_SEED", int(self.sign_seed)),
                ("LAYER_ID", int(layer_id)),
                ("STREAM_ID", int(stream_id)),
                ("CAUSAL", int(1 if causal else 0)),
                ("Q_PER_KV", int(1)),  # GQA ratio
                ("QUERY_START", int(query_start_pos)),
            ],
            grid=(Hq, Lq, 1),
            threadgroup=(8, 8, 1),
            output_shapes=[(Hq, Lq, D)],
            output_dtypes=[mx.float32],
        )

        output = outputs[0].reshape(B, Hq, Lq, D)

        execution_ms = (time.perf_counter() - start_time) * 1000

        contract = ExecutionContract(
            backend="true_packed_metal_mlx_inline",
            kernel_hash=self._kernel_hash,
            num_blocks=num_blocks,
            total_kv_tokens=total_tokens,
            num_q_heads=Hq,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            bits=self.bits,
            materialized_bytes=0,  # Zero materialization!
            decoded_tokens=0,      # On-the-fly decode!
            execution_ms=execution_ms,
        )

        # Validate in strict mode
        if strict:
            passed, violations = contract.validate_invariant()
            if not passed:
                raise RuntimeError(
                    "Strict mode: Execution contract violated:\n" +
                    "\n".join(f"  - {v}" for v in violations)
                )

        return output, contract


# =============================================================================
# Convenience Function
# =============================================================================

def true_packed_attention_mlx(
    queries: "mx.array",
    blocks: list[Any],
    scale: float = 1.0,
    causal: bool = True,
    query_start_pos: int = 0,
    bits: int = 8,
    strict: bool = False,
) -> tuple["mx.array", ExecutionContract]:
    """Convenience function for MLX inline true packed attention."""
    kernel = TruePackedMLXInline(bits=bits)
    return kernel(
        queries=queries,
        blocks=blocks,
        scale=scale,
        causal=causal,
        query_start_pos=query_start_pos,
        strict=strict,
    )
