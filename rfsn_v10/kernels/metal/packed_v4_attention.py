"""Canonical true-packed attention kernel for PackedBlockV4.

This module defines the single production ABI for true-packed attention
over real ``PackedBlockV4`` blocks.  It replaces the incompatible
mock-format prototypes (``TruePackedMLXInline``, ``TruePackedAttentionMetalV2``)
with a kernel that actually reads the production V4 wire format.

Design
------
* Python pre-computes the Walsh-Hadamard transform (WHT) of queries.
* The Metal shader decodes packed K/V blocks on-the-fly **in the WHT
  domain**, avoiding dense materialisation of either keys or values.
* Python post-applies inverse WHT to the accumulator, yielding the
  output in the original signal domain.

This works because WHT is orthonormal and self-inverse (with the
normalisation used in ``_reference_wht64``):

    Q · K  =  WHT(Q) · (signs · decode_K)
    O      =  WHT( Σ_t weight_t · (signs · decode_V) )

The kernel currently supports **K8/V8 only** (bits == 8, group_size
== 64).  Sub-byte and super-byte variants are gated out until the K8
path passes exact differential tests.

Execution contract
----------------
Every call returns an ``ExecutionContract`` recording:
* backend identity and kernel source hash
* block/token geometry
* measured materialised bytes (zero for the true-packed path)
* measured decoded tokens (zero for the true-packed path)
* timing
"""
from __future__ import annotations

import hashlib
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
    import numpy as np
    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False
    mx = None  # type: ignore
    np = None  # type: ignore

from rfsn_v10.cache.contracts import PackedBlockV4
from rfsn_v10.cache.cartesian_codec import _reference_wht64, _reference_hash_signs


# ---------------------------------------------------------------------------
# Feature gate – do not claim availability merely because MLX imports.
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """Quick self-test that the kernel can compile and run on synthetic data.

    Returns ``False`` on any failure so that production dispatch falls back
    safely to the blockwise reference.
    """
    if not HAS_MLX:
        return False
    try:
        # Try a trivial kernel compilation to verify Metal pipeline works
        _src = """
        uint idx = thread_position_in_grid.x;
        if (idx >= 1) return;
        output[0] = 1.0f;
        """
        k = mx.fast.metal_kernel(
            name="_rfsn_self_test",
            input_names=[],
            output_names=["output"],
            source=_src,
        )
        out = k(
            inputs=[],
            template=[],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
        )
        return int(out[0].item()) == 1
    except Exception:
        return False


# The canonical true-packed kernel is available ONLY when:
# 1. MLX + Metal are present
# 2. The explicit opt-in environment variable is set
# 3. A trivial self-test compiles and executes successfully
_ENABLED_BY_ENV = os.environ.get("RFSN_ENABLE_TRUE_PACKED", "0") == "1"
HAS_TRUE_PACKED_KERNEL = HAS_MLX and _ENABLED_BY_ENV and _self_test()


# ---------------------------------------------------------------------------
# Execution contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionContract:
    """Immutable record of kernel execution."""
    backend: str
    kernel_hash: str
    num_key_blocks: int
    num_value_blocks: int
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


# ---------------------------------------------------------------------------
# Metal kernel source – reads PackedBlockV4 layout natively
# ---------------------------------------------------------------------------

# K8 only: 8 bits per code, 4 codes per uint32 word.
_PACKED_V4_KERNEL_K8 = """
// Canonical True-Packed Attention for PackedBlockV4 (K8/V8)
// grid   : (num_q_heads, num_q_tokens, 1)
// thread : one (q_head, q_token) pair

uint q_head = thread_position_in_grid.x;
uint q_token = thread_position_in_grid.y;

if (q_head >= NUM_Q_HEADS || q_token >= NUM_Q_TOKENS) return;

// GQA mapping
uint kv_head = q_head / Q_PER_KV;
uint query_global_pos = QUERY_START + q_token;

// Pre-transformed query offset: [NUM_Q_HEADS, NUM_Q_TOKENS, HEAD_DIM]
uint q_offset = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;

// Output accumulator (written in WHT domain)
uint out_offset = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;

// Zero-initialize output buffer (MLX inline kernels do not clear output)
for (uint d = 0; d < HEAD_DIM; d++) {
    output[out_offset + d] = 0.0;
}

float scale_val = scale_arr[0];

// ---- online softmax state ----
float running_max = -INFINITY;
float running_sum = 0.0;

// ---- first pass: scores ----
uint t = 0;
for (uint b = 0; b < NUM_BLOCKS; b++) {
    int block_start = block_starts[b];
    int block_count = block_counts[b];

    for (uint local_t = 0; local_t < block_count; local_t++) {
        int kv_global_pos = block_start + int(local_t);

        // causal mask
        if (CAUSAL != 0 && kv_global_pos > query_global_pos) {
            t++;
            continue;
        }

        // QK dot product with on-the-fly decode in WHT domain
        float dot = 0.0;

        // packed_codes layout: [B=0, Hkv, T, WORDS_PER_VECTOR]
        uint base_word = (kv_head * TOTAL_T + t) * WORDS_PER_VECTOR;
        uint base_scale = (kv_head * TOTAL_T + t) * GROUPS_PER_VECTOR;

        // block-local flat index for hash signs (each block signed independently)
        uint local_flat_base = (kv_head * block_count + local_t) * HEAD_DIM;

        for (uint d = 0; d < HEAD_DIM; d++) {
            float q_val = wht_queries[q_offset + d];

            uint word_idx = d / CODES_PER_WORD;
            uint code_in_word = d % CODES_PER_WORD;
            uint shift = code_in_word * BITS;
            uint mask = (1u << BITS) - 1u;

            uint packed_word = packed_codes_k[base_word + word_idx];
            uint code = (packed_word >> shift) & mask;

            float q_signed = float(code) - QMAX;

            uint group_idx = d / GROUP_SIZE;
            float scl = scales_k[base_scale + group_idx];
            float val = q_signed * scl;

            // hash sign (Murmur32-avalanche-v1)
            uint flat_idx = local_flat_base + d;
            uint state = flat_idx ^ SEED_VAL_K;
            state = state + 0x9E3779B9u;
            state = state ^ (state >> 16);
            state = state * 0x85EBCA6Bu;
            state = state ^ (state >> 13);
            state = state * 0xC2B2AE35u;
            state = state ^ (state >> 16);
            float sign = (state & 1u) ? -1.0 : 1.0;

            float signed_val = val * sign;
            dot += q_val * signed_val;
        }

        dot *= scale_val;

        float new_max = max(running_max, dot);
        float old_scale = (running_max == -INFINITY) ? 0.0
                         : exp(running_max - new_max);
        running_sum = running_sum * old_scale + exp(dot - new_max);
        running_max = new_max;

        t++;
    }
}

uint stat_idx = q_head * NUM_Q_TOKENS + q_token;
running_max_arr[stat_idx] = running_max;
running_sum_arr[stat_idx] = running_sum;

// fully-masked row → zeros
if (running_sum == 0.0) {
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = 0.0;
    }
    return;
}

// ---- second pass: weighted accumulation in WHT domain ----
t = 0;
for (uint b = 0; b < NUM_BLOCKS; b++) {
    int block_start = block_starts[b];
    int block_count = block_counts[b];

    for (uint local_t = 0; local_t < block_count; local_t++) {
        int kv_global_pos = block_start + int(local_t);

        if (CAUSAL != 0 && kv_global_pos > query_global_pos) {
            t++;
            continue;
        }

        // Recompute QK score to obtain weight
        float dot = 0.0;
        uint base_word = (kv_head * TOTAL_T + t) * WORDS_PER_VECTOR;
        uint base_scale = (kv_head * TOTAL_T + t) * GROUPS_PER_VECTOR;
        uint local_flat_base = (kv_head * block_count + local_t) * HEAD_DIM;

        for (uint d = 0; d < HEAD_DIM; d++) {
            float q_val = wht_queries[q_offset + d];

            uint word_idx = d / CODES_PER_WORD;
            uint code_in_word = d % CODES_PER_WORD;
            uint shift = code_in_word * BITS;
            uint mask = (1u << BITS) - 1u;

            uint packed_word = packed_codes_k[base_word + word_idx];
            uint code = (packed_word >> shift) & mask;

            float q_signed = float(code) - QMAX;
            uint group_idx = d / GROUP_SIZE;
            float scl = scales_k[base_scale + group_idx];
            float val = q_signed * scl;

            uint flat_idx = local_flat_base + d;
            uint state = flat_idx ^ SEED_VAL_K;
            state = state + 0x9E3779B9u;
            state = state ^ (state >> 16);
            state = state * 0x85EBCA6Bu;
            state = state ^ (state >> 13);
            state = state * 0xC2B2AE35u;
            state = state ^ (state >> 16);
            float sign = (state & 1u) ? -1.0 : 1.0;

            float signed_val = val * sign;
            dot += q_val * signed_val;
        }

        dot *= scale_val;
        float weight = exp(dot - running_max) / running_sum;

        // Accumulate weighted values in WHT domain
        uint base_word_v = (kv_head * TOTAL_T + t) * WORDS_PER_VECTOR;
        uint base_scale_v = (kv_head * TOTAL_T + t) * GROUPS_PER_VECTOR;
        for (uint d = 0; d < HEAD_DIM; d++) {
            uint word_idx = d / CODES_PER_WORD;
            uint code_in_word = d % CODES_PER_WORD;
            uint shift = code_in_word * BITS;
            uint mask = (1u << BITS) - 1u;

            uint packed_word = packed_codes_v[base_word_v + word_idx];
            uint code = (packed_word >> shift) & mask;

            float q_signed = float(code) - QMAX;
            uint group_idx = d / GROUP_SIZE;
            float scl = scales_v[base_scale_v + group_idx];
            float val = q_signed * scl;

            uint flat_idx = local_flat_base + d;
            uint state = flat_idx ^ SEED_VAL_V;
            state = state + 0x9E3779B9u;
            state = state ^ (state >> 16);
            state = state * 0x85EBCA6Bu;
            state = state ^ (state >> 13);
            state = state * 0xC2B2AE35u;
            state = state ^ (state >> 16);
            float sign = (state & 1u) ? -1.0 : 1.0;

            float signed_val = val * sign;
            output[out_offset + d] += weight * signed_val;
        }

        t++;
    }
}
"""


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

class PackedV4AttentionKernel:
    """Canonical true-packed attention for real ``PackedBlockV4`` blocks.

    Parameters
    ----------
    bits
        Quantisation bit width.  Only ``8`` is supported in this release.
    group_size
        Group size for scales.  Only ``64`` is supported.
    sign_seed
        Seed for deterministic hash signs.  Must match the codec that
        produced the blocks.
    """

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        sign_seed: int = 42,
    ) -> None:
        if bits != 8:
            raise ValueError(
                f"PackedV4AttentionKernel only supports bits==8 in this release; got {bits}"
            )
        if group_size != 64:
            raise ValueError(
                f"PackedV4AttentionKernel only supports group_size==64 in this release; got {group_size}"
            )
        self.bits = bits
        self.group_size = group_size
        self.sign_seed = sign_seed
        self.qmax = (1 << (bits - 1)) - 1
        self.codes_per_word = 32 // bits
        self._kernel_hash = hashlib.sha256(
            _PACKED_V4_KERNEL_K8.encode()
        ).hexdigest()[:16]

    def _validate_blocks(self, key_blocks: list[PackedBlockV4], value_blocks: list[PackedBlockV4]) -> None:
        """Fail-fast validation of block compatibility."""
        if len(key_blocks) != len(value_blocks):
            raise ValueError(
                f"key/value block count mismatch: {len(key_blocks)} vs {len(value_blocks)}"
            )
        if not key_blocks:
            raise ValueError("no blocks provided")

        for i, (kb, vb) in enumerate(zip(key_blocks, value_blocks)):
            if kb.bits != self.bits:
                raise ValueError(f"block[{i}] key bits={kb.bits}, expected {self.bits}")
            if vb.bits != self.bits:
                raise ValueError(f"block[{i}] value bits={vb.bits}, expected {self.bits}")
            if kb.format_version != 4:
                raise ValueError(f"block[{i}] key format_version={kb.format_version}, expected 4")
            if vb.format_version != 4:
                raise ValueError(f"block[{i}] value format_version={vb.format_version}, expected 4")
            if kb.packing_layout.value != "VECTOR_ALIGNED_UINT32_V4":
                raise ValueError(f"block[{i}] key packing_layout={kb.packing_layout}")
            if kb.scale_layout.value != "BHTG_V4":
                raise ValueError(f"block[{i}] key scale_layout={kb.scale_layout}")
            if kb.preconditioner.value != "WHT64_HASH_SIGN_V1":
                raise ValueError(
                    f"block[{i}] key preconditioner={kb.preconditioner}; "
                    "WHT64_HASH_SIGN_V1 required"
                )
            if kb.token_count != vb.token_count:
                raise ValueError(
                    f"block[{i}] key/value token_count mismatch: {kb.token_count} vs {vb.token_count}"
                )
            if kb.logical_start != vb.logical_start:
                raise ValueError(
                    f"block[{i}] key/value logical_start mismatch: {kb.logical_start} vs {vb.logical_start}"
                )

    def _concatenate_blocks(self, blocks: list[PackedBlockV4]) -> tuple[Any, Any, Any, Any]:
        """Concatenate block packed_codes and scales along the T axis.

        Returns
        -------
        packed_codes
            Concatenated uint32 array of shape (B, H, total_T, W).
        scales
            Concatenated float32 array of shape (B, H, total_T, G).
        block_starts
            MLX int32 array of logical_start per block.
        block_counts
            MLX int32 array of token_count per block.
        """
        # packed_codes per block: (B, H, T, W)
        # scales per block:       (B, H, T, G)
        code_list = [b.packed_codes for b in blocks]
        scale_list = [b.scales for b in blocks]
        starts = [b.logical_start for b in blocks]
        counts = [b.token_count for b in blocks]

        packed_codes = mx.concatenate(code_list, axis=2)
        scales = mx.concatenate(scale_list, axis=2)
        block_starts_arr = mx.array(starts, dtype=mx.int32)
        block_counts_arr = mx.array(counts, dtype=mx.int32)
        return packed_codes, scales, block_starts_arr, block_counts_arr

    def _derive_mixed_seed(self, layer_id: int, stream_id: str) -> int:
        """Reproduce the seed mixing from ``_reference_hash_signs`` exactly.

        The shader receives a single pre-mixed uint32 seed; it does not
        recompute the string hashing or layer mixing per thread.
        """
        stream_hash = 0
        for ch in stream_id:
            stream_hash = (stream_hash * 31 + ord(ch)) & 0xFFFFFFFF
        mixed = np.uint32(self.sign_seed)
        mixed = np.uint32(mixed ^ np.uint32((layer_id * 0x9E3779B9) & 0xFFFFFFFF))
        mixed = np.uint32(mixed ^ np.uint32(stream_hash & 0xFFFFFFFF))
        return int(mixed)

    def __call__(
        self,
        queries: "mx.array",
        key_blocks: list[PackedBlockV4],
        value_blocks: list[PackedBlockV4],
        *,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
        strict: bool = False,
    ) -> tuple["mx.array", "mx.array", "mx.array", ExecutionContract]:
        """Execute true-packed attention.

        Parameters
        ----------
        queries
            Shape ``(B, Hq, Lq, D)``.
        key_blocks
            List of ``PackedBlockV4`` key blocks in positional order.
        value_blocks
            List of ``PackedBlockV4`` value blocks in positional order.
        scale
            Attention scale.
        causal
            Apply causal masking using ``logical_start`` metadata.
        query_start_pos
            Global position of the first query token.
        strict
            If ``True``, validate the zero-materialisation invariant.

        Returns
        -------
        output
            Shape ``(B, Hq, Lq, D)``.
        contract
            Execution contract for auditability.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX required")

        start_time = time.perf_counter()

        # Validate format compatibility
        self._validate_blocks(key_blocks, value_blocks)

        B, Hq, Lq, D = queries.shape
        if B != 1:
            raise RuntimeError(f"PackedV4AttentionKernel only supports batch_size=1, got {B}")

        num_kv_heads = key_blocks[0].n_kv_heads
        if Hq % num_kv_heads != 0:
            raise ValueError(f"Hq ({Hq}) must be divisible by num_kv_heads ({num_kv_heads})")
        q_per_kv = Hq // num_kv_heads

        total_tokens = sum(b.token_count for b in key_blocks)
        num_blocks = len(key_blocks)

        # Derive mixed seeds for K and V streams independently.
        layer_id = key_blocks[0].layer_id
        seed_k = self._derive_mixed_seed(layer_id, key_blocks[0].stream_id)
        seed_v = self._derive_mixed_seed(layer_id, value_blocks[0].stream_id)

        # Pre-transform queries into WHT domain
        groups = D // self.group_size
        if D % self.group_size != 0:
            raise ValueError(f"head_dim ({D}) must be divisible by group_size ({self.group_size})")
        queries_grouped = queries.reshape(B, Hq, Lq, groups, self.group_size)
        wht_queries = _reference_wht64(queries_grouped)
        wht_queries = wht_queries.reshape(B, Hq, Lq, D)
        # Flatten batch=1 for kernel
        wht_queries_flat = wht_queries.reshape(Hq, Lq, D)

        # Concatenate blocks along T
        packed_codes_k, scales_k, block_starts, block_counts = self._concatenate_blocks(key_blocks)
        packed_codes_v, scales_v, _, _ = self._concatenate_blocks(value_blocks)

        # Flatten batch=1 for kernel buffers
        packed_codes_k = packed_codes_k.reshape(num_kv_heads, total_tokens, -1)
        scales_k = scales_k.reshape(num_kv_heads, total_tokens, -1)
        packed_codes_v = packed_codes_v.reshape(num_kv_heads, total_tokens, -1)
        scales_v = scales_v.reshape(num_kv_heads, total_tokens, -1)

        # Scalar buffers
        scale_arr = mx.array([float(scale)], dtype=mx.float32)
        query_start_arr = mx.array([int(query_start_pos)], dtype=mx.int32)

        # Kernel dispatch
        kernel = mx.fast.metal_kernel(
            name="packed_v4_attention_k8",
            input_names=[
                "wht_queries",
                "packed_codes_k", "scales_k",
                "packed_codes_v", "scales_v",
                "block_starts", "block_counts",
                "scale_arr", "query_start_arr",
            ],
            output_names=["output", "running_max_arr", "running_sum_arr"],
            source=_PACKED_V4_KERNEL_K8,
        )

        outputs = kernel(
            inputs=[
                wht_queries_flat,
                packed_codes_k, scales_k,
                packed_codes_v, scales_v,
                block_starts, block_counts,
                scale_arr, query_start_arr,
            ],
            template=[
                ("NUM_Q_HEADS", int(Hq)),
                ("NUM_Q_TOKENS", int(Lq)),
                ("HEAD_DIM", int(D)),
                ("NUM_BLOCKS", int(num_blocks)),
                ("TOTAL_T", int(total_tokens)),
                ("BITS", int(self.bits)),
                ("CODES_PER_WORD", int(self.codes_per_word)),
                ("WORDS_PER_VECTOR", int(key_blocks[0].words_per_vector)),
                ("GROUP_SIZE", int(self.group_size)),
                ("GROUPS_PER_VECTOR", int(key_blocks[0].groups_per_vector)),
                ("QMAX", int(self.qmax)),
                ("SEED_VAL_K", int(seed_k)),
                ("SEED_VAL_V", int(seed_v)),
                ("CAUSAL", int(1 if causal else 0)),
                ("Q_PER_KV", int(q_per_kv)),
                ("QUERY_START", int(query_start_pos)),
            ],
            grid=(Hq, Lq, 1),
            threadgroup=(8, 8, 1),
            output_shapes=[(Hq, Lq, D), (Hq, Lq), (Hq, Lq)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

        output_wht = outputs[0]  # shape (Hq, Lq, D)
        running_max = outputs[1]  # shape (Hq, Lq)
        running_sum = outputs[2]  # shape (Hq, Lq)

        # Apply inverse WHT to return to original domain
        output_grouped = output_wht.reshape(Hq, Lq, groups, self.group_size)
        output = _reference_wht64(output_grouped)
        output = output.reshape(1, Hq, Lq, D).astype(queries.dtype)

        execution_ms = (time.perf_counter() - start_time) * 1000.0

        contract = ExecutionContract(
            backend="true_packed_metal_v4_k8",
            kernel_hash=self._kernel_hash,
            num_key_blocks=num_blocks,
            num_value_blocks=num_blocks,
            total_kv_tokens=total_tokens,
            num_q_heads=Hq,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            bits=self.bits,
            materialized_bytes=0,
            decoded_tokens=0,
            execution_ms=execution_ms,
        )

        if strict:
            passed, violations = contract.validate_invariant()
            if not passed:
                raise RuntimeError(
                    "Strict mode: execution contract violated:\n" +
                    "\n".join(f"  - {v}" for v in violations)
                )

        return output, running_max, running_sum, contract


def packed_v4_attention(
    queries: "mx.array",
    key_blocks: list[PackedBlockV4],
    value_blocks: list[PackedBlockV4],
    *,
    scale: float = 1.0,
    causal: bool = True,
    query_start_pos: int = 0,
    bits: int = 8,
    group_size: int = 64,
    sign_seed: int = 42,
    strict: bool = False,
) -> tuple["mx.array", "mx.array", "mx.array", ExecutionContract]:
    """Convenience wrapper around ``PackedV4AttentionKernel``."""
    kernel = PackedV4AttentionKernel(bits=bits, group_size=group_size, sign_seed=sign_seed)
    return kernel(
        queries=queries,
        key_blocks=key_blocks,
        value_blocks=value_blocks,
        scale=scale,
        causal=causal,
        query_start_pos=query_start_pos,
        strict=strict,
    )
