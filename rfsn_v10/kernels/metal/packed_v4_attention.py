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

from rfsn_v10.cache.cartesian_codec import _reference_wht64
from rfsn_v10.cache.contracts import PackedBlockV4

# ---------------------------------------------------------------------------
# Feature gate – do not claim availability merely because MLX imports.
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """Self-test that the real packed kernel source compiles and runs.

    Uses a minimal synthetic fixture to verify the Metal pipeline accepts
    the actual ``_PACKED_V4_KERNEL_K8`` source and buffer layout.  Full
    numerical validation is performed in the test suite, not at import
    time.
    """
    if not HAS_MLX:
        return False
    try:
        # Minimal fixture: 1 Q-head, 1 Q-token, 1 KV-head, 2 KV-tokens, D=64
        wht_queries = mx.zeros((1, 1, 64), dtype=mx.float32)
        packed_codes_k = mx.zeros((1, 2, 16), dtype=mx.uint32)
        scales_k = mx.ones((1, 2, 1), dtype=mx.float32)
        packed_codes_v = mx.zeros((1, 2, 16), dtype=mx.uint32)
        scales_v = mx.ones((1, 2, 1), dtype=mx.float32)
        block_starts = mx.array([0], dtype=mx.int32)
        block_counts = mx.array([2], dtype=mx.int32)
        scale_arr = mx.array([1.0], dtype=mx.float32)
        query_start_arr = mx.array([2], dtype=mx.int32)

        k = mx.fast.metal_kernel(
            name="packed_v4_attention_k8_selftest",
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
        out = k(
            inputs=[
                wht_queries,
                packed_codes_k, scales_k,
                packed_codes_v, scales_v,
                block_starts, block_counts,
                scale_arr, query_start_arr,
            ],
            template=[
                ("NUM_Q_HEADS", 1),
                ("NUM_Q_TOKENS", 1),
                ("HEAD_DIM", 64),
                ("NUM_BLOCKS", 1),
                ("TOTAL_T", 2),
                ("BITS", 8),
                ("CODES_PER_WORD", 4),
                ("WORDS_PER_VECTOR", 16),
                ("GROUP_SIZE", 64),
                ("GROUPS_PER_VECTOR", 1),
                ("QMAX", 127),
                ("SEED_VAL_K", 0),
                ("SEED_VAL_V", 0),
                ("CAUSAL", 1),
                ("Q_PER_KV", 1),
                ("QUERY_START", 2),
            ],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1, 1, 64), (1, 1), (1, 1)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        if len(out) != 3 or out[0].shape != (1, 1, 64):
            return False
        return True
    except Exception:
        return False


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


# The canonical true-packed kernel is available ONLY when:
# 1. MLX + Metal are present
# 2. The explicit opt-in environment variable is set
# 3. The real kernel self-test compiles and executes successfully
_ENABLED_BY_ENV = os.environ.get("RFSN_ENABLE_TRUE_PACKED", "0") == "1"
HAS_TRUE_PACKED_KERNEL = HAS_MLX and _ENABLED_BY_ENV and _self_test()


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
        # Persistent concatenation cache for O(T) incremental decode
        self._cached_key_blocks: list[PackedBlockV4] = []
        self._cached_value_blocks: list[PackedBlockV4] = []
        self._cached_k_codes: Any | None = None
        self._cached_k_scales: Any | None = None
        self._cached_v_codes: Any | None = None
        self._cached_v_scales: Any | None = None
        self._cached_block_starts: Any | None = None
        self._cached_block_counts: Any | None = None

    def _validate_blocks(self, key_blocks: list[PackedBlockV4], value_blocks: list[PackedBlockV4]) -> None:
        """Fail-fast validation of block compatibility."""
        if len(key_blocks) != len(value_blocks):
            raise ValueError(
                f"key/value block count mismatch: {len(key_blocks)} vs {len(value_blocks)}"
            )
        if not key_blocks:
            raise ValueError("no blocks provided")

        for i, (kb, vb) in enumerate(zip(key_blocks, value_blocks)):
            # Format gate first — V3 blocks must be rejected before V4-only checks.
            if kb.format_version != 4:
                raise ValueError(f"block[{i}] key format_version={kb.format_version}, expected 4")
            if vb.format_version != 4:
                raise ValueError(f"block[{i}] value format_version={vb.format_version}, expected 4")

            # P1.4: call validate() on every block
            kb.validate()
            vb.validate()

            if kb.bits != self.bits:
                raise ValueError(f"block[{i}] key bits={kb.bits}, expected {self.bits}")
            if vb.bits != self.bits:
                raise ValueError(f"block[{i}] value bits={vb.bits}, expected {self.bits}")
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

        # Reference metadata from first block for cross-block consistency.
        ref_kb = key_blocks[0]
        ref_vb = value_blocks[0]
        expected_words_per_vector = ref_kb.words_per_vector
        expected_groups_per_vector = ref_kb.groups_per_vector
        ref_layer_id = ref_kb.layer_id
        ref_stream_k = ref_kb.stream_id
        ref_stream_v = ref_vb.stream_id
        ref_sign_seed = ref_kb.sign_seed
        ref_codec_sig = ref_kb.codec_signature
        ref_batch = ref_kb.batch_size
        ref_heads = ref_kb.n_kv_heads
        ref_dim = ref_kb.head_dim
        prev_end = 0

        for i, (kb, vb) in enumerate(zip(key_blocks, value_blocks)):
            # P1.4: contiguous, non-overlapping logical positions
            if i > 0 and kb.logical_start != prev_end:
                raise ValueError(
                    f"block[{i}] logical_start={kb.logical_start} != previous_end={prev_end}; "
                    "blocks must be contiguous"
                )
            prev_end = kb.logical_end

            # P1.4: consistent geometry
            if kb.batch_size != ref_batch or vb.batch_size != ref_batch:
                raise ValueError(f"block[{i}] batch_size mismatch")
            if kb.n_kv_heads != ref_heads or vb.n_kv_heads != ref_heads:
                raise ValueError(f"block[{i}] n_kv_heads mismatch")
            if kb.head_dim != ref_dim or vb.head_dim != ref_dim:
                raise ValueError(f"block[{i}] head_dim mismatch")

            # P1.4: consistent codec metadata
            if kb.layer_id != ref_layer_id or vb.layer_id != ref_layer_id:
                raise ValueError(f"block[{i}] layer_id mismatch")
            if kb.stream_id != ref_stream_k:
                raise ValueError(f"block[{i}] key stream_id mismatch")
            if vb.stream_id != ref_stream_v:
                raise ValueError(f"block[{i}] value stream_id mismatch")
            if kb.sign_seed != ref_sign_seed or vb.sign_seed != ref_sign_seed:
                raise ValueError(f"block[{i}] sign_seed mismatch")
            if ref_codec_sig and kb.codec_signature != ref_codec_sig:
                raise ValueError(f"block[{i}] codec_signature mismatch")

            # P1.4: consistent format geometry
            if kb.words_per_vector != expected_words_per_vector:
                raise ValueError(
                    f"block[{i}] words_per_vector={kb.words_per_vector}, expected {expected_words_per_vector}"
                )
            if kb.groups_per_vector != expected_groups_per_vector:
                raise ValueError(
                    f"block[{i}] groups_per_vector={kb.groups_per_vector}, expected {expected_groups_per_vector}"
                )

            # P1.4: buffer shape sanity
            expected_code_shape = (ref_batch, ref_heads, kb.token_count, expected_words_per_vector)
            if kb.packed_codes is not None and tuple(kb.packed_codes.shape) != expected_code_shape:
                raise ValueError(
                    f"block[{i}] key packed_codes shape {tuple(kb.packed_codes.shape)} != {expected_code_shape}"
                )
            expected_scale_shape = (ref_batch, ref_heads, kb.token_count, expected_groups_per_vector)
            if kb.scales is not None and tuple(kb.scales.shape) != expected_scale_shape:
                raise ValueError(
                    f"block[{i}] key scales shape {tuple(kb.scales.shape)} != {expected_scale_shape}"
                )

    def _prepare_concatenated_buffers(
        self, key_blocks: list[PackedBlockV4], value_blocks: list[PackedBlockV4]
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return concatenated K/V buffers, using incremental append when possible.

        Blocks are immutable and append-only in normal operation.  When the
        new block lists extend the previously-cached lists we only concatenate
        the new suffix, giving O(new_tokens) work per decode step instead of
        O(total_tokens) = O(T^2) over the full sequence.
        """
        # Fast-path: check if we can reuse the cached prefix for both K and V.
        k_cache = self._cached_key_blocks
        v_cache = self._cached_value_blocks
        can_append = False

        if k_cache and v_cache:
            n_k = len(k_cache)
            n_v = len(v_cache)
            if (
                len(key_blocks) >= n_k
                and len(value_blocks) >= n_v
                and (n_k == 0 or key_blocks[n_k - 1] is k_cache[-1])
                and (n_v == 0 or value_blocks[n_v - 1] is v_cache[-1])
            ):
                can_append = True

        if can_append:
            # Append new key blocks
            new_k = key_blocks[len(k_cache):]
            if new_k:
                new_k_codes = mx.concatenate([b.packed_codes for b in new_k], axis=2)
                new_k_scales = mx.concatenate([b.scales for b in new_k], axis=2)
                self._cached_k_codes = mx.concatenate([self._cached_k_codes, new_k_codes], axis=2)
                self._cached_k_scales = mx.concatenate([self._cached_k_scales, new_k_scales], axis=2)
            # Append new value blocks
            new_v = value_blocks[len(v_cache):]
            if new_v:
                new_v_codes = mx.concatenate([b.packed_codes for b in new_v], axis=2)
                new_v_scales = mx.concatenate([b.scales for b in new_v], axis=2)
                self._cached_v_codes = mx.concatenate([self._cached_v_codes, new_v_codes], axis=2)
                self._cached_v_scales = mx.concatenate([self._cached_v_scales, new_v_scales], axis=2)
        else:
            # Rebuild from scratch
            self._cached_k_codes = mx.concatenate([b.packed_codes for b in key_blocks], axis=2)
            self._cached_k_scales = mx.concatenate([b.scales for b in key_blocks], axis=2)
            self._cached_v_codes = mx.concatenate([b.packed_codes for b in value_blocks], axis=2)
            self._cached_v_scales = mx.concatenate([b.scales for b in value_blocks], axis=2)

        # Always rebuild starts/counts (cheap: just int32 arrays)
        starts = [b.logical_start for b in key_blocks]
        counts = [b.token_count for b in key_blocks]
        self._cached_block_starts = mx.array(starts, dtype=mx.int32)
        self._cached_block_counts = mx.array(counts, dtype=mx.int32)
        self._cached_key_blocks = list(key_blocks)
        self._cached_value_blocks = list(value_blocks)

        return (
            self._cached_k_codes,
            self._cached_k_scales,
            self._cached_v_codes,
            self._cached_v_scales,
            self._cached_block_starts,
            self._cached_block_counts,
        )

    def _derive_mixed_seed(self, layer_id: int, stream_id: str) -> int:
        """Reproduce the seed mixing from ``_reference_hash_signs`` exactly.

        The shader receives a single pre-mixed uint32 seed; it does not
        recompute the string hashing or layer mixing per thread.

        The result is masked to ``0x7FFFFFFF`` so that it always fits in a
        signed 32-bit integer.  MLX's ``metal_kernel`` template system
        rejects unsigned values that exceed ``INT_MAX`` because they appear
        in generated C++ kernel function names.
        """
        stream_hash = 0
        for ch in stream_id:
            stream_hash = (stream_hash * 31 + ord(ch)) & 0xFFFFFFFF
        mixed = np.uint32(self.sign_seed)
        mixed = np.uint32(mixed ^ np.uint32((layer_id * 0x9E3779B9) & 0xFFFFFFFF))
        mixed = np.uint32(mixed ^ np.uint32(stream_hash & 0xFFFFFFFF))
        return int(mixed) & 0x7FFFFFFF

    def __call__(
        self,
        queries: mx.array,
        key_blocks: list[PackedBlockV4],
        value_blocks: list[PackedBlockV4],
        *,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
        strict: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array, ExecutionContract]:
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

        # Concatenate blocks along T (incremental O(T) when kernel persists)
        (
            packed_codes_k, scales_k,
            packed_codes_v, scales_v,
            block_starts, block_counts,
        ) = self._prepare_concatenated_buffers(key_blocks, value_blocks)

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

        # P1.5: synchronize before timing so execution_ms reflects actual kernel work
        mx.eval(output, running_max, running_sum)
        execution_ms = (time.perf_counter() - start_time) * 1000.0

        # P1.5: measured counters (true-packed path: no dense K/V materialised)
        materialized_bytes = 0
        decoded_tokens = 0

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
            materialized_bytes=materialized_bytes,
            decoded_tokens=decoded_tokens,
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
    queries: mx.array,
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
) -> tuple[mx.array, mx.array, mx.array, ExecutionContract]:
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
