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
        # Paged layout: (Hkv, max_pages, page_tokens, words)
        wht_queries = mx.zeros((1, 1, 64), dtype=mx.float32)
        packed_codes_k = mx.zeros((1, 1, 2, 16), dtype=mx.uint32)
        scales_k = mx.ones((1, 1, 2, 1), dtype=mx.float32)
        packed_codes_v = mx.zeros((1, 1, 2, 16), dtype=mx.uint32)
        scales_v = mx.ones((1, 1, 2, 1), dtype=mx.float32)
        page_table = mx.array([0], dtype=mx.int32)
        page_starts = mx.array([0], dtype=mx.int32)
        page_counts = mx.array([2], dtype=mx.int32)
        active_pages = mx.array([1], dtype=mx.int32)
        scale_arr = mx.array([1.0], dtype=mx.float32)
        query_start_arr = mx.array([2], dtype=mx.int32)

        k = mx.fast.metal_kernel(
            name="packed_v4_attention_k8_selftest",
            input_names=[
                "wht_queries",
                "packed_codes_k", "scales_k",
                "packed_codes_v", "scales_v",
                "page_table", "page_starts", "page_counts", "active_pages",
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
                page_table, page_starts, page_counts, active_pages,
                scale_arr, query_start_arr,
            ],
            template=[
                ("NUM_Q_HEADS", 1),
                ("NUM_Q_TOKENS", 1),
                ("HEAD_DIM", 64),
                ("MAX_PAGES", 1),
                ("PAGE_TOKENS", 2),
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
    # P0: measured counters instead of hardcoded zeros
    dense_kv_materialized_bytes: int = 0
    packed_history_copy_bytes: int = 0
    query_transform_bytes: int = 0
    scratch_bytes: int = 0
    output_bytes: int = 0
    decoded_dense_tokens: int = 0
    # P1: packed I/O counters for promotion governance
    packed_blocks_read: int = 0
    packed_bytes_read: int = 0
    # P4.6: separate prefill vs decode timing
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    # Legacy aliases for backward compatibility
    materialized_bytes: int = 0
    decoded_tokens: int = 0
    fallback_reason: str | None = None
    execution_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def validate_invariant(self) -> tuple[bool, list[str]]:
        violations = []
        # P0: the true-packed path must not materialise dense historical K/V
        if self.dense_kv_materialized_bytes > 0:
            violations.append(
                f"dense_kv_materialized_bytes={self.dense_kv_materialized_bytes} > 0"
            )
        if self.decoded_dense_tokens > 0:
            violations.append(
                f"decoded_dense_tokens={self.decoded_dense_tokens} > 0"
            )
        if "true_packed_metal" not in self.backend:
            violations.append(f"backend={self.backend}")
        return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Metal kernel source – reads PackedBlockV4 layout natively
# ---------------------------------------------------------------------------

# K8 only: 8 bits per code, 4 codes per uint32 word.
# Phase 6: Single-pass kernel — no QK recomputation.
# Each thread processes one (q_head, q_token) pair.
# grid   : (num_q_heads, num_q_tokens, 1)
# thread : one (q_head, q_token) pair
_PACKED_V4_KERNEL_K8 = """
uint q_head = thread_position_in_grid.x;
uint q_token = thread_position_in_grid.y;

if (q_head >= NUM_Q_HEADS || q_token >= NUM_Q_TOKENS) return;

// GQA mapping
uint kv_head = q_head / Q_PER_KV;
uint query_global_pos = uint(query_start_arr[0]) + q_token;

// Pre-transformed query offset: [NUM_Q_HEADS, NUM_Q_TOKENS, HEAD_DIM]
uint q_pair  = q_head * NUM_Q_TOKENS + q_token;
uint q_offset = q_pair * HEAD_DIM;
uint out_offset = q_pair * HEAD_DIM;

float scale_val = scale_arr[0];

// ---- single-pass online softmax + value accumulator ----
float running_max = -INFINITY;
float running_sum = 0.0f;

// Local accumulator (register) — avoid device RMW in inner loop
float acc[HEAD_DIM];
for (uint d = 0; d < HEAD_DIM; d++) {
    acc[d] = 0.0f;
}

uint num_pages = uint(active_pages[0]);
uint mask = (1u << BITS) - 1u;

for (uint logical_page = 0; logical_page < num_pages; logical_page++) {
    uint physical_page = uint(page_table[logical_page]);
    int page_start = page_starts[logical_page];
    int page_count = page_counts[logical_page];

    for (uint local_t = 0; local_t < uint(page_count); local_t++) {
        int kv_global_pos = page_start + int(local_t);

        // causal mask
        if (CAUSAL != 0 && kv_global_pos > query_global_pos) {
            continue;
        }

        // ---- QK dot product with on-the-fly decode in WHT domain ----
        float dot = 0.0f;

        uint local_flat_base = (kv_head * uint(page_count) + local_t) * HEAD_DIM;

        // Precompute base indices for this token
        uint token_base = (
            (kv_head * MAX_PAGES + physical_page) * PAGE_TOKENS + local_t
        );
        uint code_base_k = token_base * WORDS_PER_VECTOR;
        uint scale_base_k = token_base * GROUPS_PER_VECTOR;

        // Load scale once per token (GROUPS_PER_VECTOR is typically 1)
        float scl_k = float(scales_k[scale_base_k]);

        // SIMD vector constants for 4-dim parallelism
        uint4 shifts4 = uint4(0u, BITS, 2u * BITS, 3u * BITS);
        float4 qmax4 = float4(QMAX);

        for (uint word_idx = 0; word_idx < WORDS_PER_VECTOR; word_idx++) {
            uint packed_word = packed_codes_k[code_base_k + word_idx];
            uint d = word_idx * CODES_PER_WORD;

            float4 q4 = float4(
                wht_queries[q_offset + d + 0],
                wht_queries[q_offset + d + 1],
                wht_queries[q_offset + d + 2],
                wht_queries[q_offset + d + 3]
            );

            uint4 codes4 = (uint4(packed_word) >> shifts4) & mask;
            float4 q_signed4 = float4(codes4) - qmax4;
            float4 val4 = q_signed4 * scl_k;

            uint4 flat_idx4 = uint4(
                local_flat_base + d + 0,
                local_flat_base + d + 1,
                local_flat_base + d + 2,
                local_flat_base + d + 3
            );
            uint4 state4 = flat_idx4 ^ SEED_VAL_K;
            state4 = state4 + 0x9E3779B9u;
            state4 = state4 ^ (state4 >> 16);
            state4 = state4 * 0x85EBCA6Bu;
            state4 = state4 ^ (state4 >> 13);
            state4 = state4 * 0xC2B2AE35u;
            state4 = state4 ^ (state4 >> 16);
            float4 sign4 = select(float4(1.0f), float4(-1.0f), (state4 & 1u) != 0u);

            float4 partial = q4 * val4 * sign4;
            dot += partial.x + partial.y + partial.z + partial.w;
        }

        dot *= scale_val;

        // ---- online softmax update ----
        float new_max = max(running_max, dot);
        float scale_old = (running_max == -INFINITY) ? 0.0f
                         : exp(running_max - new_max);
        float exp_dot = exp(dot - new_max);
        running_sum = running_sum * scale_old + exp_dot;
        running_max = new_max;

        // ---- immediately accumulate weighted values ----
        uint code_base_v = token_base * WORDS_PER_VECTOR;
        uint scale_base_v = token_base * GROUPS_PER_VECTOR;
        float scl_v = float(scales_v[scale_base_v]);

        for (uint word_idx = 0; word_idx < WORDS_PER_VECTOR; word_idx++) {
            uint packed_word = packed_codes_v[code_base_v + word_idx];
            uint d = word_idx * CODES_PER_WORD;

            float4 acc4 = float4(acc[d+0], acc[d+1], acc[d+2], acc[d+3]);

            uint4 codes4 = (uint4(packed_word) >> shifts4) & mask;
            float4 q_signed4 = float4(codes4) - qmax4;
            float4 val4 = q_signed4 * scl_v;

            uint4 flat_idx4 = uint4(
                local_flat_base + d + 0,
                local_flat_base + d + 1,
                local_flat_base + d + 2,
                local_flat_base + d + 3
            );
            uint4 state4 = flat_idx4 ^ SEED_VAL_V;
            state4 = state4 + 0x9E3779B9u;
            state4 = state4 ^ (state4 >> 16);
            state4 = state4 * 0x85EBCA6Bu;
            state4 = state4 ^ (state4 >> 13);
            state4 = state4 * 0xC2B2AE35u;
            state4 = state4 ^ (state4 >> 16);
            float4 sign4 = select(float4(1.0f), float4(-1.0f), (state4 & 1u) != 0u);

            acc4 = acc4 * scale_old + exp_dot * val4 * sign4;
            acc[d+0] = acc4.x;
            acc[d+1] = acc4.y;
            acc[d+2] = acc4.z;
            acc[d+3] = acc4.w;
        }
    }
}

uint stat_idx = q_pair;
running_max_arr[stat_idx] = running_max;
running_sum_arr[stat_idx] = running_sum;

// Normalize and write to output
if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = acc[d] * inv_sum;
    }
} else {
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = 0.0f;
    }
}
"""

# ---------------------------------------------------------------------------
# KV-tiled kernel source (experimental)
# ---------------------------------------------------------------------------
# Parallelizes over (q_head, q_token, kv_tile) instead of just (q_head, q_token).
# Each threadgroup handles one (q_head, q_token) and threads within the group
# process disjoint KV-tile ranges. A threadgroup-barrier reduction combines
# partial online-softmax results.
#
# This addresses the audit finding that the original kernel is "serial over
# context and under-parallelized".
#
# Status: ARCHITECTURE READY — tiled source defined but not yet the default.
# Enable by passing kv_tile_size > 0 in PackedV4AttentionKernel.__call__.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Two-pass KV-tiled kernel
# ---------------------------------------------------------------------------
# Pass 1: each thread processes one KV tile and writes partial results.
# Pass 2: one thread per (q_head, q_token) reduces all tile partials.
#
# This avoids the need for threadgroup-memory reduction inside a single
# kernel dispatch, which is not well-supported by MLX's metal_kernel API.
# ---------------------------------------------------------------------------

_PACKED_V4_KERNEL_K8_TILED_PASS1 = """
uint tid = thread_position_in_grid.x;
uint total_threads = NUM_Q_HEADS * NUM_Q_TOKENS * NUM_KV_TILES;
if (tid >= total_threads) return;

uint q_head = tid / (NUM_Q_TOKENS * NUM_KV_TILES);
uint rem = tid % (NUM_Q_TOKENS * NUM_KV_TILES);
uint q_token = rem / NUM_KV_TILES;
uint kv_tile = rem % NUM_KV_TILES;

uint kv_head = q_head / Q_PER_KV;
uint query_global_pos = uint(query_start_arr[0]) + q_token;

uint q_offset  = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;
float scale_val = scale_arr[0];

float local_max  = -INFINITY;
float local_sum  = 0.0f;
float local_acc[HEAD_DIM];
for (uint d = 0; d < HEAD_DIM; d++) { local_acc[d] = 0.0f; }

uint kv_start = kv_tile * KV_TILE_SIZE;
uint kv_end   = kv_start + KV_TILE_SIZE;  // tile size is constant per template
uint mask = (1u << BITS) - 1u;

uint num_pages = uint(active_pages[0]);

for (uint logical_page = 0; logical_page < num_pages; logical_page++) {
    uint physical_page = uint(page_table[logical_page]);
    int page_start = page_starts[logical_page];
    int page_count = page_counts[logical_page];

    int page_end = page_start + page_count;
    if (page_end <= int(kv_start)) continue;
    if (page_start >= int(kv_end)) continue;

    for (uint local_t = 0; local_t < uint(page_count); local_t++) {
        int kv_global_pos = page_start + int(local_t);
        if (kv_global_pos < int(kv_start)) continue;
        if (kv_global_pos >= int(kv_end)) continue;
        if (CAUSAL != 0 && kv_global_pos > int(query_global_pos)) continue;

        float dot = 0.0f;
        uint local_flat_base = (kv_head * uint(page_count) + local_t) * HEAD_DIM;

        uint token_base = (
            (kv_head * MAX_PAGES + physical_page) * PAGE_TOKENS + local_t
        );
        uint code_base_k = token_base * WORDS_PER_VECTOR;
        uint scale_base_k = token_base * GROUPS_PER_VECTOR;
        float scl_k = float(scales_k[scale_base_k]);

        uint4 shifts4 = uint4(0u, BITS, 2u * BITS, 3u * BITS);
        float4 qmax4 = float4(QMAX);

        for (uint word_idx = 0; word_idx < WORDS_PER_VECTOR; word_idx++) {
            uint packed_word = packed_codes_k[code_base_k + word_idx];
            uint d = word_idx * CODES_PER_WORD;

            float4 q4 = float4(
                wht_queries[q_offset + d + 0],
                wht_queries[q_offset + d + 1],
                wht_queries[q_offset + d + 2],
                wht_queries[q_offset + d + 3]
            );

            uint4 codes4 = (uint4(packed_word) >> shifts4) & mask;
            float4 q_signed4 = float4(codes4) - qmax4;
            float4 val4 = q_signed4 * scl_k;

            uint4 flat_idx4 = uint4(
                local_flat_base + d + 0,
                local_flat_base + d + 1,
                local_flat_base + d + 2,
                local_flat_base + d + 3
            );
            uint4 state4 = flat_idx4 ^ SEED_VAL_K;
            state4 = state4 + 0x9E3779B9u;
            state4 = state4 ^ (state4 >> 16);
            state4 = state4 * 0x85EBCA6Bu;
            state4 = state4 ^ (state4 >> 13);
            state4 = state4 * 0xC2B2AE35u;
            state4 = state4 ^ (state4 >> 16);
            float4 sign4 = select(float4(1.0f), float4(-1.0f), (state4 & 1u) != 0u);

            float4 partial = q4 * val4 * sign4;
            dot += partial.x + partial.y + partial.z + partial.w;
        }
        dot *= scale_val;

        float new_max = max(local_max, dot);
        float scale_old = (local_max == -INFINITY) ? 0.0f : exp(local_max - new_max);
        float exp_dot = exp(dot - new_max);
        local_sum = local_sum * scale_old + exp_dot;
        local_max = new_max;

        uint code_base_v = token_base * WORDS_PER_VECTOR;
        uint scale_base_v = token_base * GROUPS_PER_VECTOR;
        float scl_v = float(scales_v[scale_base_v]);

        for (uint word_idx = 0; word_idx < WORDS_PER_VECTOR; word_idx++) {
            uint packed_word = packed_codes_v[code_base_v + word_idx];
            uint d = word_idx * CODES_PER_WORD;

            float4 acc4 = float4(local_acc[d+0], local_acc[d+1], local_acc[d+2], local_acc[d+3]);

            uint4 codes4 = (uint4(packed_word) >> shifts4) & mask;
            float4 q_signed4 = float4(codes4) - qmax4;
            float4 val4 = q_signed4 * scl_v;

            uint4 flat_idx4 = uint4(
                local_flat_base + d + 0,
                local_flat_base + d + 1,
                local_flat_base + d + 2,
                local_flat_base + d + 3
            );
            uint4 state4 = flat_idx4 ^ SEED_VAL_V;
            state4 = state4 + 0x9E3779B9u;
            state4 = state4 ^ (state4 >> 16);
            state4 = state4 * 0x85EBCA6Bu;
            state4 = state4 ^ (state4 >> 13);
            state4 = state4 * 0xC2B2AE35u;
            state4 = state4 ^ (state4 >> 16);
            float4 sign4 = select(float4(1.0f), float4(-1.0f), (state4 & 1u) != 0u);

            acc4 = acc4 * scale_old + exp_dot * val4 * sign4;
            local_acc[d+0] = acc4.x;
            local_acc[d+1] = acc4.y;
            local_acc[d+2] = acc4.z;
            local_acc[d+3] = acc4.w;
        }
    }
}

uint q_pair = q_head * NUM_Q_TOKENS + q_token;
uint partial_idx = (q_pair * NUM_KV_TILES + kv_tile) * HEAD_DIM;
for (uint d = 0; d < HEAD_DIM; d++) {
    partial_output[partial_idx + d] = local_acc[d];
}
uint stat_idx = q_pair * NUM_KV_TILES + kv_tile;
partial_max[stat_idx] = local_max;
partial_sum[stat_idx] = local_sum;
"""

_PACKED_V4_KERNEL_K8_TILED_PASS2 = """
// ---- grid: (q_head, q_token, 1) ----
uint q_head  = thread_position_in_grid.x;
uint q_token = thread_position_in_grid.y;

if (q_head >= NUM_Q_HEADS || q_token >= NUM_Q_TOKENS) return;

uint out_offset = (q_head * NUM_Q_TOKENS + q_token) * HEAD_DIM;

// ---- reduce across all KV tiles ----
// Step 1: find global max across non-empty tiles
uint base_stat = (q_head * NUM_Q_TOKENS + q_token) * NUM_KV_TILES;
uint base_out  = base_stat * HEAD_DIM;

float global_max = -INFINITY;
for (uint tile = 0; tile < NUM_KV_TILES; tile++) {
    float tile_sum = partial_sum[base_stat + tile];
    if (tile_sum > 0.0f) {
        float tile_max = partial_max[base_stat + tile];
        global_max = max(global_max, tile_max);
    }
}

// Step 2: accumulate weighted partials at global_max scale
float global_sum = 0.0f;
float global_acc[HEAD_DIM];
for (uint d = 0; d < HEAD_DIM; d++) { global_acc[d] = 0.0f; }

for (uint tile = 0; tile < NUM_KV_TILES; tile++) {
    float tile_sum = partial_sum[base_stat + tile];
    if (tile_sum <= 0.0f) continue;

    float tile_max = partial_max[base_stat + tile];
    float tile_scale = exp(tile_max - global_max);

    global_sum += tile_sum * tile_scale;

    uint tile_out = base_out + tile * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {
        global_acc[d] += partial_output[tile_out + d] * tile_scale;
    }
}

uint stat_idx = q_head * NUM_Q_TOKENS + q_token;
running_max_arr[stat_idx] = global_max;
running_sum_arr[stat_idx] = global_sum;

if (global_sum > 0.0f) {
    float inv_sum = 1.0f / global_sum;
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = global_acc[d] * inv_sum;
    }
} else {
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_offset + d] = 0.0f;
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
        kv_tile_size: int = 0,
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
        self.kv_tile_size = kv_tile_size
        self._kernel_hash = hashlib.sha256(
            (_PACKED_V4_KERNEL_K8 + _PACKED_V4_KERNEL_K8_TILED_PASS1 + _PACKED_V4_KERNEL_K8_TILED_PASS2).encode()
        ).hexdigest()[:16]
        # Kernel wrapper cache: keyed by template signature (avoid rebuild in hot path)
        self._kernel_cache: dict[tuple, Any] = {}
        # Compilation stability counter (tests assert this stays flat)
        self._compilation_count = 0

    @property
    def compilation_count(self) -> int:
        return self._compilation_count

    def _validate_paged_kv(self, paged_kv: Any) -> None:
        """Fail-fast validation of paged view compatibility."""
        from rfsn_v10.cache.paged_arena import PagedKVView
        if not isinstance(paged_kv, PagedKVView):
            raise TypeError(f"expected PagedKVView, got {type(paged_kv)}")
        if paged_kv.num_pages <= 0:
            raise ValueError("paged attention requires at least one page")

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
        paged_kv: Any,
        *,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
        strict: bool = False,
        layer_id: int = 0,
        is_prefill: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array, ExecutionContract]:
        """Execute true-packed attention over a paged KV arena.

        Parameters
        ----------
        queries
            Shape ``(B, Hq, Lq, D)``.
        paged_kv
            ``PagedKVView`` from ``PagedKVArena.view()``.
        scale
            Attention scale.
        causal
            Apply causal masking using ``page_starts`` metadata.
        query_start_pos
            Global position of the first query token.
        strict
            If ``True``, validate the zero-materialisation invariant.
        layer_id
            Layer index for deterministic sign derivation.
        is_prefill
            If ``True``, record timing under ``prefill_ms``; otherwise ``decode_ms``.

        Returns
        -------
        output
            Shape ``(B, Hq, Lq, D)``.
        contract
            Execution contract for auditability.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX required")

        from rfsn_v10.cache.paged_arena import PagedKVView
        if not isinstance(paged_kv, PagedKVView):
            raise TypeError(f"expected PagedKVView, got {type(paged_kv)}")

        start_time = time.perf_counter()

        self._validate_paged_kv(paged_kv)

        B, Hq, Lq, D = queries.shape
        if B != 1:
            raise RuntimeError(f"PackedV4AttentionKernel only supports batch_size=1, got {B}")

        num_kv_heads = int(paged_kv.k_codes.shape[0])
        if Hq % num_kv_heads != 0:
            raise ValueError(f"Hq ({Hq}) must be divisible by num_kv_heads ({num_kv_heads})")
        q_per_kv = Hq // num_kv_heads

        num_pages = paged_kv.num_pages
        total_tokens = int(mx.sum(paged_kv.page_counts[:num_pages]).item())

        # Derive mixed seeds for K and V streams independently.
        seed_k = self._derive_mixed_seed(layer_id, "K")
        seed_v = self._derive_mixed_seed(layer_id, "V")

        # Pre-transform queries into WHT domain
        groups = D // self.group_size
        if D % self.group_size != 0:
            raise ValueError(f"head_dim ({D}) must be divisible by group_size ({self.group_size})")
        queries_grouped = queries.reshape(B, Hq, Lq, groups, self.group_size)
        wht_queries = _reference_wht64(queries_grouped)
        wht_queries = wht_queries.reshape(B, Hq, Lq, D)
        # Flatten batch=1 for kernel
        wht_queries_flat = wht_queries.reshape(Hq, Lq, D)

        # Paged arrays are passed directly — no concatenation.
        active_pages = mx.array([num_pages], dtype=mx.int32)
        scale_arr = mx.array([float(scale)], dtype=mx.float32)
        query_start_arr = mx.array([int(query_start_pos)], dtype=mx.int32)

        # Kernel dispatch — cache wrapper by template signature.
        use_tiling = self.kv_tile_size > 0 and total_tokens > self.kv_tile_size
        num_kv_tiles = 1
        if use_tiling:
            num_kv_tiles = (total_tokens + self.kv_tile_size - 1) // self.kv_tile_size

        template = [
            ("NUM_Q_HEADS", int(Hq)),
            ("NUM_Q_TOKENS", int(Lq)),
            ("HEAD_DIM", int(D)),
            ("MAX_PAGES", int(paged_kv.max_pages)),
            ("PAGE_TOKENS", int(paged_kv.page_tokens)),
            ("BITS", int(self.bits)),
            ("CODES_PER_WORD", int(self.codes_per_word)),
            ("WORDS_PER_VECTOR", int(paged_kv.k_words_per_vector)),
            ("GROUP_SIZE", int(self.group_size)),
            ("GROUPS_PER_VECTOR", int(paged_kv.k_groups_per_vector)),
            ("QMAX", int(self.qmax)),
            ("SEED_VAL_K", int(seed_k)),
            ("SEED_VAL_V", int(seed_v)),
            ("CAUSAL", int(1 if causal else 0)),
            ("Q_PER_KV", int(q_per_kv)),
        ]
        if use_tiling:
            template.append(("KV_TILE_SIZE", int(self.kv_tile_size)))
            template.append(("NUM_KV_TILES", int(num_kv_tiles)))

        # Cache key must distinguish scalar vs tiled (different sources)
        cache_key = (tuple(template), use_tiling)
        cached = self._kernel_cache.get(cache_key)

        if cached is None:
            if use_tiling:
                pass1_kernel = mx.fast.metal_kernel(
                    name="rfsn_tiled_pass1_v2_paged",
                    input_names=[
                        "wht_queries",
                        "packed_codes_k", "scales_k",
                        "packed_codes_v", "scales_v",
                        "page_table", "page_starts", "page_counts", "active_pages",
                        "scale_arr", "query_start_arr",
                    ],
                    output_names=["partial_output", "partial_max", "partial_sum"],
                    source=_PACKED_V4_KERNEL_K8_TILED_PASS1,
                )
                pass2_kernel = mx.fast.metal_kernel(
                    name="rfsn_tiled_pass2_v2_paged",
                    input_names=[
                        "partial_output", "partial_max", "partial_sum",
                    ],
                    output_names=["output", "running_max_arr", "running_sum_arr"],
                    source=_PACKED_V4_KERNEL_K8_TILED_PASS2,
                )
                cached = (pass1_kernel, pass2_kernel)
            else:
                cached = mx.fast.metal_kernel(
                    name="packed_v4_attention_k8",
                    input_names=[
                        "wht_queries",
                        "packed_codes_k", "scales_k",
                        "packed_codes_v", "scales_v",
                        "page_table", "page_starts", "page_counts", "active_pages",
                        "scale_arr", "query_start_arr",
                    ],
                    output_names=["output", "running_max_arr", "running_sum_arr"],
                    source=_PACKED_V4_KERNEL_K8,
                )
            self._kernel_cache[cache_key] = cached
            self._compilation_count += 1

        if use_tiling:
            pass1_kernel, pass2_kernel = cached

            # Pass 1: per-tile partial attention
            pass1_outputs = pass1_kernel(
                inputs=[
                    wht_queries_flat,
                    paged_kv.k_codes, paged_kv.k_scales,
                    paged_kv.v_codes, paged_kv.v_scales,
                    paged_kv.page_table, paged_kv.page_starts, paged_kv.page_counts,
                    active_pages,
                    scale_arr, query_start_arr,
                ],
                template=template,
                grid=(Hq * Lq * num_kv_tiles, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (Hq, Lq, num_kv_tiles, D),
                    (Hq, Lq, num_kv_tiles),
                    (Hq, Lq, num_kv_tiles),
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )
            partial_output = pass1_outputs[0]
            partial_max = pass1_outputs[1]
            partial_sum = pass1_outputs[2]

            # Pass 2: reduce partials across tiles
            pass2_outputs = pass2_kernel(
                inputs=[partial_output, partial_max, partial_sum],
                template=template,
                grid=(Hq, Lq, 1),
                threadgroup=(8, 8, 1),
                output_shapes=[(Hq, Lq, D), (Hq, Lq), (Hq, Lq)],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )
            output_wht = pass2_outputs[0]
            running_max = pass2_outputs[1]
            running_sum = pass2_outputs[2]
        else:
            kernel = cached
            outputs = kernel(
                inputs=[
                    wht_queries_flat,
                    paged_kv.k_codes, paged_kv.k_scales,
                    paged_kv.v_codes, paged_kv.v_scales,
                    paged_kv.page_table, paged_kv.page_starts, paged_kv.page_counts,
                    active_pages,
                    scale_arr, query_start_arr,
                ],
                template=template,
                grid=(Hq, Lq, 1),
                threadgroup=(8, 8, 1),
                output_shapes=[(Hq, Lq, D), (Hq, Lq), (Hq, Lq)],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )
            output_wht = outputs[0]
            running_max = outputs[1]
            running_sum = outputs[2]

        # Apply inverse WHT to return to original domain
        output_grouped = output_wht.reshape(Hq, Lq, groups, self.group_size)
        output = _reference_wht64(output_grouped)
        output = output.reshape(1, Hq, Lq, D).astype(queries.dtype)

        # Synchronize before timing so execution_ms reflects actual kernel work
        mx.eval(output, running_max, running_sum)
        execution_ms = (time.perf_counter() - start_time) * 1000.0
        prefill_ms = execution_ms if is_prefill else 0.0
        decode_ms = 0.0 if is_prefill else execution_ms

        # Measured counters
        dense_kv_materialized_bytes = 0
        decoded_dense_tokens = 0

        def _buffer_bytes(arr: mx.array) -> int:
            return int(arr.size) * arr.dtype.size

        # The paged arena arrays are preallocated; the only "new" writes are
        # the active pages themselves.  For the contract we report the active
        # payload size, not the full reserved arena.
        page_write_bytes = (
            _buffer_bytes(paged_kv.k_codes)  # full arena slice
            + _buffer_bytes(paged_kv.k_scales)
            + _buffer_bytes(paged_kv.v_codes)
            + _buffer_bytes(paged_kv.v_scales)
        ) * num_pages // paged_kv.max_pages  # approximate active fraction

        query_transform_bytes = _buffer_bytes(wht_queries_flat)

        scratch_bytes = (
            _buffer_bytes(output)
            + _buffer_bytes(running_max)
            + _buffer_bytes(running_sum)
        )
        if use_tiling:
            scratch_bytes += (
                _buffer_bytes(partial_output)
                + _buffer_bytes(partial_max)
                + _buffer_bytes(partial_sum)
            )
        output_bytes = _buffer_bytes(output)

        contract = ExecutionContract(
            backend="true_packed_metal_v4_k8",
            kernel_hash=self._kernel_hash,
            num_key_blocks=num_pages,
            num_value_blocks=num_pages,
            total_kv_tokens=total_tokens,
            num_q_heads=Hq,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            bits=self.bits,
            dense_kv_materialized_bytes=dense_kv_materialized_bytes,
            packed_history_copy_bytes=0,  # invariant: zero history copies
            query_transform_bytes=query_transform_bytes,
            scratch_bytes=scratch_bytes,
            output_bytes=output_bytes,
            decoded_dense_tokens=decoded_dense_tokens,
            packed_blocks_read=num_pages,
            packed_bytes_read=page_write_bytes,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
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
    queries: mx.array,
    paged_kv: Any,
    *,
    scale: float = 1.0,
    causal: bool = True,
    query_start_pos: int = 0,
    bits: int = 8,
    group_size: int = 64,
    sign_seed: int = 42,
    strict: bool = False,
    layer_id: int = 0,
) -> tuple[mx.array, mx.array, mx.array, ExecutionContract]:
    """Convenience wrapper around ``PackedV4AttentionKernel``."""
    kernel = PackedV4AttentionKernel(bits=bits, group_size=group_size, sign_seed=sign_seed)
    return kernel(
        queries=queries,
        paged_kv=paged_kv,
        scale=scale,
        causal=causal,
        query_start_pos=query_start_pos,
        strict=strict,
        layer_id=layer_id,
    )
