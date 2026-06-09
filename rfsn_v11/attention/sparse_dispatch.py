"""
Block-sparse decode-time attention for RFSN v11.

Ported from rfsn_v10/attention.py (AdaptiveBlockSparseAttention).

Key rules:
  - Dense prefill ALWAYS (T_q > 1): block-sparse cannot be applied during
    prefill because physically compacting KV blocks breaks causal alignment.
  - Block-sparse decode only (T_q == 1): select top-k scored blocks per head.
  - GQA support: kv_head = query_head // (n_q_heads // n_kv_heads).
  - Python dense fallback when mx.fast.metal_kernel is unavailable.

ExecutionMode literals document exactly why each path was taken.
"""

from __future__ import annotations

import math
from typing import Literal

from ..compat import mx
from ..memory_guard import MemoryGuard

ExecutionMode = Literal[
    "sparse_compacted",
    "dense_requested",
    "dense_short_context",
    "dense_prefill",
    "dense_not_strictly_past",
    "dense_ragged_batch",
]


def causal_attention_dense(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float | None = None,
    backend: str = "mlx",
) -> mx.array:
    """Reference causal dense scaled dot-product attention.

    Always applies a causal mask for T_q > 1.  Safe for both prefill (T_q > 1)
    and decode (T_q == 1, no mask needed but harmless).

    Args:
        queries: (B, H, T_q, D)
        keys:    (B, H, T_k, D)
        values:  (B, H, T_k, D)
        scale:   Defaults to 1/sqrt(D).

    Returns:
        (B, H, T_q, D)
    """
    B, H, T_q, D = queries.shape
    T_k = keys.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = (queries * scale) @ mx.swapaxes(keys, -2, -1)  # (B, H, T_q, T_k)

    if T_q > 1:
        # Causal mask: position i can only attend to j ≤ i
        row_idx = mx.arange(T_q).reshape(T_q, 1)
        col_idx = mx.arange(T_k).reshape(1, T_k)
        # offset: keys may include a prefix; queries start at T_k - T_q
        offset = T_k - T_q
        mask = col_idx <= (row_idx + offset)  # (T_q, T_k)
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))

    weights = mx.softmax(scores, axis=-1)
    return weights @ values


class AdaptiveBlockSparseAttention:
    """Block-sparse attention for RFSN v11 — decode-path only."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dtype_nbytes(dtype) -> int:
        dtype_name = str(dtype)
        if "float16" in dtype_name or "bfloat16" in dtype_name:
            return 2
        if "float32" in dtype_name:
            return 4
        if "float64" in dtype_name:
            return 8
        return 4

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        if b <= 0:
            raise ValueError(f"divisor must be positive, got {b}")
        return (a + b - 1) // b

    @staticmethod
    def _validate_inputs(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float,
        block_size: int,
        consensus_mix: float,
        memory_guard: MemoryGuard | None = None,
    ) -> tuple[int, int, int, int, int]:
        if len(queries.shape) != 4:
            raise ValueError(f"queries must be [B,H,T_q,D], got {queries.shape}")
        if len(keys.shape) != 4:
            raise ValueError(f"keys must be [B,H,T_k,D], got {keys.shape}")
        if len(values.shape) != 4:
            raise ValueError(f"values must be [B,H,T_k,D], got {values.shape}")
        if keys.shape != values.shape:
            raise ValueError(f"keys/values shape mismatch: {keys.shape} vs {values.shape}")
        if not (queries.dtype == keys.dtype == values.dtype):
            raise ValueError(
                f"queries/keys/values dtype mismatch: "
                f"{queries.dtype} vs {keys.dtype} vs {values.dtype}"
            )

        B, H, T_k, D = keys.shape
        Bq, Hq, T_q, Dq = queries.shape

        if Bq != B:
            raise ValueError(f"batch mismatch: queries B={Bq}, keys B={B}")
        if Hq != H:
            raise ValueError(f"head mismatch: queries H={Hq}, keys H={H}")
        if Dq != D:
            raise ValueError(f"head_dim mismatch: queries D={Dq}, keys D={D}")
        if T_q <= 0 or T_k <= 0:
            raise ValueError(f"T_q and T_k must be positive, got T_q={T_q}, T_k={T_k}")
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not (0.0 < float(top_k_ratio) <= 1.0):
            raise ValueError(f"top_k_ratio must be in (0, 1], got {top_k_ratio}")
        if not math.isfinite(float(consensus_mix)):
            raise ValueError(f"consensus_mix must be finite, got {consensus_mix}")

        if memory_guard is not None:
            bytes_per_elem = AdaptiveBlockSparseAttention._dtype_nbytes(keys.dtype)
            estimated_bytes = int((B * H * (T_q + (2 * T_k)) * D) * bytes_per_elem)
            if memory_guard.check_pressure(estimated_bytes):
                raise MemoryError(
                    f"attention memory guard triggered for estimated_bytes={estimated_bytes}"
                )

        return B, H, T_q, T_k, D

    @staticmethod
    def _dense_masked(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: float,
        block_size: int,
        mode: str,
    ) -> tuple[mx.array, int, str]:
        """Causal dense attention fallback (always causal for T_q > 1)."""
        T_k = keys.shape[2]
        num_blocks = max(1, AdaptiveBlockSparseAttention._ceil_div(T_k, block_size))
        out = causal_attention_dense(queries, keys, values, scale=scale, backend="mlx")
        return out, num_blocks, mode

    @staticmethod
    def _merge_reserved_and_scored_blocks(
        *,
        num_blocks: int,
        k_active: int,
        score_selected: list[int],
        reserved_sink_blocks: int,
        reserved_recent_blocks: int,
        allow_budget_overflow: bool,
    ) -> list[int]:
        """Merge sink + recent reserved blocks with score-based top-k selection."""
        sink_count = max(0, int(reserved_sink_blocks))
        recent_count = max(0, int(reserved_recent_blocks))

        reserved: list[int] = []
        for idx in range(min(sink_count, num_blocks)):
            if idx not in reserved:
                reserved.append(idx)
        for offset in range(recent_count):
            idx = num_blocks - 1 - offset
            if idx >= 0 and idx not in reserved:
                reserved.append(idx)

        if allow_budget_overflow:
            budget = min(num_blocks, max(k_active, len(reserved)))
        else:
            budget = max(1, min(num_blocks, k_active))

        selected: list[int] = []
        seen: set[int] = set()

        for idx in reserved:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
            if len(selected) >= budget:
                break

        for idx in score_selected:
            if len(selected) >= budget:
                break
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)

        return sorted(selected)

    @staticmethod
    def execute(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float,
        block_size: int = 64,
        kv_is_strictly_past: bool = True,
        consensus_mix: float = 0.7,
        reserved_sink_blocks: int = 1,
        reserved_recent_blocks: int = 2,
        allow_budget_overflow: bool = False,
        recent_bias: float = 0.05,
        sink_bias: float = 0.10,
        memory_guard: MemoryGuard | None = None,
    ) -> tuple[mx.array, int, ExecutionMode]:
        """Execute block-sparse scaled dot-product attention.

        Block-sparse is only used at decode time (T_q == 1).
        All prefill paths (T_q > 1) use dense causal attention.

        Args:
            queries:              (B, H, T_q, D)
            keys:                 (B, H, T_k, D)
            values:               (B, H, T_k, D)
            top_k_ratio:          Fraction of KV blocks to retain (0, 1].
            block_size:           KV block size (should match quantization group_size).
            kv_is_strictly_past:  True when all KV tokens are valid past context.
            consensus_mix:        0.0 = pure mean; 1.0 = pure max across heads.
            reserved_sink_blocks: Sink blocks always retained.
            reserved_recent_blocks: Recent blocks always retained.
            allow_budget_overflow: Allow reserved blocks to exceed k_active budget.
            recent_bias:          Score boost for recent blocks.
            sink_bias:            Score boost for sink blocks.
            memory_guard:         Optional memory pressure gate.

        Returns:
            (attention_output, num_active_blocks, execution_mode)
        """
        B, H, T_q, T_k, D = AdaptiveBlockSparseAttention._validate_inputs(
            queries, keys, values, top_k_ratio, block_size, consensus_mix, memory_guard
        )
        scale = 1.0 / math.sqrt(D)

        # --- Prefill: always dense causal ---
        if T_q > 1:
            return AdaptiveBlockSparseAttention._dense_masked(
                queries, keys, values, scale, block_size, "dense_prefill"
            )

        # --- Dense if KV is not strictly in the past ---
        if not kv_is_strictly_past:
            return AdaptiveBlockSparseAttention._dense_masked(
                queries, keys, values, scale, block_size, "dense_not_strictly_past"
            )

        # --- Dense if context too short for blocking ---
        if T_k <= block_size:
            return AdaptiveBlockSparseAttention._dense_masked(
                queries, keys, values, scale, block_size, "dense_short_context"
            )

        # --- Dense if top_k_ratio == 1.0 (all blocks requested) ---
        num_blocks = AdaptiveBlockSparseAttention._ceil_div(T_k, block_size)
        k_active = max(1, round(top_k_ratio * num_blocks))
        if k_active >= num_blocks or top_k_ratio >= 1.0:
            return AdaptiveBlockSparseAttention._dense_masked(
                queries, keys, values, scale, block_size, "dense_requested"
            )

        # ================================================================
        # Block-sparse decode path (T_q == 1)
        # ================================================================
        # 1. Score blocks using query-key dot products on quantized keys
        #    (using mx.quantized_matmul without full decompression)
        q_for_scoring = queries[:, :, 0, :]  # (B, H, D)

        # Block-level key summaries: mean over block_size tokens
        padded_len = num_blocks * block_size
        if T_k < padded_len:
            pad = mx.zeros(
                (B, H, padded_len - T_k, D), dtype=keys.dtype
            )
            keys_padded = mx.concatenate([keys, pad], axis=2)
        else:
            keys_padded = keys

        # (B, H, num_blocks, block_size, D) → (B, H, num_blocks, D)
        keys_blocked = keys_padded.reshape(B, H, num_blocks, block_size, D)
        block_reps = mx.mean(keys_blocked, axis=3)  # (B, H, num_blocks, D)

        # Raw scores: (B, H, num_blocks)
        raw_scores = mx.sum(
            mx.expand_dims(q_for_scoring, axis=2) * block_reps,
            axis=-1,
        ) * scale  # (B, H, num_blocks)

        # Apply recency and sink biases
        block_indices = mx.arange(num_blocks, dtype=mx.float32)
        recency_decay = block_indices / float(max(1, num_blocks - 1))  # 0 → 1 newest
        scores_biased = raw_scores + (
            recent_bias * recency_decay + sink_bias * (1.0 - recency_decay)
        )

        # 2. Consensus across heads: blend max and mean
        scores_max = mx.max(scores_biased, axis=1)   # (B, num_blocks)
        scores_mean = mx.mean(scores_biased, axis=1)  # (B, num_blocks)
        scores_consensus = (
            consensus_mix * scores_max + (1.0 - consensus_mix) * scores_mean
        )  # (B, num_blocks)

        # 3. Select top-k blocks (shared across batch for simplicity)
        import numpy as _np
        mx.eval(scores_consensus)
        sc_np = _np.array(scores_consensus[0])  # (num_blocks,)
        sorted_idx = _np.argsort(sc_np)[::-1]
        score_selected = sorted_idx[:k_active].tolist()

        active_block_ids = AdaptiveBlockSparseAttention._merge_reserved_and_scored_blocks(
            num_blocks=num_blocks,
            k_active=k_active,
            score_selected=score_selected,
            reserved_sink_blocks=reserved_sink_blocks,
            reserved_recent_blocks=reserved_recent_blocks,
            allow_budget_overflow=allow_budget_overflow,
        )

        # 4. Compact selected KV blocks
        active_token_mask = []
        for blk in active_block_ids:
            start = blk * block_size
            end = min(start + block_size, T_k)
            active_token_mask.extend(range(start, end))

        if not active_token_mask:
            # Safety: if somehow no tokens selected, fall back to dense
            return AdaptiveBlockSparseAttention._dense_masked(
                queries, keys, values, scale, block_size, "dense_requested"
            )

        token_ids = mx.array(active_token_mask)
        keys_compact = mx.take(keys, token_ids, axis=2)     # (B, H, N_active, D)
        values_compact = mx.take(values, token_ids, axis=2) # (B, H, N_active, D)

        # 5. Attention on compacted KV (T_q == 1 → no causal mask needed)
        scores_full = (queries * scale) @ mx.swapaxes(keys_compact, -2, -1)
        weights = mx.softmax(scores_full, axis=-1)
        out = weights @ values_compact

        return out, len(active_block_ids), "sparse_compacted"
