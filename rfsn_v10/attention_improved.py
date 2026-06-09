#!/usr/bin/env python3
"""
Improved block-sparse attention with quality-aware block selection.

This module implements enhanced sparse attention strategies to improve
sparse quality from the current 0.50-0.88 range to the target 0.90+.
"""

from __future__ import annotations

import math
from typing import Literal

from .compat import mx
from .memory_guard import MemoryGuard

ExecutionMode = Literal[
    "sparse_compacted",
    "dense_requested",
    "dense_short_context",
    "dense_prefill",
    "dense_not_strictly_past",
]


class ImprovedBlockSelection:
    """Enhanced block selection strategies for better sparse quality."""

    @staticmethod
    def _compute_block_importance(
        queries: mx.array,
        keys: mx.array,
        block_size: int,
        strategy: str = "hybrid",
    ) -> mx.array:
        """Compute importance scores for each block.

        Args:
            queries: Query tensor [B, H, 1, D]
            keys: Key tensor [B, H, T_k, D]
            block_size: Block size for partitioning
            strategy: Selection strategy ('hybrid', 'attention_max',
                'variance', 'gradient')

        Returns:
            Block importance scores [B, 1, num_blocks]
        """
        B, H, T_k, D = keys.shape
        num_blocks = math.ceil(T_k / block_size)

        # Pad keys to block boundary
        pad_len = (block_size - (T_k % block_size)) % block_size
        if pad_len > 0:
            k_pad = mx.concatenate(
                [keys, mx.zeros((B, H, pad_len, D), dtype=keys.dtype)],
                axis=2,
            )
        else:
            k_pad = keys

        k_reshaped = k_pad.reshape(B, H, num_blocks, block_size, D)

        if strategy == "attention_max":
            # Use max attention score per block (current approach)
            k_pooled = mx.mean(k_reshaped, axis=3)  # [B, H, num_blocks, D]
            scores = (queries @ k_pooled.transpose(0, 1, 3, 2)) / math.sqrt(D)
            block_scores = mx.max(scores, axis=1)  # [B, 1, num_blocks]

        elif strategy == "variance":
            # Use variance within block as importance signal
            k_var = mx.var(k_reshaped, axis=3)  # [B, H, num_blocks, D]
            var_importance = mx.mean(k_var, axis=3)  # [B, H, num_blocks]
            # Normalize
            var_importance = var_importance / (
                mx.max(var_importance) + 1e-8
            )
            block_scores = mx.mean(var_importance, axis=1, keepdims=True)

        elif strategy == "gradient":
            # Use gradient-based importance (simulated via norm)
            k_norm = mx.norm(k_reshaped, axis=(3, 4))  # [B, H, num_blocks]
            block_scores = mx.mean(k_norm, axis=1, keepdims=True)

        else:  # hybrid
            # Combine multiple signals
            k_pooled = mx.mean(k_reshaped, axis=3)
            attention_scores = (queries @ k_pooled.transpose(0, 1, 3, 2)) / math.sqrt(D)
            max_attention = mx.max(attention_scores, axis=1)

            k_var = mx.var(k_reshaped, axis=3)
            var_importance = mx.mean(k_var, axis=3)
            var_importance = var_importance / (mx.max(var_importance) + 1e-8)
            var_importance = mx.mean(var_importance, axis=1, keepdims=True)

            # Weighted combination
            block_scores = 0.7 * max_attention + 0.3 * var_importance

        return block_scores

    @staticmethod
    def _select_blocks_with_quality_awareness(
        block_scores: mx.array,
        k_active: int,
        reserved_sink_blocks: int,
        reserved_recent_blocks: int,
        num_blocks: int,
        quality_history: list[float] | None = None,
    ) -> list[int]:
        """Select blocks with quality-aware scheduling.

        Args:
            block_scores: Importance scores [B, 1, num_blocks]
            k_active: Number of blocks to select
            reserved_sink_blocks: Number of sink blocks to reserve
            reserved_recent_blocks: Number of recent blocks to reserve
            num_blocks: Total number of blocks
            quality_history: Historical quality scores for adaptive selection

        Returns:
            List of selected block indices
        """
        scores_np = block_scores[0, 0].tolist()
        scored_blocks = sorted(enumerate(scores_np), key=lambda x: -x[1])

        # Always reserve sink blocks
        reserved = set(range(min(reserved_sink_blocks, num_blocks)))

        # Always reserve recent blocks
        for offset in range(reserved_recent_blocks):
            idx = num_blocks - 1 - offset
            if idx >= 0:
                reserved.add(idx)

        # Adaptive selection based on quality history
        if quality_history and len(quality_history) > 0:
            recent_quality = quality_history[-1]
            if recent_quality < 0.85:
                # Low quality: be more conservative, select more high-scoring blocks
                selection_buffer = 2
            else:
                # Good quality: can be more aggressive
                selection_buffer = 0
        else:
            selection_buffer = 0

        selected = list(reserved)
        for idx, _ in scored_blocks:
            if len(selected) >= k_active + selection_buffer:
                break
            if idx not in reserved:
                selected.append(idx)

        # Trim to exact budget
        selected = sorted(selected)[:k_active]
        return selected


class QualityAwareSparseAttention:
    """Sparse attention with per-layer quality monitoring and adaptive selection."""

    def __init__(
        self,
        block_size: int = 64,
        selection_strategy: str = "hybrid",
        enable_quality_monitoring: bool = True,
    ):
        self.block_size = block_size
        self.selection_strategy = selection_strategy
        self.enable_quality_monitoring = enable_quality_monitoring
        self.quality_history: dict[str, list[float]] = {}

    def _record_quality(self, layer_id: str, quality: float) -> None:
        """Record quality metric for a layer."""
        if layer_id not in self.quality_history:
            self.quality_history[layer_id] = []
        self.quality_history[layer_id].append(quality)
        # Keep only recent history
        if len(self.quality_history[layer_id]) > 10:
            self.quality_history[layer_id] = self.quality_history[layer_id][-10:]

    def _get_layer_quality(self, layer_id: str) -> float | None:
        """Get recent average quality for a layer."""
        if layer_id not in self.quality_history or not self.quality_history[layer_id]:
            return None
        return sum(self.quality_history[layer_id]) / len(self.quality_history[layer_id])

    @staticmethod
    def _dense_masked(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: float,
        block_size: int,
        mode: str,
    ) -> tuple[mx.array, int, str]:
        """Causal dense attention fallback with masking for prefill."""
        B, H, T_q, D = queries.shape
        T_k = keys.shape[2]
        num_blocks = max(1, math.ceil(T_k / block_size))
        if T_q == 1:
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale
            )
            return out, num_blocks, mode
        scores = queries @ keys.transpose(0, 1, 3, 2) * scale
        q_pos = mx.arange(T_q, dtype=mx.int32).reshape(1, 1, T_q, 1)
        k_pos = mx.arange(T_k, dtype=mx.int32).reshape(1, 1, 1, T_k)
        offset = T_k - T_q
        causal = (k_pos <= (q_pos + offset)).astype(scores.dtype)
        scores = scores * causal + (1.0 - causal) * mx.array(
            -1e9, dtype=scores.dtype
        )
        weights = mx.softmax(scores, axis=-1)
        out = weights @ values
        return out, num_blocks, mode

    def execute(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float,
        layer_id: str = "default",
        kv_is_strictly_past: bool = True,
        reserved_sink_blocks: int = 1,
        reserved_recent_blocks: int = 2,
        memory_guard: MemoryGuard | None = None,  # noqa: ARG002
    ) -> tuple[mx.array, int, ExecutionMode, dict]:
        """
        Execute quality-aware sparse attention.

        Returns:
            (attention_output, num_active_blocks, execution_mode,
                metadata)
        """
        _, _, T_q, T_k = keys.shape
        D = keys.shape[3]
        scale = 1.0 / math.sqrt(D)

        # Dense fallback conditions
        if top_k_ratio >= 1.0:
            return (*QualityAwareSparseAttention._dense_masked(
                queries, keys, values, scale, self.block_size, "dense_requested",
            ), {})

        if T_k <= self.block_size or T_q > 1 or not kv_is_strictly_past:
            mode = (
                "dense_prefill" if T_q > 1
                else "dense_not_strictly_past" if not kv_is_strictly_past
                else "dense_short_context"
            )
            return (*QualityAwareSparseAttention._dense_masked(
                queries, keys, values, scale, self.block_size, mode,
            ), {})

        # Compute block importance
        block_scores = ImprovedBlockSelection._compute_block_importance(
            queries, keys, self.block_size, self.selection_strategy
        )

        num_blocks = math.ceil(T_k / self.block_size)
        k_active = max(1, int(math.ceil(num_blocks * top_k_ratio)))

        # Quality-aware block selection
        layer_quality = (
            self._get_layer_quality(layer_id)
            if self.enable_quality_monitoring
            else None
        )
        selected_blocks = (
            ImprovedBlockSelection._select_blocks_with_quality_awareness(
                block_scores,
                k_active,
                reserved_sink_blocks,
                reserved_recent_blocks,
                num_blocks,
                [layer_quality] if layer_quality else None,
            )
        )

        # Compact selected blocks
        compacted_keys = []
        compacted_values = []

        for block_idx in selected_blocks:
            start = block_idx * self.block_size
            end = min(start + self.block_size, T_k)
            compacted_keys.append(keys[:, :, start:end, :])
            compacted_values.append(values[:, :, start:end, :])

        if compacted_keys:
            k_compacted = mx.concatenate(compacted_keys, axis=2)
            v_compacted = mx.concatenate(compacted_values, axis=2)
        else:
            k_compacted = keys[:, :, :1, :]
            v_compacted = values[:, :, :1, :]

        # Compute attention with compacted KV
        out = mx.fast.scaled_dot_product_attention(
            queries, k_compacted, v_compacted, scale=1.0 / math.sqrt(D)
        )

        metadata = {
            "selected_blocks": selected_blocks,
            "block_scores": (
                block_scores[0, 0].tolist()
                if hasattr(block_scores, "tolist")
                else []
            ),
            "layer_quality": layer_quality,
            "selection_strategy": self.selection_strategy,
        }

        return out, len(selected_blocks), "sparse_compacted", metadata
