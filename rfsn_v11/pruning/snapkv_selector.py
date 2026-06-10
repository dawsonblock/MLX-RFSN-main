"""SnapKV-style observation-window KV position selector.

Algorithm (from SnapKV paper, Zhang et al. 2024):
1. Take Q vectors from the last `window_size` tokens (observation window).
2. Compute attention scores Q @ K_prefix^T / sqrt(D).
3. Average pooling with kernel size `pool_kernel` over the key axis.
4. Sum votes across observation window tokens.
5. Select top-k key positions by vote score.

Per-head selection: each attention head independently selects positions.
Block-aligned selection: key positions are selected in blocks of `block_size`
tokens to improve memory access patterns.

Parameters
----------
window_size : int
    Number of recent tokens used as queries for voting. Default 512.
pool_kernel : int
    Kernel size for max/avg pooling over key scores. Default 7.
retention_ratio : float
    Fraction of key positions to retain. Default 0.20.
block_size : int
    Block alignment for selected positions. Default 64.
enable_threshold : int
    Minimum context length to enable pruning. Default 8192.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def _average_pool_1d(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """1D average pooling with 'same' padding."""
    pad = kernel_size // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    cumsum = np.cumsum(np.insert(padded, 0, 0))
    return (cumsum[kernel_size:] - cumsum[:-kernel_size]) / kernel_size


class SnapKVSelector:
    """Per-head observation-window KV position selector."""

    def __init__(
        self,
        window_size: int = 512,
        pool_kernel: int = 7,
        retention_ratio: float = 0.20,
        block_size: int = 64,
        enable_threshold: int = 8192,
    ) -> None:
        self.window_size = window_size
        self.pool_kernel = pool_kernel
        self.retention_ratio = retention_ratio
        self.block_size = block_size
        self.enable_threshold = enable_threshold

    def should_enable(self, context_length: int) -> bool:
        return context_length >= self.enable_threshold

    def select_positions(
        self,
        Q_obs: np.ndarray,
        K_prefix: np.ndarray,
        scale: float | None = None,
    ) -> dict[str, Any]:
        """Select key positions per head.

        Parameters
        ----------
        Q_obs : (H, W, D) observation-window queries
        K_prefix : (H, T, D) full prefix keys
        scale : 1/sqrt(D)

        Returns
        -------
        dict with:
            selected_indices: dict[int, list[int]]
                — per-head selected token indices
            vote_time_ms: float
            retention_ratio_actual: float
            selected_tokens: int
        """
        t0 = time.perf_counter()
        H, W, D = Q_obs.shape
        _, T, _ = K_prefix.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        selected_per_head: dict[int, list[int]] = {}
        total_selected = 0

        for h in range(H):
            q_h = Q_obs[h]   # (W, D)
            k_h = K_prefix[h]  # (T, D)

            # Attention scores: (W, T)
            scores = q_h @ k_h.T * scale

            # Pool over key axis to smooth noise
            pooled_scores = np.apply_along_axis(
                lambda row: _average_pool_1d(row, self.pool_kernel),
                axis=1,
                arr=scores,
            )

            # Sum votes across observation window tokens
            vote_scores = pooled_scores.sum(axis=0)  # (T,)

            # Select top-k tokens
            k_retain = max(1, int(T * self.retention_ratio))
            top_indices = np.argpartition(vote_scores, -k_retain)[-k_retain:]
            top_indices = np.sort(top_indices)

            selected_per_head[int(h)] = top_indices.tolist()
            total_selected += len(top_indices)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_retention = (total_selected / max(H * T, 1))

        return {
            "selected_indices": selected_per_head,
            "vote_time_ms": elapsed_ms,
            "retention_ratio_actual": avg_retention,
            "selected_tokens": total_selected,
        }

    def select_blocks(
        self,
        Q_obs: np.ndarray,
        K_prefix: np.ndarray,
        scale: float | None = None,
    ) -> dict[str, Any]:
        """Block-aligned selection.

        Converts per-token votes to block scores, selects top blocks.
        """
        H, W, D = Q_obs.shape
        _, T, _ = K_prefix.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        t0 = time.perf_counter()
        n_blocks = max(T // self.block_size, 1)
        selected_blocks_per_head: dict[int, list[int]] = {}
        total_selected = 0

        for h in range(H):
            q_h = Q_obs[h]
            k_h = K_prefix[h]
            scores = q_h @ k_h.T * scale
            pooled = np.apply_along_axis(
                lambda row: _average_pool_1d(row, self.pool_kernel),
                axis=1,
                arr=scores,
            )
            vote_scores = pooled.sum(axis=0)

            # Aggregate to block scores
            block_scores = np.zeros(n_blocks)
            for b in range(n_blocks):
                start = b * self.block_size
                end = min((b + 1) * self.block_size, T)
                block_scores[b] = vote_scores[start:end].mean()

            k_retain = max(1, int(n_blocks * self.retention_ratio))
            top_blocks = np.argpartition(block_scores, -k_retain)[-k_retain:]
            top_blocks = np.sort(top_blocks)
            selected_blocks_per_head[int(h)] = top_blocks.tolist()

            # Expand blocks to token indices
            total_selected += k_retain * self.block_size

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_retention = total_selected / max(H * T, 1)

        return {
            "selected_blocks": selected_blocks_per_head,
            "vote_time_ms": elapsed_ms,
            "retention_ratio_actual": avg_retention,
            "selected_tokens": total_selected,
        }

    def estimate_memory_saved_mb(
        self,
        context_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        retention_ratio: float | None = None,
    ) -> float:
        """Estimate memory saved by pruning (in MB)."""
        rr = retention_ratio or self.retention_ratio
        saved_ratio = 1.0 - rr
        # K+V, FP16
        bytes_per_token = 2 * num_layers * num_heads * head_dim * 2
        saved_bytes = context_length * bytes_per_token * saved_ratio
        return saved_bytes / (1024 ** 2)
