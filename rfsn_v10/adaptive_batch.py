#!/usr/bin/env python3
"""Adaptive batch sizing for optimal performance.

Implements dynamic batch size adjustment based on performance metrics
to optimize throughput and latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveBatchConfig:
    """Configuration for adaptive batch sizing."""

    min_batch_size: int = 1
    max_batch_size: int = 128
    initial_batch_size: int = 32
    target_latency_ms: float = 10.0
    latency_tolerance: float = 2.0
    adjustment_factor: float = 1.2


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on performance."""

    def __init__(self, config: AdaptiveBatchConfig | None = None):
        self.config = config or AdaptiveBatchConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.latency_history: list[float] = []
        self.batch_history: list[int] = []

    def update(self, latency_ms: float) -> int:
        """Update batch size based on observed latency.

        Args:
            latency_ms: Observed latency for current batch

        Returns:
            New batch size to use
        """
        self.latency_history.append(latency_ms)
        self.batch_history.append(self.current_batch_size)

        # Keep history limited
        if len(self.latency_history) > 10:
            self.latency_history = self.latency_history[-10:]
            self.batch_history = self.batch_history[-10:]

        # Calculate average recent latency
        if len(self.latency_history) >= 3:
            avg_latency = sum(self.latency_history[-3:]) / 3
        else:
            avg_latency = latency_ms

        # Adjust batch size based on latency
        if avg_latency > self.config.target_latency_ms + self.config.latency_tolerance:
            # Latency too high, reduce batch size
            new_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size / self.config.adjustment_factor),
            )
            self.current_batch_size = new_size
        elif avg_latency < self.config.target_latency_ms - self.config.latency_tolerance:
            # Latency low, can increase batch size
            new_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.adjustment_factor),
            )
            self.current_batch_size = new_size

        return self.current_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size

    def get_stats(self) -> dict:
        """Get statistics about batch sizing."""
        return {
            "current_batch_size": self.current_batch_size,
            "avg_latency_ms": sum(self.latency_history) / len(self.latency_history)
            if self.latency_history
            else 0.0,
            "latency_history": self.latency_history,
            "batch_history": self.batch_history,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_batch_size = self.config.initial_batch_size
        self.latency_history.clear()
        self.batch_history.clear()


class PerformancePredictor:
    """Simple performance predictor based on historical data."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: list[tuple[int, float]] = []  # (batch_size, latency)

    def record(self, batch_size: int, latency_ms: float) -> None:
        """Record a performance observation."""
        self.history.append((batch_size, latency_ms))
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def predict_latency(self, batch_size: int) -> float | None:
        """Predict latency for a given batch size."""
        if not self.history:
            return None

        # Simple linear regression on recent data
        n = len(self.history)
        sum_x = sum(bs for bs, _ in self.history)
        sum_y = sum(lat for _, lat in self.history)
        sum_xy = sum(bs * lat for bs, lat in self.history)
        sum_x2 = sum(bs * bs for bs, _ in self.history)

        if n < 2 or sum_x2 * n - sum_x * sum_x == 0:
            # Not enough data or would divide by zero
            return None

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        predicted = slope * batch_size + intercept
        return max(0.0, predicted)

    def recommend_batch_size(self, target_latency_ms: float) -> int | None:
        """Recommend batch size for target latency."""
        if not self.history:
            return None

        # Use average of recent batch sizes as starting point
        avg_batch = sum(bs for bs, _ in self.history) / len(self.history)

        # Binary search for best batch size
        low = 1
        high = max(bs for bs, _ in self.history) * 2
        best_batch = avg_batch

        for _ in range(10):  # Limited iterations
            mid = (low + high) // 2
            predicted = self.predict_latency(mid)

            if predicted is None:
                break

            if predicted > target_latency_ms:
                high = mid
            else:
                best_batch = mid
                low = mid + 1

        return int(best_batch)

    def reset(self) -> None:
        """Reset history."""
        self.history.clear()
