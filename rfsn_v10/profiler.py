#!/usr/bin/env python3
"""Performance profiling utilities for RFSN v10.

Provides tools to profile Metal kernel bottlenecks, memory access patterns,
and generate performance reports for optimization.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileEvent:
    """A single profiling event."""

    name: str
    start_time: float
    end_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class ProfileStats:
    """Statistics for a profiled operation."""

    name: str
    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    durations: list[float] = field(default_factory=list)

    def add_event(self, event: ProfileEvent) -> None:
        self.count += 1
        duration = event.duration_ms
        self.total_duration_ms += duration
        self.min_duration_ms = min(self.min_duration_ms, duration)
        self.max_duration_ms = max(self.max_duration_ms, duration)
        self.durations.append(duration)

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_duration_ms(self) -> float:
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        return sorted_durations[len(sorted_durations) // 2]


class RFSNProfiler:
    """Profiler for RFSN operations."""

    def __init__(self):
        self.events: list[ProfileEvent] = []
        self.stats: dict[str, ProfileStats] = defaultdict(ProfileStats)
        self._active_scopes: dict[str, float] = {}

    def start(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        """Start profiling a named operation."""
        self._active_scopes[name] = time.perf_counter()

    def end(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        """End profiling a named operation."""
        if name not in self._active_scopes:
            return

        start_time = self._active_scopes.pop(name)
        end_time = time.perf_counter()

        event = ProfileEvent(
            name=name,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {},
        )
        self.events.append(event)
        self.stats[name].add_event(event)

    def profile(self, name: str, metadata: dict[str, Any] | None = None):
        """Context manager for profiling a block of code."""

        class ProfileContext:
            def __init__(self, profiler: RFSNProfiler, n: str, meta: dict | None):
                self.profiler = profiler
                self.name = n
                self.metadata = meta

            def __enter__(self):
                self.profiler.start(self.name, self.metadata)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.end(self.name, self.metadata)

        return ProfileContext(self, name, metadata)

    def get_report(self) -> dict[str, Any]:
        """Generate a profiling report."""
        report = {
            "summary": {
                "total_events": len(self.events),
                "total_duration_ms": sum(e.duration_ms for e in self.events),
            },
            "operations": [],
        }

        for name, stats in sorted(self.stats.items()):
            report["operations"].append(
                {
                    "name": name,
                    "count": stats.count,
                    "total_ms": stats.total_duration_ms,
                    "avg_ms": stats.avg_duration_ms,
                    "p50_ms": stats.p50_duration_ms,
                    "min_ms": stats.min_duration_ms,
                    "max_ms": stats.max_duration_ms,
                }
            )

        # Sort by total duration
        report["operations"].sort(key=lambda x: x["total_ms"], reverse=True)

        return report

    def save_report(self, path: str) -> None:
        """Save profiling report to file."""
        report = self.get_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def reset(self) -> None:
        """Reset profiler state."""
        self.events.clear()
        self.stats.clear()
        self._active_scopes.clear()


class MemoryAccessProfiler:
    """Profile memory access patterns for optimization."""

    def __init__(self):
        self.access_log: list[dict[str, Any]] = []
        self.block_access_counts: dict[int, int] = defaultdict(int)

    def log_access(
        self,
        block_idx: int,
        access_type: str,
        size_bytes: int,
        timestamp: float | None = None,
    ) -> None:
        """Log a memory access event."""
        if timestamp is None:
            timestamp = time.perf_counter()

        self.access_log.append(
            {
                "block_idx": block_idx,
                "access_type": access_type,
                "size_bytes": size_bytes,
                "timestamp": timestamp,
            }
        )
        self.block_access_counts[block_idx] += 1

    def get_hot_blocks(self, top_k: int = 10) -> list[tuple[int, int]]:
        """Get the most frequently accessed blocks."""
        sorted_blocks = sorted(
            self.block_access_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_blocks[:top_k]

    def get_access_pattern_report(self) -> dict[str, Any]:
        """Generate access pattern report."""
        if not self.access_log:
            return {"total_accesses": 0, "hot_blocks": []}

        total_bytes = sum(log["size_bytes"] for log in self.access_log)
        avg_access_size = total_bytes / len(self.access_log)

        return {
            "total_accesses": len(self.access_log),
            "total_bytes": total_bytes,
            "avg_bytes_per_access": avg_access_size,
            "unique_blocks": len(self.block_access_counts),
            "hot_blocks": self.get_hot_blocks(),
        }

    def reset(self) -> None:
        """Reset profiler state."""
        self.access_log.clear()
        self.block_access_counts.clear()


def profile_kernel_execution(
    kernel_func: Callable,
    *args,
    iterations: int = 10,
    warmup: int = 3,
    **kwargs,
) -> dict[str, Any]:
    """Profile a kernel function execution.

    Args:
        kernel_func: The kernel function to profile
        *args: Positional arguments for the kernel
        iterations: Number of profiling iterations
        warmup: Number of warmup iterations
        **kwargs: Keyword arguments for the kernel

    Returns:
        Profiling statistics
    """
    profiler = RFSNProfiler()

    # Warmup
    for _ in range(warmup):
        kernel_func(*args, **kwargs)

    # Profile
    latencies = []
    for i in range(iterations):
        with profiler.profile(f"kernel_iteration_{i}"):
            kernel_func(*args, **kwargs)
        latencies.append(profiler.stats[f"kernel_iteration_{i}"].avg_duration_ms)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "iterations": iterations,
        "profiler_report": profiler.get_report(),
    }
