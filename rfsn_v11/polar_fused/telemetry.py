"""Minimal telemetry for polar_fused backend.

Collects per-call latency and backend selection without external dependencies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolarTelemetry:
    """Per-attention-call telemetry record."""

    backend: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    heads: int = 0
    head_dim: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class PolarTelemetryCollector:
    """Simple collector for polar_fused telemetry events."""

    def __init__(self) -> None:
        self._events: list[PolarTelemetry] = []

    def record(self, event: PolarTelemetry) -> None:
        self._events.append(event)

    def clear(self) -> None:
        self._events.clear()

    def summary(self) -> dict[str, Any]:
        if not self._events:
            return {}
        backends = {}
        for ev in self._events:
            backends.setdefault(ev.backend, {"count": 0, "total_ms": 0.0})
            backends[ev.backend]["count"] += 1
            backends[ev.backend]["total_ms"] += ev.latency_ms
        return {
            "total_calls": len(self._events),
            "backends": {
                name: {
                    "count": info["count"],
                    "mean_latency_ms": info["total_ms"] / info["count"],
                }
                for name, info in backends.items()
            },
        }
