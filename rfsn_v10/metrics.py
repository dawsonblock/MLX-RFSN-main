#!/usr/bin/env python3
"""Metrics export for RFSN v10.

Provides metrics collection and export for monitoring systems.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Metric:
    """A metric data point."""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsRegistry:
    """Registry for collecting metrics."""

    def __init__(self):
        self.metrics: list[Metric] = []
        self.gauges: dict[str, float] = {}
        self.counters: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        with self.lock:
            self.gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type="gauge",
            )
            self.metrics.append(metric)

    def counter(self, name: str, increment: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Record a counter metric."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0.0) + increment
            metric = Metric(
                name=name,
                value=self.counters[name],
                timestamp=time.time(),
                tags=tags or {},
                metric_type="counter",
            )
            self.metrics.append(metric)

    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        with self.lock:
            self.histograms[name].append(value)
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type="histogram",
            )
            self.metrics.append(metric)

    def get_metric_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            summary = {
                "gauges": dict(self.gauges),
                "counters": dict(self.counters),
                "histograms": {},
            }

            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    summary["histograms"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": sorted_values[len(sorted_values) // 2],
                        "p95": sorted_values[int(len(sorted_values) * 0.95)],
                        "p99": sorted_values[int(len(sorted_values) * 0.99)],
                    }

            return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Histograms
        for name, values in self.histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values)}")

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics in JSON format."""
        import json

        return json.dumps(self.get_metric_summary(), indent=2)

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.gauges.clear()
            self.counters.clear()
            self.histograms.clear()


class MetricsCollector:
    """Collects metrics from RFSN components."""

    def __init__(self, registry: MetricsRegistry | None = None):
        self.registry = registry or MetricsRegistry()
        self.collectors: dict[str, Callable[[], dict[str, Any]]] = {}

    def register_collector(self, name: str, collector: Callable[[], dict[str, Any]]) -> None:
        """Register a metrics collector."""
        self.collectors[name] = collector

    def collect_all(self) -> None:
        """Collect metrics from all registered collectors."""
        for name, collector in self.collectors.items():
            try:
                metrics = collector()
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.registry.gauge(f"{name}_{metric_name}", value)
            except Exception:
                pass

    def get_registry(self) -> MetricsRegistry:
        """Get the metrics registry."""
        return self.registry


# Singleton instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def setup_default_metrics() -> MetricsCollector:
    """Setup default metrics collection."""
    collector = get_metrics_collector()

    # KV cache metrics
    def collect_kv_metrics() -> dict[str, Any]:
        return {
            "cache_size_bytes": 0,  # Would be populated from actual cache
            "cache_hit_rate": 0.0,
            "compression_ratio": 0.0,
        }

    # Attention metrics
    def collect_attention_metrics() -> dict[str, Any]:
        return {
            "sparse_ratio": 0.0,
            "fallback_count": 0,
            "avg_latency_ms": 0.0,
        }

    # Memory metrics
    def collect_memory_metrics() -> dict[str, Any]:
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "memory_rss_bytes": memory_info.rss,
                "memory_vms_bytes": memory_info.vms,
            }
        except Exception:
            return {"memory_rss_bytes": 0, "memory_vms_bytes": 0}

    collector.register_collector("kv_cache", collect_kv_metrics)
    collector.register_collector("attention", collect_attention_metrics)
    collector.register_collector("memory", collect_memory_metrics)

    return collector
