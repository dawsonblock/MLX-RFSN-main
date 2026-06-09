#!/usr/bin/env python3
"""RFSN v10 — Metrics collection and export tests.

Covers gauge/counter/histogram recording, Prometheus/JSON export,
and metrics collector registration without external dependencies.
"""
from __future__ import annotations

import json

from rfsn_v10.metrics import (
    Metric,
    MetricsCollector,
    MetricsRegistry,
    get_metrics_collector,
    setup_default_metrics,
)

# ------------------------------------------------------------------
# Metric dataclass
# ------------------------------------------------------------------

class TestMetric:
    def test_creation(self):
        m = Metric(name="test", value=1.0, timestamp=0.0)
        assert m.name == "test"
        assert m.value == 1.0
        assert m.tags == {}
        assert m.metric_type == "gauge"


# ------------------------------------------------------------------
# MetricsRegistry
# ------------------------------------------------------------------

class TestMetricsRegistry:
    def test_gauge(self):
        reg = MetricsRegistry()
        reg.gauge("cpu", 0.5)
        assert reg.gauges["cpu"] == 0.5
        assert len(reg.metrics) == 1

    def test_counter(self):
        reg = MetricsRegistry()
        reg.counter("requests")
        reg.counter("requests", 2)
        assert reg.counters["requests"] == 3.0

    def test_histogram(self):
        reg = MetricsRegistry()
        reg.histogram("latency", 5.0)
        reg.histogram("latency", 10.0)
        assert len(reg.histograms["latency"]) == 2

    def test_histogram_summary(self):
        reg = MetricsRegistry()
        for i in range(1, 11):
            reg.histogram("latency", float(i))
        summary = reg.get_metric_summary()
        hist = summary["histograms"]["latency"]
        assert hist["count"] == 10
        assert hist["min"] == 1.0
        assert hist["max"] == 10.0
        assert hist["mean"] == 5.5
        # p50 index = len//2 = 10//2 = 5, value at index 5 in [1..10] is 6.0
        assert hist["p50"] == 6.0

    def test_export_prometheus(self):
        reg = MetricsRegistry()
        reg.gauge("temp", 37.0)
        reg.counter("visits", 5)
        prom = reg.export_prometheus()
        assert "# TYPE temp gauge" in prom
        assert "temp 37.0" in prom
        assert "# TYPE visits counter" in prom
        assert "visits 5" in prom

    def test_export_json(self):
        reg = MetricsRegistry()
        reg.gauge("x", 1.0)
        js = reg.export_json()
        data = json.loads(js)
        assert data["gauges"]["x"] == 1.0

    def test_reset(self):
        reg = MetricsRegistry()
        reg.gauge("x", 1.0)
        reg.reset()
        assert reg.gauges == {}
        assert reg.metrics == []
        assert reg.histograms == {}

    def test_tags(self):
        reg = MetricsRegistry()
        reg.gauge("temp", 25.0, tags={"zone": "a"})
        assert reg.metrics[0].tags == {"zone": "a"}

    def test_thread_safety_basic(self):
        # Smoke test that lock exists and operations complete
        reg = MetricsRegistry()
        reg.gauge("x", 1.0)
        reg.counter("y")
        reg.histogram("z", 2.0)
        summary = reg.get_metric_summary()
        assert "gauges" in summary
        assert "counters" in summary
        assert "histograms" in summary


# ------------------------------------------------------------------
# MetricsCollector
# ------------------------------------------------------------------

class TestMetricsCollector:
    def test_register_and_collect(self):
        coll = MetricsCollector()
        coll.register_collector("test", lambda: {"metric_a": 42})
        coll.collect_all()
        reg = coll.get_registry()
        assert reg.gauges["test_metric_a"] == 42

    def test_collector_exception_swallowed(self):
        coll = MetricsCollector()
        coll.register_collector("bad", lambda: (_ for _ in ()).throw(ValueError("boom")))
        # Should not raise
        coll.collect_all()

    def test_non_numeric_values_skipped(self):
        coll = MetricsCollector()
        coll.register_collector("test", lambda: {"metric_a": "string_value"})
        coll.collect_all()
        reg = coll.get_registry()
        assert "test_metric_a" not in reg.gauges


# ------------------------------------------------------------------
# Global helpers
# ------------------------------------------------------------------

class TestGlobalHelpers:
    def test_get_metrics_collector_is_same(self):
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_setup_default_metrics_registers_collectors(self):
        coll = setup_default_metrics()
        assert "kv_cache" in coll.collectors
        assert "attention" in coll.collectors
        assert "memory" in coll.collectors
