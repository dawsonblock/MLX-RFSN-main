#!/usr/bin/env python3
"""RFSN v10 — Adaptive batch sizing tests.

Covers batch size adjustment, history limits, performance prediction,
and recommendation logic without requiring MLX.
"""
from __future__ import annotations

from rfsn_v10.adaptive_batch import (
    AdaptiveBatchConfig,
    AdaptiveBatchSizer,
    PerformancePredictor,
)

# ------------------------------------------------------------------
# AdaptiveBatchConfig
# ------------------------------------------------------------------

class TestAdaptiveBatchConfig:
    def test_default_values(self):
        cfg = AdaptiveBatchConfig()
        assert cfg.min_batch_size == 1
        assert cfg.max_batch_size == 128
        assert cfg.initial_batch_size == 32
        assert cfg.target_latency_ms == 10.0
        assert cfg.latency_tolerance == 2.0
        assert cfg.adjustment_factor == 1.2

    def test_custom_values(self):
        cfg = AdaptiveBatchConfig(
            min_batch_size=8,
            max_batch_size=64,
            initial_batch_size=16,
            target_latency_ms=5.0,
        )
        assert cfg.min_batch_size == 8
        assert cfg.max_batch_size == 64
        assert cfg.initial_batch_size == 16


# ------------------------------------------------------------------
# AdaptiveBatchSizer
# ------------------------------------------------------------------

class TestAdaptiveBatchSizer:
    def test_initial_batch_size(self):
        sizer = AdaptiveBatchSizer()
        assert sizer.get_batch_size() == 32

    def test_update_with_high_latency_reduces_batch(self):
        sizer = AdaptiveBatchSizer(
            AdaptiveBatchConfig(
                initial_batch_size=64, target_latency_ms=10.0,
                latency_tolerance=1.0, adjustment_factor=2.0,
            )
        )
        new_size = sizer.update(20.0)  # High latency
        assert new_size < 64
        assert new_size >= 1

    def test_update_with_low_latency_increases_batch(self):
        sizer = AdaptiveBatchSizer(
            AdaptiveBatchConfig(
                initial_batch_size=16, target_latency_ms=10.0,
                latency_tolerance=1.0, adjustment_factor=2.0, max_batch_size=128,
            )
        )
        new_size = sizer.update(1.0)  # Low latency
        assert new_size > 16
        assert new_size <= 128

    def test_update_respects_min_max(self):
        config = AdaptiveBatchConfig(
            min_batch_size=4, max_batch_size=8, initial_batch_size=8,
            target_latency_ms=10.0, latency_tolerance=0.5, adjustment_factor=2.0,
        )
        sizer = AdaptiveBatchSizer(config)
        # High latency should not go below min
        sizer.update(100.0)
        assert sizer.get_batch_size() == 4
        # Low latency should not go above max
        sizer = AdaptiveBatchSizer(config)
        sizer.update(0.1)
        assert sizer.get_batch_size() == 8

    def test_history_limited_to_10(self):
        sizer = AdaptiveBatchSizer()
        for _ in range(15):
            sizer.update(5.0)
        assert len(sizer.latency_history) <= 10
        assert len(sizer.batch_history) <= 10

    def test_get_stats(self):
        sizer = AdaptiveBatchSizer()
        sizer.update(5.0)
        sizer.update(6.0)
        stats = sizer.get_stats()
        assert stats["current_batch_size"] == sizer.get_batch_size()
        assert len(stats["latency_history"]) == 2
        assert stats["avg_latency_ms"] == 5.5

    def test_reset(self):
        sizer = AdaptiveBatchSizer()
        sizer.update(5.0)
        sizer.reset()
        assert sizer.get_batch_size() == 32
        assert sizer.latency_history == []
        assert sizer.batch_history == []

    def test_avg_latency_with_few_samples(self):
        sizer = AdaptiveBatchSizer()
        sizer.update(15.0)
        # With only 1 sample, avg is the input itself
        assert sizer.latency_history == [15.0]


# ------------------------------------------------------------------
# PerformancePredictor
# ------------------------------------------------------------------

class TestPerformancePredictor:
    def test_empty_history_returns_none(self):
        pred = PerformancePredictor()
        assert pred.predict_latency(10) is None
        assert pred.recommend_batch_size(10.0) is None

    def test_linear_prediction(self):
        pred = PerformancePredictor()
        for bs in range(1, 6):
            pred.record(bs, bs * 2.0)  # latency = 2 * batch_size
        predicted = pred.predict_latency(10)
        assert predicted is not None
        assert predicted > 0
        # For perfect linear relationship, predicted should be close to 20
        assert abs(predicted - 20.0) < 1.0

    def test_recommend_batch_size(self):
        pred = PerformancePredictor()
        for bs in range(1, 11):
            pred.record(bs, bs * 2.0)
        recommended = pred.recommend_batch_size(10.0)
        assert recommended is not None
        assert recommended > 0

    def test_window_size_limit(self):
        pred = PerformancePredictor(window_size=5)
        for i in range(10):
            pred.record(i, float(i))
        assert len(pred.history) == 5

    def test_predict_with_insufficient_data(self):
        pred = PerformancePredictor()
        pred.record(1, 1.0)
        # Only 1 point, not enough for regression
        assert pred.predict_latency(5) is None

    def test_reset(self):
        pred = PerformancePredictor()
        pred.record(1, 1.0)
        pred.reset()
        assert pred.history == []

    def test_predict_latency_non_negative(self):
        pred = PerformancePredictor()
        pred.record(1, 100.0)
        pred.record(2, 50.0)
        predicted = pred.predict_latency(5)
        assert predicted is None or predicted >= 0.0
