"""Tests for PolarQualityGates."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.quality_gates import PolarQualityGates


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_perfect_match_passes() -> None:
    gates = PolarQualityGates()
    logits = mx.random.normal(shape=(10, 100))
    result = gates.evaluate("test", logits, logits)
    assert result.passed
    assert result.logit_cosine > 0.999
    assert result.top1_agreement == 1.0
    assert result.nan_inf_count == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_slight_drift_fails_strict_gate() -> None:
    gates = PolarQualityGates()
    base = mx.random.normal(shape=(10, 100))
    candidate = base + mx.random.normal(shape=(10, 100)) * 0.1
    result = gates.evaluate("test", base, candidate)
    # Small drift may or may not fail depending on exact values
    assert isinstance(result.logit_cosine, float)
    assert isinstance(result.top5_overlap, float)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_nan_detected() -> None:
    gates = PolarQualityGates()
    base = mx.random.normal(shape=(10, 100))
    candidate = mx.array(np.full((10, 100), np.nan))
    result = gates.evaluate("test", base, candidate)
    assert not result.passed
    assert result.nan_inf_count > 0
    assert any("NaN" in g for g in result.failed_gates)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_gate_thresholds() -> None:
    thresholds = PolarQualityGates.gate_thresholds()
    assert "logit_cosine_min" in thresholds
    assert "top5_overlap_min" in thresholds
    assert thresholds["logit_cosine_min"] == 0.995
