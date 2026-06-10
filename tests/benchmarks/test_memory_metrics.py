"""Tests for rfsn_v11/candidates/memory_metrics.py."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_bytes_to_mb():
    from rfsn_v11.candidates.memory_metrics import bytes_to_mb
    assert bytes_to_mb(1024 * 1024) == pytest.approx(1.0)
    assert bytes_to_mb(2 * 1024 * 1024) == pytest.approx(2.0)


@pytest.mark.unit
def test_compression_factor():
    from rfsn_v11.candidates.memory_metrics import compression_factor
    assert compression_factor(1024, 256) == pytest.approx(4.0)
    assert compression_factor(1024, 1024) == pytest.approx(1.0)


@pytest.mark.unit
def test_compression_factor_zero_raises():
    from rfsn_v11.candidates.memory_metrics import compression_factor
    with pytest.raises(ValueError):
        compression_factor(1024, 0)
    with pytest.raises(ValueError):
        compression_factor(1024, -1)


@pytest.mark.unit
def test_size_ratio():
    from rfsn_v11.candidates.memory_metrics import size_ratio
    assert size_ratio(1024, 256) == pytest.approx(0.25)
    assert size_ratio(1024, 1024) == pytest.approx(1.0)


@pytest.mark.unit
def test_size_ratio_zero_raises():
    from rfsn_v11.candidates.memory_metrics import size_ratio
    with pytest.raises(ValueError):
        size_ratio(0, 256)
    with pytest.raises(ValueError):
        size_ratio(-1, 256)
