"""Tests for PromotionEngine."""
from __future__ import annotations

from rfsn_v11.polar_fused.promotion import PromotionEngine, PromotionLevel


def test_all_criteria_met_default() -> None:
    engine = PromotionEngine()
    evidence = {c: True for c in engine.CRITERIA}
    evidence["faster_above_threshold"] = {"pass": True}
    result = engine.evaluate("polar_k4v4", evidence)
    assert result.can_promote
    assert result.level == PromotionLevel.DEFAULT
    assert len(result.missing_criteria) == 0


def test_all_criteria_met_not_faster() -> None:
    engine = PromotionEngine()
    evidence = {c: True for c in engine.CRITERIA}
    result = engine.evaluate("polar_k4v4", evidence)
    assert result.can_promote
    assert result.level == PromotionLevel.SUPPORTED


def test_quality_passes_but_not_all_criteria() -> None:
    engine = PromotionEngine()
    evidence = {"quality_gates_pass": True}
    result = engine.evaluate("polar_k4v4", evidence)
    assert not result.can_promote
    assert result.level == PromotionLevel.CANDIDATE
    assert len(result.completed_criteria) == 1
    assert len(result.missing_criteria) == len(engine.CRITERIA) - 1


def test_nothing_passes_lab() -> None:
    engine = PromotionEngine()
    evidence = {}
    result = engine.evaluate("polar_k4v4", evidence)
    assert not result.can_promote
    assert result.level == PromotionLevel.LAB


def test_status_tracking() -> None:
    engine = PromotionEngine()
    evidence = {"correctness_tests_pass": True}
    result = engine.evaluate("polar_k4v4", evidence)
    assert engine.get_status("polar_k4v4") == result
    assert "polar_k4v4" in engine.all_statuses()
