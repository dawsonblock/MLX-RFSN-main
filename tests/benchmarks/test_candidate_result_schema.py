"""Verify CandidateResult dataclass has required fields and sane defaults."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_candidate_result_has_required_fields():
    """CandidateResult must have all expected fields with sane defaults."""
    import sys
    sys.path.insert(0, ".")
    from rfsn_v11.candidates.base import CandidateResult

    r = CandidateResult(
        name="test",
        model_id="model",
        prompt="hello",
        gate_status="PENDING_LOGIT_GATE",
    )
    assert r.name == "test"
    assert r.model_id == "model"
    assert r.prompt == "hello"
    assert r.gate_status == "PENDING_LOGIT_GATE"
    assert r.error == ""
    assert r.tokens_per_sec is None
    assert r.size_ratio is None
    assert r.compression_factor is None
    assert r.notes == ""
    assert r.logit_gate_passed is None
    assert r.memory_gate_passed is None
    assert r.text_heuristic_passed is None
    assert r.promotion_eligible is False


@pytest.mark.unit
def test_candidate_result_with_metrics():
    """CandidateResult accepts metric fields."""
    from rfsn_v11.candidates.base import CandidateResult

    r = CandidateResult(
        name="rfsn_v10_k8_v5_gs64",
        model_id="qwen2",
        prompt="hello",
        gate_status="PASS",
        promotion_eligible=True,
        tokens_per_sec=85.3,
        size_ratio=0.25,
        compression_factor=4.0,
        logit_cosine=0.9995,
        notes="baseline test",
    )
    assert r.tokens_per_sec == pytest.approx(85.3)
    assert r.size_ratio == pytest.approx(0.25)
    assert r.compression_factor == pytest.approx(4.0)
    assert r.gate_status == "PASS"
    assert r.promotion_eligible is True


@pytest.mark.unit
def test_candidate_result_gate_status_values():
    """Allowed gate_status values are enforced."""
    from rfsn_v11.candidates.base import CandidateResult
    from rfsn_v11.candidates.quality_gates import (
        GATE_STATUS_ERROR,
        GATE_STATUS_FAIL,
        GATE_STATUS_PASS,
        GATE_STATUS_PENDING_LOGIT_GATE,
        GATE_STATUS_PENDING_MEMORY_METRICS,
        GATE_STATUS_PENDING_REAL_CACHE_INJECTION,
    )

    for status in [
        GATE_STATUS_PASS,
        GATE_STATUS_FAIL,
        GATE_STATUS_PENDING_LOGIT_GATE,
        GATE_STATUS_PENDING_MEMORY_METRICS,
        GATE_STATUS_PENDING_REAL_CACHE_INJECTION,
        GATE_STATUS_ERROR,
    ]:
        r = CandidateResult(name="x", model_id="m", prompt="p", gate_status=status)
        assert r.gate_status == status


@pytest.mark.unit
def test_candidate_result_promotion_eligibility_rule():
    """Promotion eligible only when logit and memory gates pass with full metrics."""
    from rfsn_v11.candidates.quality_gates import compute_promotion_eligibility

    # Not eligible without logit gate
    eligible, status = compute_promotion_eligibility(
        logit_gate_passed=None,
        memory_gate_passed=True,
        actual_kv_memory_mb=100.0,
        working_set_memory_mb=120.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    assert eligible is False
    assert status == "PENDING_LOGIT_GATE"

    # Not eligible without memory metrics
    eligible, status = compute_promotion_eligibility(
        logit_gate_passed=True,
        memory_gate_passed=True,
        actual_kv_memory_mb=None,
        working_set_memory_mb=120.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    assert eligible is False
    assert status == "PENDING_MEMORY_METRICS"

    # Eligible when all present
    eligible, status = compute_promotion_eligibility(
        logit_gate_passed=True,
        memory_gate_passed=True,
        actual_kv_memory_mb=100.0,
        working_set_memory_mb=120.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    assert eligible is True
    assert status == "PASS"
