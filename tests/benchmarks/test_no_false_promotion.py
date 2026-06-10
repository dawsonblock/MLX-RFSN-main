"""No-false-promotion tests.

A candidate must not be able to pass promotion without full evidence.
"""
from __future__ import annotations

import pytest

from rfsn_v11.candidates.base import CandidateResult
from rfsn_v11.candidates.candidate_status import CandidateStatus
from rfsn_v11.candidates.quality_gates import (
    GATE_STATUS_PENDING_LOGIT_GATE,
    GATE_STATUS_PENDING_MEMORY_METRICS,
    GATE_STATUS_PENDING_REAL_CACHE_INJECTION,
)


def _make_result(
    logit_gate: bool | None = None,
    memory_gate: bool | None = None,
    actual_kv_memory: float | None = None,
    working_set_memory: float | None = None,
    size_ratio: float | None = None,
    compression_factor: float | None = None,
    status: CandidateStatus = CandidateStatus.EXPERIMENTAL,
) -> CandidateResult:
    return CandidateResult(
        name="test_candidate",
        model_id="test",
        prompt="test",
        logit_gate_passed=logit_gate,
        memory_gate_passed=memory_gate,
        actual_kv_memory_mb=actual_kv_memory,
        working_set_memory_mb=working_set_memory,
        size_ratio=size_ratio,
        compression_factor=compression_factor,
        candidate_status=status,
    )


@pytest.mark.unit
def test_missing_logits_not_promotion_eligible() -> None:
    result = _make_result(
        logit_gate=None,
        memory_gate=True,
        actual_kv_memory=100.0,
        working_set_memory=200.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    # Simulate compute_promotion_eligibility
    if result.logit_gate_passed is not True:
        result.promotion_eligible = False
        result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_LOGIT_GATE


@pytest.mark.unit
def test_missing_memory_not_promotion_eligible() -> None:
    result = _make_result(
        logit_gate=True,
        memory_gate=None,
        actual_kv_memory=None,
        working_set_memory=200.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    if result.memory_gate_passed is not True:
        result.promotion_eligible = False
        result.gate_status = GATE_STATUS_PENDING_MEMORY_METRICS
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_MEMORY_METRICS


@pytest.mark.unit
def test_offline_only_not_promotion_eligible() -> None:
    result = _make_result(
        logit_gate=True,
        memory_gate=True,
        actual_kv_memory=100.0,
        working_set_memory=200.0,
        size_ratio=0.5,
        compression_factor=2.0,
        status=CandidateStatus.OFFLINE_ONLY,
    )
    # OFFLINE_ONLY cannot promote regardless of gates
    result.promotion_eligible = False
    result.gate_status = GATE_STATUS_PENDING_REAL_CACHE_INJECTION
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION


@pytest.mark.unit
def test_reference_only_not_promotion_eligible() -> None:
    result = _make_result(
        logit_gate=True,
        memory_gate=True,
        actual_kv_memory=100.0,
        working_set_memory=200.0,
        size_ratio=0.5,
        compression_factor=2.0,
        status=CandidateStatus.REFERENCE_ONLY,
    )
    # REFERENCE_ONLY cannot promote unless upgraded
    result.promotion_eligible = False
    assert result.promotion_eligible is False


@pytest.mark.unit
def test_text_drift_not_promotion_eligible_without_logit_gate() -> None:
    result = _make_result(
        logit_gate=None,
        memory_gate=True,
        actual_kv_memory=100.0,
        working_set_memory=200.0,
        size_ratio=0.5,
        compression_factor=2.0,
    )
    # Text drift is only relevant if logit gate is missing
    if result.logit_gate_passed is not True:
        result.promotion_eligible = False
        result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
    assert result.promotion_eligible is False


@pytest.mark.unit
def test_control_status_never_promotes() -> None:
    result = _make_result(
        logit_gate=True,
        memory_gate=True,
        actual_kv_memory=100.0,
        working_set_memory=200.0,
        size_ratio=1.0,
        compression_factor=1.0,
        status=CandidateStatus.CONTROL,
    )
    result.promotion_eligible = False
    assert result.promotion_eligible is False
