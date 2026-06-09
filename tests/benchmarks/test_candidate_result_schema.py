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
        passed_quality_gate=False,
    )
    assert r.name == "test"
    assert r.model_id == "model"
    assert r.prompt == "hello"
    assert r.passed_quality_gate is False
    assert r.error == ""
    assert r.tokens_per_sec is None
    assert r.size_ratio is None
    assert r.compression_factor is None
    assert r.notes == ""


@pytest.mark.unit
def test_candidate_result_with_metrics():
    """CandidateResult accepts metric fields."""
    from rfsn_v11.candidates.base import CandidateResult

    r = CandidateResult(
        name="rfsn_v10_k8_v5_gs32",
        model_id="qwen2",
        prompt="hello",
        passed_quality_gate=True,
        tokens_per_sec=85.3,
        size_ratio=0.25,
        compression_factor=4.0,
        logit_cosine=0.9995,
        notes="baseline test",
    )
    assert r.tokens_per_sec == pytest.approx(85.3)
    assert r.size_ratio == pytest.approx(0.25)
    assert r.compression_factor == pytest.approx(4.0)
    assert r.passed_quality_gate is True
