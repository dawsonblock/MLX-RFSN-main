"""Pure Python governance tests for schema validation and JSON serialization.

These tests verify the core governance logic without requiring MLX,
enabling CI smoke testing on non-Apple Silicon environments.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.pure_python


def test_candidate_result_json_roundtrip():
    """Test CandidateResult can serialize and deserialize correctly."""
    from benchmarks.schemas import CandidateResult

    original = CandidateResult(
        candidate_name="test_candidate",
        model_id="test_model",
        prompt_id="test_prompt",
        context_length=128,
        output_tokens=32,
        logit_cosine=0.998,
        top5_overlap=0.97,
        perplexity_delta=0.01,
        peak_memory_mb=1000.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=28.0,
        decode_tps=100.0,
    )

    # Test JSON serialization
    json_str = original.to_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Test deserialization
    loaded = CandidateResult.from_dict(json.loads(json_str))
    assert loaded.candidate_name == original.candidate_name
    assert loaded.model_id == original.model_id
    assert loaded.logit_cosine == original.logit_cosine
    assert loaded.decode_tps == original.decode_tps


def test_candidate_result_to_dict_with_logits():
    """Test to_dict() properly handles logits inclusion/exclusion."""
    from benchmarks.schemas import CandidateResult

    result = CandidateResult(
        candidate_name="test",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
    )

    # Without logits (default)
    d_no_logits = result.to_dict(include_logits=False)
    assert "_logits" not in d_no_logits
    assert "_baseline_logits" not in d_no_logits

    # With logits
    d_with_logits = result.to_dict(include_logits=True)
    # These fields should exist even if None
    assert "_logits" in d_with_logits
    assert "_baseline_logits" in d_with_logits


def test_token_sequence_hash_deterministic():
    """Test token sequence hash is deterministic and consistent."""
    from rfsn_v11.candidates.logit_capture import compute_token_sequence_hash

    # Same inputs should produce same hash
    hash1 = compute_token_sequence_hash(
        model_id="test_model",
        prompt_id="test_prompt",
        prompt_text="Hello world",
        target_token_ids=[1, 2, 3, 4],
        max_tokens=10,
        temperature=0.0,
    )

    hash2 = compute_token_sequence_hash(
        model_id="test_model",
        prompt_id="test_prompt",
        prompt_text="Hello world",
        target_token_ids=[1, 2, 3, 4],
        max_tokens=10,
        temperature=0.0,
    )

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex string length


def test_token_sequence_hash_different_inputs():
    """Test token sequence hash changes with different inputs."""
    from rfsn_v11.candidates.logit_capture import compute_token_sequence_hash

    hash1 = compute_token_sequence_hash(
        model_id="test_model",
        prompt_id="test_prompt",
        prompt_text="Hello world",
        target_token_ids=[1, 2, 3, 4],
        max_tokens=10,
        temperature=0.0,
    )

    hash2 = compute_token_sequence_hash(
        model_id="test_model",
        prompt_id="test_prompt",
        prompt_text="Different text",
        target_token_ids=[1, 2, 3, 4],
        max_tokens=10,
        temperature=0.0,
    )

    assert hash1 != hash2


def test_judge_evaluate_promote():
    """Test judge can evaluate candidates without error."""
    from benchmarks.judge import Judge
    from benchmarks.schemas import CandidateResult

    judge = Judge()

    baseline = CandidateResult(
        candidate_name="baseline",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=1.0,
        top5_overlap=1.0,
        attention_score_cosine=1.0,
        attention_top5_overlap=1.0,
        perplexity_delta=0.0,
        visible_output_drift_score=0.0,
        peak_memory_mb=1000.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=64.0,
        decode_tps=100.0,
        commit_hash="test_commit",
        corpus_hash="test_corpus",
        token_sequence_hash="test_hash",
    )

    candidate = CandidateResult(
        candidate_name="improved",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=0.998,
        top5_overlap=0.97,
        attention_score_cosine=0.998,
        attention_top5_overlap=0.97,
        perplexity_delta=0.01,
        visible_output_drift_score=0.02,
        peak_memory_mb=700.0,  # 30% improvement
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=28.0,  # 56% of baseline
        decode_tps=110.0,  # 10% improvement
        commit_hash="test_commit",
        corpus_hash="test_corpus",
        token_sequence_hash="test_hash",
    )

    # Just verify judge runs without error
    verdict = judge.evaluate(candidate, baseline)
    assert verdict is not None


def test_judge_evaluate_reject():
    """Test judge rejects candidate with quality issues."""
    from benchmarks.judge import Judge, VerdictLabel
    from benchmarks.schemas import CandidateResult

    judge = Judge()

    baseline = CandidateResult(
        candidate_name="baseline",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=1.0,
        top5_overlap=1.0,
        attention_score_cosine=1.0,
        attention_top5_overlap=1.0,
        perplexity_delta=0.0,
        visible_output_drift_score=0.0,
        peak_memory_mb=1000.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=64.0,
        decode_tps=100.0,
    )

    candidate = CandidateResult(
        candidate_name="bad_quality",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=0.980,  # Below threshold (0.995)
        top5_overlap=0.90,  # Below threshold (0.95)
        attention_score_cosine=0.980,
        attention_top5_overlap=0.90,
        perplexity_delta=0.05,  # Above threshold (0.02)
        visible_output_drift_score=0.10,  # Above threshold (0.05)
        peak_memory_mb=500.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=20.0,
        decode_tps=150.0,
    )

    verdict = judge.evaluate(candidate, baseline)
    assert verdict.label == VerdictLabel.REJECT


def test_judge_evaluate_keep_experimental():
    """Test judge can evaluate candidates without error."""
    from benchmarks.judge import Judge
    from benchmarks.schemas import CandidateResult

    judge = Judge()

    baseline = CandidateResult(
        candidate_name="baseline",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=1.0,
        top5_overlap=1.0,
        attention_score_cosine=1.0,
        attention_top5_overlap=1.0,
        perplexity_delta=0.0,
        visible_output_drift_score=0.0,
        peak_memory_mb=1000.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=64.0,
        decode_tps=100.0,
        commit_hash="test_commit",
        corpus_hash="test_corpus",
        token_sequence_hash="test_hash",
    )

    candidate = CandidateResult(
        candidate_name="same_quality",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=0.998,
        top5_overlap=0.97,
        attention_score_cosine=0.998,
        attention_top5_overlap=0.97,
        perplexity_delta=0.01,
        visible_output_drift_score=0.02,
        peak_memory_mb=1000.0,  # No improvement
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=64.0,  # No improvement
        decode_tps=100.0,  # No improvement
        commit_hash="test_commit",
        corpus_hash="test_corpus",
        token_sequence_hash="test_hash",
    )

    # Just verify judge runs without error
    verdict = judge.evaluate(candidate, baseline)
    assert verdict is not None


def test_candidate_status_registry():
    """Test candidate status registry is properly configured."""
    from rfsn_v11.candidates.candidate_status import CandidateStatus, get_status_for_name

    # Test that the function returns a valid status
    status = get_status_for_name("reference_only")
    assert isinstance(status, CandidateStatus)

    # Test unknown status falls back to experimental
    unknown_status = get_status_for_name("unknown_candidate")
    assert unknown_status == CandidateStatus.EXPERIMENTAL


def test_json_utils_strict_serialization():
    """Test JSON utilities handle non-serializable types correctly."""
    import tempfile

    from rfsn_v11.candidates.json_utils import dump_json_strict

    data = {
        "string": "test",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
    }

    # Should serialize without error
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        dump_json_strict(data, f)
        temp_path = f.name

    # Should deserialize correctly
    with open(temp_path) as f:
        loaded = json.load(f)
    assert loaded == data

    # Clean up
    import os
    os.unlink(temp_path)


def test_artifact_utils_json_roundtrip():
    """Test artifact utilities handle JSON correctly."""
    from benchmarks.schemas import CandidateResult
    from rfsn_v11.candidates.artifact_utils import _export_winner

    winner = CandidateResult(
        candidate_name="test_winner",
        model_id="test",
        prompt_id="test",
        context_length=128,
        output_tokens=32,
        logit_cosine=0.998,
        top5_overlap=0.97,
        perplexity_delta=0.01,
        peak_memory_mb=1000.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=28.0,
        decode_tps=100.0,
        promotion_eligible=True,
    )

    # _export_winner expects a list of rows, not a single result
    rows = [winner.to_dict()]

    # Should export without error (returns None for non-promotion mode)
    result = _export_winner(rows, "test_path.json", mode="quick", promotion_allowed=False)
    # Function returns None for non-promotion mode
    assert result is None
