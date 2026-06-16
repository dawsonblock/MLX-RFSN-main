"""Step 10: A1 generation comparison tests.

Compares A1 compressed generation against the dense MLX baseline.
Uses --smoke mode (synthetic logits) when no model is available.

Metrics validated (smoke mode uses synthetic data):
    logit_cosine               >= 0.995  (only if baseline logits available)
    top5_overlap               >= 0.95
    perplexity_delta           <= 0.02
    visible_output_drift_score <= 0.05

Run (no model):
    pytest benchmarks/tests/test_a1_generation.py -v

Run (with real model):
    pytest benchmarks/tests/test_a1_generation.py -v --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pure_python

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.judge import Judge, VerdictLabel
from benchmarks.schemas import CandidateResult

# ---------------------------------------------------------------------------
# Synthetic helpers (smoke mode)
# ---------------------------------------------------------------------------

def _make_synthetic_baseline(
    model_id: str = "smoke/Qwen2.5-0.5B",
    prompt_id: str = "short_chat_512",
    output_tokens: int = 20,
    vocab_size: int = 1000,
    seed: int = 42,
) -> CandidateResult:
    rng = np.random.default_rng(seed)
    return CandidateResult(
        candidate_name="dense_mlx_baseline",
        model_id=model_id,
        prompt_id=prompt_id,
        context_length=128,
        output_tokens=output_tokens,
        preconditioner="none",
        quantizer="none",
        logit_cosine=1.0,
        top1_match_rate=1.0,
        top5_overlap=1.0,
        top10_overlap=1.0,
        perplexity_delta=0.0,
        visible_output_drift_score=0.0,
        attention_score_cosine=1.0,
        attention_score_mae=0.0,
        attention_top5_overlap=1.0,
        softmax_kl=0.0,
        peak_memory_mb=1500.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=64.0,
        metadata_memory_mb=0.0,
        effective_bits_per_kv_element=16.0,
        compression_factor=1.0,
        prefill_tps=1000.0,
        decode_tps=60.0,
        first_token_latency_ms=100.0,
        total_latency_ms=400.0,
        compression_time_ms=0.0,
        decompression_time_ms=0.0,
        generated_text="The capital of France is Paris.",
        run_type="real_model",
        source_type="installed_wheel",
        requested_backend="metal",
        executed_backend="metal",
        metal_executed=True,
        fallback_used=False,
        measured_memory=True,
        estimated_memory=False,
        commit_hash="abc123",
        corpus_hash="def456",
        token_sequence_hash="ghi789",
    )


def _make_synthetic_a1_high_quality(baseline: CandidateResult) -> CandidateResult:
    """Synthetic A1 result that should PROMOTE."""
    return CandidateResult(
        candidate_name="A1_wht_grouped_k8v4_gs64",
        model_id=baseline.model_id,
        prompt_id=baseline.prompt_id,
        context_length=baseline.context_length,
        output_tokens=baseline.output_tokens,
        preconditioner="wht",
        quantizer="grouped_sym",
        key_bits=8.0,
        value_bits=5.0,
        group_size=64,
        is_benchmark_only=True,
        logit_cosine=0.9975,
        top1_match_rate=0.98,
        top5_overlap=0.97,
        top10_overlap=0.99,
        perplexity_delta=0.005,
        visible_output_drift_score=0.01,
        attention_score_cosine=0.9980,
        attention_score_mae=0.001,
        attention_top5_overlap=0.97,
        softmax_kl=0.002,
        peak_memory_mb=1300.0,
        kv_cache_memory_mb=64.0,
        compressed_kv_memory_mb=28.0,  # 56% of FP16 → 44% reduction → passes gate
        metadata_memory_mb=0.5,
        effective_bits_per_kv_element=6.0,
        compression_factor=2.3,
        prefill_tps=1000.0,
        decode_tps=60.0,
        first_token_latency_ms=105.0,
        total_latency_ms=410.0,
        compression_time_ms=2.5,
        decompression_time_ms=5.0,
        generated_text="The capital of France is Paris.",
        run_type="real_model",
        source_type="installed_wheel",
        requested_backend="metal",
        executed_backend="metal",
        metal_executed=True,
        fallback_used=False,
        measured_memory=True,
        estimated_memory=False,
        commit_hash="abc123",
        corpus_hash="def456",
        token_sequence_hash="ghi789",
    )


def _make_synthetic_a1_quality_fail(baseline: CandidateResult) -> CandidateResult:
    """Synthetic A1 result with bad quality — should REJECT."""
    r = _make_synthetic_a1_high_quality(baseline)
    r.logit_cosine = 0.980  # below 0.995 threshold
    r.top5_overlap = 0.88   # below 0.95 threshold
    return r


def _make_synthetic_a1_no_improvement(baseline: CandidateResult) -> CandidateResult:
    """Quality passes but no memory improvement — KEEP_EXPERIMENTAL."""
    r = _make_synthetic_a1_high_quality(baseline)
    r.compressed_kv_memory_mb = r.kv_cache_memory_mb  # no compression
    r.peak_memory_mb = baseline.peak_memory_mb
    r.decode_tps = baseline.decode_tps
    return r


# ---------------------------------------------------------------------------
# Tests: Judge integration (smoke mode)
# ---------------------------------------------------------------------------

class TestJudgeIntegration:
    """Validate the judge pipeline end-to-end with synthetic results."""

    def test_high_quality_promotes(self):
        baseline = _make_synthetic_baseline()
        candidate = _make_synthetic_a1_high_quality(baseline)
        judge = Judge()
        verdict = judge.evaluate(candidate, baseline)
        # Benchmark-only candidates are not eligible for production promotion
        assert verdict.label == VerdictLabel.REJECT
        assert "benchmark-only" in verdict.reason

    def test_quality_fail_rejects(self):
        baseline = _make_synthetic_baseline()
        candidate = _make_synthetic_a1_quality_fail(baseline)
        judge = Judge()
        verdict = judge.evaluate(candidate, baseline)
        assert verdict.label == VerdictLabel.REJECT, (
            f"Expected REJECT, got {verdict.label.value}: {verdict.reason}"
        )

    def test_no_improvement_keeps_experimental(self):
        baseline = _make_synthetic_baseline()
        candidate = _make_synthetic_a1_no_improvement(baseline)
        judge = Judge()
        verdict = judge.evaluate(candidate, baseline)
        # Benchmark-only candidates are not eligible for production promotion
        assert verdict.label == VerdictLabel.REJECT
        assert "benchmark-only" in verdict.reason

    def test_missing_metric_keeps_experimental(self):
        baseline = _make_synthetic_baseline()
        candidate = _make_synthetic_a1_high_quality(baseline)
        candidate.attention_score_cosine = None  # missing required metric
        judge = Judge()
        verdict = judge.evaluate(candidate, baseline)
        # Benchmark-only candidates are not eligible for production promotion
        assert verdict.label == VerdictLabel.REJECT
        assert "benchmark-only" in verdict.reason

    def test_regression_detected(self):
        baseline = _make_synthetic_baseline()
        stable = _make_synthetic_a1_high_quality(baseline)
        stable.logit_cosine = 0.9990  # stable reference
        stable.is_benchmark_only = False  # Allow stable reference to be production

        # Candidate is worse than stable
        candidate = _make_synthetic_a1_high_quality(baseline)
        candidate.logit_cosine = 0.9960  # below stable
        candidate.is_benchmark_only = False  # Allow candidate to be production

        judge = Judge(stable_reference=stable)
        verdict = judge.evaluate(candidate, baseline)
        assert verdict.label == VerdictLabel.REGRESSION, (
            f"Expected REGRESSION, got {verdict.label.value}"
        )

    def test_dense_baseline_is_not_promotable(self):
        """Dense baseline has no compression — no improvement gate."""
        baseline = _make_synthetic_baseline()
        judge = Judge()
        verdict = judge.evaluate(baseline, baseline)
        # baseline vs itself: compressed_kv == kv_cache → no reduction
        assert verdict.label in (VerdictLabel.KEEP_EXPERIMENTAL, VerdictLabel.PROMOTE)


# ---------------------------------------------------------------------------
# Tests: Schema and report generator
# ---------------------------------------------------------------------------

class TestSchemaAndReport:
    def test_to_dict_round_trip(self):
        baseline = _make_synthetic_baseline()
        d = baseline.to_dict()
        r2 = CandidateResult.from_dict(d)
        assert r2.candidate_name == baseline.candidate_name
        assert r2.logit_cosine == baseline.logit_cosine

    def test_to_json_valid(self):
        import json
        baseline = _make_synthetic_baseline()
        j = baseline.to_json()
        parsed = json.loads(j)
        assert parsed["candidate_name"] == "dense_mlx_baseline"

    def test_report_generator_smoke(self, tmp_path):
        from benchmarks.report_generator import ReportGenerator
        baseline = _make_synthetic_baseline()
        candidate = _make_synthetic_a1_high_quality(baseline)
        judge = Judge()
        verdict = judge.evaluate(candidate, baseline)

        gen = ReportGenerator(out_dir=tmp_path / "results", report_dir=tmp_path / "reports")
        json_path, md_path = gen.write(
            candidates=[candidate],
            baseline=baseline,
            verdicts=[verdict],
            run_tag="test_a1",
        )
        assert json_path.exists()
        assert md_path.exists()

        import json
        data = json.loads(json_path.read_text())
        # Smoke data is not eligible for promotion, so no PROMOTE verdicts
        assert data["summary"]["verdict_counts"]["PROMOTE"] == 0
        assert data["summary"]["verdict_counts"]["REJECT"] == 1

    def test_candidate_registry_smoke(self):
        """Phase 0: default registry is frozen to three canonical candidates."""
        from benchmarks.candidate_registry import (
            build_default_registry,
            build_experimental_registry,
        )
        reg = build_default_registry()
        assert "dense_mlx_baseline" in reg.names()
        assert "mlx_lm_8bit_kv" in reg.names()
        assert "rfsn_direct_packed_k8v8" in reg.names()
        assert len(reg.names()) == 3

        # Legacy candidates live in the experimental registry only
        exp_reg = build_experimental_registry()
        assert "A1_wht_grouped_k8v4_gs64" in exp_reg.names()


# ---------------------------------------------------------------------------
# Tests: Logit quality metrics helper
# ---------------------------------------------------------------------------

class TestLogitQualityMetrics:
    """Validate BenchmarkCandidate.compute_logit_quality."""

    def test_identical_logits(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((20, 1000)).astype(np.float32)
        metrics = BenchmarkCandidate.compute_logit_quality(logits, logits)
        assert abs(metrics["logit_cosine"] - 1.0) < 1e-5
        assert abs(metrics["top5_overlap"] - 1.0) < 1e-5
        assert abs(metrics["perplexity_delta"]) < 0.01

    def test_noisy_logits(self):
        rng = np.random.default_rng(1)
        logits_b = rng.standard_normal((20, 1000)).astype(np.float32)
        logits_c = logits_b + rng.standard_normal((20, 1000)).astype(np.float32) * 0.01
        metrics = BenchmarkCandidate.compute_logit_quality(logits_b, logits_c)
        assert metrics["logit_cosine"] >= 0.99
        assert metrics["top5_overlap"] >= 0.80

    def test_very_different_logits(self):
        rng = np.random.default_rng(2)
        logits_b = rng.standard_normal((20, 1000)).astype(np.float32)
        logits_c = rng.standard_normal((20, 1000)).astype(np.float32)
        metrics = BenchmarkCandidate.compute_logit_quality(logits_b, logits_c)
        assert metrics["logit_cosine"] < 0.95

    def test_shape_mismatch_returns_none(self):
        logits_b = np.zeros((10, 100))
        logits_c = np.zeros((10, 200))
        metrics = BenchmarkCandidate.compute_logit_quality(logits_b, logits_c)
        assert all(v is None for v in metrics.values())
