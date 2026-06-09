from __future__ import annotations

from pathlib import Path

from tools.proof_regression import compare_proof_dirs, merge_thresholds


def _write_payload(path: Path, scenario: str, metrics: dict[str, float]) -> None:
    payload = {
        "metadata": {"timestamp": "2026-06-01T00:00:00+00:00"},
        "iterations": 1,
        "runs": [
            {
                "scenario": scenario,
                **metrics,
            }
        ],
    }
    path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_regression_report_has_no_breaches_for_small_drift(tmp_path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_payload(
        baseline_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.970,
            "key_rel_mae": 0.01,
            "value_rel_mae": 0.20,
            "key_max_abs_error": 0.03,
            "value_max_abs_error": 0.50,
        },
    )
    _write_payload(
        current_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 11.0,
            "store_latency_ms": 8.2,
            "key_cosine_sim": 0.998,
            "value_cosine_sim": 0.965,
            "key_rel_mae": 0.02,
            "value_rel_mae": 0.23,
            "key_max_abs_error": 0.05,
            "value_max_abs_error": 0.65,
        },
    )

    _write_payload(
        baseline_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.75,
            "audit_cosine": 0.75,
        },
    )
    _write_payload(
        current_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 23.0,
            "cache_hit_total_latency_ms": 5.4,
            "quant_audit_cosine": 0.965,
            "sparse_audit_cosine": 0.73,
            "audit_cosine": 0.74,
        },
    )

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=merge_thresholds(None),
        strict_missing=True,
    )

    assert report["total_breaches"] == 0


def test_regression_report_detects_latency_and_quality_breach(tmp_path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_payload(
        baseline_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.970,
            "key_rel_mae": 0.01,
            "value_rel_mae": 0.20,
            "key_max_abs_error": 0.03,
            "value_max_abs_error": 0.50,
        },
    )
    _write_payload(
        current_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 20.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.980,
            "value_cosine_sim": 0.920,
            "key_rel_mae": 0.06,
            "value_rel_mae": 0.30,
            "key_max_abs_error": 0.10,
            "value_max_abs_error": 1.10,
        },
    )

    _write_payload(
        baseline_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.75,
            "audit_cosine": 0.75,
        },
    )
    _write_payload(
        current_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 30.0,
            "cache_hit_total_latency_ms": 8.0,
            "quant_audit_cosine": 0.90,
            "sparse_audit_cosine": 0.60,
            "audit_cosine": 0.60,
        },
    )

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=merge_thresholds(None),
        strict_missing=True,
    )

    assert report["total_breaches"] > 0


def test_regression_report_missing_scenario_is_breach_when_strict(tmp_path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_payload(
        baseline_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.970,
            "key_rel_mae": 0.01,
            "value_rel_mae": 0.20,
            "key_max_abs_error": 0.03,
            "value_max_abs_error": 0.50,
        },
    )
    _write_payload(
        current_dir / "kv_cache_runs.json",
        "other_kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.970,
            "key_rel_mae": 0.01,
            "value_rel_mae": 0.20,
            "key_max_abs_error": 0.03,
            "value_max_abs_error": 0.50,
        },
    )

    _write_payload(
        baseline_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.75,
            "audit_cosine": 0.75,
        },
    )
    _write_payload(
        current_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.75,
            "audit_cosine": 0.75,
        },
    )

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=merge_thresholds(None),
        strict_missing=True,
    )

    assert report["total_breaches"] > 0


def test_absolute_quality_min_emits_warning_without_breach(tmp_path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_payload(
        baseline_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.910,
        },
    )
    _write_payload(
        current_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.891,
        },
    )

    _write_payload(
        baseline_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.91,
        },
    )
    _write_payload(
        current_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.89,
        },
    )

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=merge_thresholds(None),
        strict_missing=True,
        strict_absolute=False,
    )

    assert report["warning_unsafe_for_llm_deployment"] is True
    assert len(report["absolute_warnings"]) >= 1
    assert report["total_breaches"] == 0


def test_absolute_quality_min_can_be_strict_breach(tmp_path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_payload(
        baseline_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.910,
        },
    )
    _write_payload(
        current_dir / "kv_cache_runs.json",
        "kv_case",
        {
            "retrieve_latency_ms": 10.0,
            "store_latency_ms": 8.0,
            "key_cosine_sim": 0.999,
            "value_cosine_sim": 0.891,
        },
    )

    _write_payload(
        baseline_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.91,
        },
    )
    _write_payload(
        current_dir / "e2e_scenarios.json",
        "e2e_case",
        {
            "cache_miss_total_latency_ms": 20.0,
            "cache_hit_total_latency_ms": 5.0,
            "quant_audit_cosine": 0.97,
            "sparse_audit_cosine": 0.89,
        },
    )

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=merge_thresholds(None),
        strict_missing=True,
        strict_absolute=True,
    )

    assert report["warning_unsafe_for_llm_deployment"] is True
    assert len(report["absolute_breaches"]) >= 1
    assert report["total_breaches"] >= 1
