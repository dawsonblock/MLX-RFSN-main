#!/usr/bin/env python3
"""Proof artifact trend comparison and regression gating utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "kv": {
        "latency_max_regression_pct": {
            "store_latency_ms": 200.0,
            "retrieve_latency_ms": 200.0,
        },
        "quality_max_drop_abs": {
            "key_cosine_sim": 0.008,
            "value_cosine_sim": 0.02,
        },
        "quality_max_rise_abs": {
            "key_rel_mae": 0.02,
            "value_rel_mae": 0.06,
            "key_max_abs_error": 0.04,
            "value_max_abs_error": 0.20,
        },
    },
    "e2e": {
        "latency_max_regression_pct": {
            "cache_miss_total_latency_ms": 30.0,
            "cache_hit_total_latency_ms": 30.0,
        },
        "quality_max_drop_abs": {
            "quant_audit_cosine": 0.015,
            "sparse_audit_cosine": 0.03,
            "audit_cosine": 0.03,
        },
    },
    "absolute_quality_min": {
        "absolute_sparse_audit_cosine_min": 0.90,
        "absolute_quant_audit_cosine_min": 0.95,
        "absolute_value_cosine_min": 0.90,
    },
}


@dataclass
class Breach:
    section: str
    scenario: str
    metric: str
    rule: str
    threshold: float
    baseline: Optional[float]
    current: Optional[float]
    observed: Optional[float]
    details: str


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_thresholds(override: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not override:
        return DEFAULT_THRESHOLDS
    return _deep_merge_dict(DEFAULT_THRESHOLDS, override)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds_file(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return DEFAULT_THRESHOLDS
    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")
    return merge_thresholds(load_json(path))


def _get_float(run: dict[str, Any], metric: str) -> Optional[float]:
    value = run.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_by_scenario(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for run in payload.get("runs", []):
        scenario = str(run.get("scenario", ""))
        if scenario:
            indexed[scenario] = run
    return indexed


def _apply_latency_rule(
    section: str,
    scenario: str,
    metric: str,
    threshold: float,
    baseline_val: Optional[float],
    current_val: Optional[float],
    breaches: list[Breach],
) -> Optional[float]:
    if baseline_val is None or current_val is None or baseline_val <= 0.0:
        return None
    regression_pct = ((current_val - baseline_val) / baseline_val) * 100.0
    if regression_pct > threshold:
        breaches.append(
            Breach(
                section=section,
                scenario=scenario,
                metric=metric,
                rule="latency_max_regression_pct",
                threshold=threshold,
                baseline=baseline_val,
                current=current_val,
                observed=regression_pct,
                details=(
                    f"{metric} regressed by {regression_pct:.2f}% "
                    f"(threshold {threshold:.2f}%)"
                ),
            )
        )
    return regression_pct


def _apply_drop_rule(
    section: str,
    scenario: str,
    metric: str,
    threshold: float,
    baseline_val: Optional[float],
    current_val: Optional[float],
    breaches: list[Breach],
) -> Optional[float]:
    if baseline_val is None or current_val is None:
        return None
    drop = baseline_val - current_val
    if drop > threshold:
        breaches.append(
            Breach(
                section=section,
                scenario=scenario,
                metric=metric,
                rule="quality_max_drop_abs",
                threshold=threshold,
                baseline=baseline_val,
                current=current_val,
                observed=drop,
                details=(
                    f"{metric} dropped by {drop:.6f} "
                    f"(threshold {threshold:.6f})"
                ),
            )
        )
    return drop


def _apply_rise_rule(
    section: str,
    scenario: str,
    metric: str,
    threshold: float,
    baseline_val: Optional[float],
    current_val: Optional[float],
    breaches: list[Breach],
) -> Optional[float]:
    if baseline_val is None or current_val is None:
        return None
    rise = current_val - baseline_val
    if rise > threshold:
        breaches.append(
            Breach(
                section=section,
                scenario=scenario,
                metric=metric,
                rule="quality_max_rise_abs",
                threshold=threshold,
                baseline=baseline_val,
                current=current_val,
                observed=rise,
                details=(
                    f"{metric} increased by {rise:.6f} "
                    f"(threshold {threshold:.6f})"
                ),
            )
        )
    return rise


def compare_section(
    section: str,
    baseline_payload: dict[str, Any],
    current_payload: dict[str, Any],
    thresholds: dict[str, Any],
    strict_missing: bool,
) -> dict[str, Any]:
    baseline_runs = _index_by_scenario(baseline_payload)
    current_runs = _index_by_scenario(current_payload)

    baseline_scenarios = set(baseline_runs.keys())
    current_scenarios = set(current_runs.keys())

    missing_scenarios = sorted(baseline_scenarios - current_scenarios)
    extra_scenarios = sorted(current_scenarios - baseline_scenarios)

    breaches: list[Breach] = []
    if strict_missing:
        for scenario in missing_scenarios:
            breaches.append(
                Breach(
                    section=section,
                    scenario=scenario,
                    metric="scenario",
                    rule="missing_scenario",
                    threshold=0.0,
                    baseline=None,
                    current=None,
                    observed=None,
                    details="Scenario missing from current proof run",
                )
            )

    metric_deltas: list[dict[str, Any]] = []

    section_thresholds = thresholds.get(section, {})
    latency_thresholds = section_thresholds.get("latency_max_regression_pct", {})
    drop_thresholds = section_thresholds.get("quality_max_drop_abs", {})
    rise_thresholds = section_thresholds.get("quality_max_rise_abs", {})

    comparable_scenarios = sorted(baseline_scenarios & current_scenarios)
    for scenario in comparable_scenarios:
        baseline_run = baseline_runs[scenario]
        current_run = current_runs[scenario]

        for metric, threshold in latency_thresholds.items():
            baseline_val = _get_float(baseline_run, metric)
            current_val = _get_float(current_run, metric)
            observed = _apply_latency_rule(
                section,
                scenario,
                metric,
                float(threshold),
                baseline_val,
                current_val,
                breaches,
            )
            metric_deltas.append(
                {
                    "section": section,
                    "scenario": scenario,
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "observed": observed,
                    "unit": "pct",
                    "rule": "latency_max_regression_pct",
                    "threshold": float(threshold),
                }
            )

        for metric, threshold in drop_thresholds.items():
            baseline_val = _get_float(baseline_run, metric)
            current_val = _get_float(current_run, metric)
            observed = _apply_drop_rule(
                section,
                scenario,
                metric,
                float(threshold),
                baseline_val,
                current_val,
                breaches,
            )
            metric_deltas.append(
                {
                    "section": section,
                    "scenario": scenario,
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "observed": observed,
                    "unit": "abs",
                    "rule": "quality_max_drop_abs",
                    "threshold": float(threshold),
                }
            )

        for metric, threshold in rise_thresholds.items():
            baseline_val = _get_float(baseline_run, metric)
            current_val = _get_float(current_run, metric)
            observed = _apply_rise_rule(
                section,
                scenario,
                metric,
                float(threshold),
                baseline_val,
                current_val,
                breaches,
            )
            metric_deltas.append(
                {
                    "section": section,
                    "scenario": scenario,
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "observed": observed,
                    "unit": "abs",
                    "rule": "quality_max_rise_abs",
                    "threshold": float(threshold),
                }
            )

    return {
        "compared_scenarios": comparable_scenarios,
        "missing_scenarios": missing_scenarios,
        "extra_scenarios": extra_scenarios,
        "metric_deltas": metric_deltas,
        "breaches": [breach.__dict__ for breach in breaches],
    }


def compare_proof_dirs(
    baseline_dir: Path,
    current_dir: Path,
    thresholds: dict[str, Any],
    strict_missing: bool = True,
    strict_absolute: bool = False,
) -> dict[str, Any]:
    baseline_kv = load_json(baseline_dir / "kv_cache_runs.json")
    baseline_e2e = load_json(baseline_dir / "e2e_scenarios.json")
    current_kv = load_json(current_dir / "kv_cache_runs.json")
    current_e2e = load_json(current_dir / "e2e_scenarios.json")

    kv_section = compare_section("kv", baseline_kv, current_kv, thresholds, strict_missing)
    e2e_section = compare_section("e2e", baseline_e2e, current_e2e, thresholds, strict_missing)

    absolute_cfg = thresholds.get("absolute_quality_min", {})
    absolute_checks = [
        (
            "absolute_sparse_audit_cosine_min",
            current_e2e,
            "sparse_audit_cosine",
            "e2e",
            "Sparse quality",
        ),
        (
            "absolute_quant_audit_cosine_min",
            current_e2e,
            "quant_audit_cosine",
            "e2e",
            "Quant quality",
        ),
        (
            "absolute_value_cosine_min",
            current_kv,
            "value_cosine_sim",
            "kv",
            "KV value quality",
        ),
    ]

    absolute_quality: dict[str, Any] = {}
    absolute_warnings: list[dict[str, Any]] = []
    absolute_breaches: list[dict[str, Any]] = []

    for key, payload, metric, section, label in absolute_checks:
        threshold = absolute_cfg.get(key)
        if threshold is None:
            continue

        values = [
            _get_float(run, metric)
            for run in payload.get("runs", [])
            if _get_float(run, metric) is not None
        ]
        if not values:
            absolute_quality[key] = {
                "label": label,
                "metric": metric,
                "threshold": float(threshold),
                "status": "unknown",
                "min_value": None,
            }
            continue

        min_value = min(values)
        status = "pass" if min_value >= float(threshold) else "warn"
        absolute_quality[key] = {
            "label": label,
            "metric": metric,
            "threshold": float(threshold),
            "status": status,
            "min_value": min_value,
        }

        if status == "warn":
            warning = {
                "section": section,
                "metric": metric,
                "rule": "absolute_quality_min",
                "threshold": float(threshold),
                "observed": min_value,
                "details": (
                    f"{label} below absolute minimum: "
                    f"{metric} min {min_value:.6f} < {float(threshold):.6f}"
                ),
            }
            absolute_warnings.append(warning)
            if strict_absolute:
                absolute_breaches.append(warning)

    total_breaches = (
        len(kv_section["breaches"]) +
        len(e2e_section["breaches"]) +
        (len(absolute_breaches) if strict_absolute else 0)
    )
    return {
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "thresholds": thresholds,
        "strict_missing": strict_missing,
        "strict_absolute": strict_absolute,
        "absolute_quality": absolute_quality,
        "absolute_warnings": absolute_warnings,
        "absolute_breaches": absolute_breaches,
        "warning_unsafe_for_llm_deployment": len(absolute_warnings) > 0,
        "sections": {
            "kv": kv_section,
            "e2e": e2e_section,
        },
        "total_breaches": total_breaches,
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Proof Regression Report",
        "",
        f"- Baseline: {report['baseline_dir']}",
        f"- Current: {report['current_dir']}",
        f"- Strict missing scenarios: {report['strict_missing']}",
        f"- Strict absolute minima: {report.get('strict_absolute', False)}",
        f"- Total breaches: {report['total_breaches']}",
        "",
    ]

    absolute_quality = report.get("absolute_quality", {})
    if absolute_quality:
        lines.extend(["## Absolute Quality", ""])
        for key, item in absolute_quality.items():
            min_value = item.get("min_value")
            min_str = "n/a" if min_value is None else f"{min_value:.6f}"
            lines.append(
                f"- {item['label']}: {item['status']} "
                f"(min={min_str}, threshold={item['threshold']:.6f})"
            )
        if report.get("warning_unsafe_for_llm_deployment"):
            lines.append("- WARNING_UNSAFE_FOR_LLM_DEPLOYMENT")
        lines.append("")

    absolute_breaches = report.get("absolute_breaches", [])
    if absolute_breaches:
        lines.extend(["## Absolute Quality Breaches", ""])
        for breach in absolute_breaches:
            lines.append(
                f"- {breach['metric']} | {breach['details']}"
            )
        lines.append("")

    for section_name, section in report["sections"].items():
        lines.extend(
            [
                f"## Section: {section_name}",
                f"- Compared scenarios: {len(section['compared_scenarios'])}",
                f"- Missing scenarios: {len(section['missing_scenarios'])}",
                f"- Extra scenarios: {len(section['extra_scenarios'])}",
                f"- Breaches: {len(section['breaches'])}",
                "",
            ]
        )

        if section["missing_scenarios"]:
            lines.append("### Missing Scenarios")
            for scenario in section["missing_scenarios"]:
                lines.append(f"- {scenario}")
            lines.append("")

        if section["breaches"]:
            lines.append("### Breaches")
            for breach in section["breaches"]:
                lines.append(
                    "- "
                    f"{breach['scenario']} | {breach['metric']} | {breach['details']}"
                )
            lines.append("")

    if report["total_breaches"] == 0:
        lines.append("All configured drift checks passed.")

    return "\n".join(lines) + "\n"
