#!/usr/bin/env python3
"""
RFSN v10 — Classify experimental configs for promotion
(hardened, real-generation aware).

Reads all available proof artifacts and produces a classification report
according to the plan's Phase 9 promotion rules.

Hard rules:
- No experimental mode can be classified as candidate without
  real-generation data.
- Teacher-forced failure is a hard reject
  (rejected_generation_quality).
- Free-running divergence without teacher-forced failure is marked
  generation_divergence_observed, not rejected.
- QJL remains disabled until its benchmark passes.
- turbo_polar/adaptive/experimental_hybrid are rejected if
  real generation fails.
- If synthetic throughput passes but real generation fails:
  rejected_generation_quality.

Usage:
    python scripts/classify_experimental_configs.py
    python scripts/classify_experimental_configs.py \
        --experimental-dir artifacts/proof/experimental \
        --out artifacts/proof/experimental/config_classification.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def teacher_forced_baseline_is_valid(real_gen: dict[str, Any]) -> bool:
    """Check if baseline_fp16 teacher-forced rows are exact identity."""
    rows = real_gen.get("teacher_forced_logits", [])
    baseline = [
        r for r in rows
        if r.get("config") == "baseline_fp16" and "error" not in r
    ]
    if not baseline:
        return False
    for row in baseline:
        if abs(float(row.get("logit_cosine_vs_fp16", 0.0)) - 1.0) > 1e-7:
            return False
        if abs(float(row.get("top5_overlap_vs_fp16", 0.0)) - 1.0) > 1e-7:
            return False
        if abs(float(row.get("kl_vs_fp16", 999.0))) > 1e-7:
            return False
    return True


def _check_real_generation_teacher_forced(
    real_gen: dict[str, Any], cfg_name: str
) -> tuple[bool, bool]:
    """Return (has_data, teacher_forced_pass) for a config."""
    tf_results = real_gen.get("teacher_forced_logits", [])
    cfg_rows = [
        r for r in tf_results
        if r.get("config") == cfg_name and "error" not in r
    ]
    if not cfg_rows:
        # If only error rows exist, we have data but it failed
        error_rows = [
            r for r in tf_results
            if r.get("config") == cfg_name and "error" in r
        ]
        if error_rows:
            return True, False
        return False, False

    # Check all available prompt lengths;
    # any severe failure rejects the config
    for row in cfg_rows:
        cosine = row.get("logit_cosine_vs_fp16", float("nan"))
        top5 = row.get("top5_overlap_vs_fp16", float("nan"))
        kl = row.get("kl_vs_fp16", float("nan"))
        if not math.isfinite(cosine) or not math.isfinite(top5):
            return True, False
        if cosine < 0.95 or top5 < 0.80 or (
            math.isfinite(kl) and kl > 0.1
        ):
            return True, False
    return True, True


def _check_real_generation_free_running(
    real_gen: dict[str, Any], cfg_name: str
) -> tuple[bool, bool]:
    """Return (has_data, free_running_pass) for a config."""
    fr_results = real_gen.get("free_running_generation", [])
    cfg_rows = [
        r for r in fr_results
        if r.get("config") == cfg_name and "error" not in r
    ]
    if not cfg_rows:
        return False, False
    # Acceptance: exact match rate > 50% on average across prompt lengths
    match_rates = [r.get("exact_token_match_rate", 0.0) for r in cfg_rows]
    avg_match = sum(match_rates) / len(match_rates) if match_rates else 0.0
    return True, avg_match >= 0.5


def _has_free_running_divergence(
    real_gen: dict[str, Any], cfg_name: str
) -> bool:
    """Check if any free-running row shows early divergence."""
    fr_results = real_gen.get("free_running_generation", [])
    cfg_rows = [
        r for r in fr_results
        if r.get("config") == cfg_name and "error" not in r
    ]
    for row in cfg_rows:
        if row.get("first_divergence_position") is not None:
            return True
    return False


def _check_decode_update_trace(
    trace_data: dict[str, Any] | None, cfg_name: str
) -> tuple[bool, bool]:
    """Return (has_data, decode_update_pass) for a config.

    A config passes decode-update if all non-error rows for it have
    status == 'pass' (cosine >= 0.99 and top5 >= 0.80).
    """
    if trace_data is None:
        return False, False
    traces = trace_data.get("traces", [])
    cfg_rows = [
        r for r in traces
        if r.get("config") == cfg_name and "error" not in r
    ]
    if not cfg_rows:
        error_rows = [
            r for r in traces
            if r.get("config") == cfg_name and "error" in r
        ]
        return bool(error_rows), False
    all_pass = all(r.get("status") == "pass" for r in cfg_rows)
    return True, all_pass


def classify_configs(exp_dir: Path) -> dict[str, Any]:
    """Classify configs based on available evidence."""
    qwen_dir = exp_dir / "qwen_1_5b"

    # Load comparison summary
    comp = _load_json(exp_dir / "comparison_summary.json") or {}
    rows = comp.get("rows", [])

    # Load throughput (synthetic)
    throughput = _load_json(exp_dir / "throughput.json") or {}
    tp_results = throughput.get("results", [])
    tp_configs = {c["config"]: c for c in tp_results}
    tp_conclusions = throughput.get("conclusions", {})

    # Load real generation throughput (required for promotion)
    real_gen_tp = _load_json(
        exp_dir / "real_generation_throughput.json"
    ) or {}
    has_real_gen_tp = bool(
        real_gen_tp.get("teacher_forced_logits")
        or real_gen_tp.get("results")
    )

    # Global gate: teacher-forced baseline must be identity
    baseline_valid = teacher_forced_baseline_is_valid(real_gen_tp)

    # Load decode diagnostics
    decode_trace_data = _load_json(exp_dir / "decode_update_trace.json")
    has_decode_diagnostics = (
        decode_trace_data is not None
        and decode_trace_data.get("status") not in {
            "awaiting_execution", "placeholder"
        }
        and bool(decode_trace_data.get("traces"))
    )

    # Load 1.5B validation if available
    qwen_real = _load_json(
        qwen_dir / "real_model_validation.json"
    ) or {}
    qwen_long = _load_json(
        qwen_dir / "long_context_validation.json"
    ) or {}

    classifications: dict[str, str] = {}
    notes: list[str] = []

    for row in rows:
        name = row.get("config", "unknown")
        if name == "baseline_fp16":
            classifications[name] = "reference"
            notes.append(f"{name}: reference — FP16 baseline")
            continue

        is_stable = name.startswith("stable_")

        # Check 0.5B evidence
        pass_512 = row.get("pass_512") == "pass"
        pass_1024 = row.get("pass_1024") == "pass"
        pass_2048 = row.get("pass_2048") == "pass"
        has_mem = row.get("total_compressed_bytes") is not None

        # Real generation gates (strict) — checked for ALL configs
        # Normalize name because real_generation_throughput.json uses
        # bare config names (e.g., "k8_v5_gs64") rather than stable_
        # prefixes.
        real_gen_name = name.replace("stable_", "")
        tf_has, tf_pass = _check_real_generation_teacher_forced(
            real_gen_tp, real_gen_name
        )
        fr_has, fr_pass = _check_real_generation_free_running(
            real_gen_tp, real_gen_name
        )

        # Decode-update diagnostic gate — checked when diagnostics available
        du_has, du_pass = _check_decode_update_trace(
            decode_trace_data, real_gen_name
        )

        if is_stable:
            if not (pass_512 and pass_1024 and pass_2048 and has_mem):
                classifications[name] = "stable_needs_data"
                notes.append(
                    f"{name}: stable_needs_data — missing context"
                )
                continue
            # Stable configs that pass 0.5B still need real-gen validation
            if has_real_gen_tp and not baseline_valid:
                classifications[name] = (
                    "needs_valid_teacher_forced_baseline"
                )
                notes.append(
                    f"{name}: "
                    "needs_valid_teacher_forced_baseline — "
                    "baseline_fp16 identity failed, cannot trust "
                    "comparisons"
                )
                continue
            if tf_has and not tf_pass:
                # Teacher-forced failed; check if decode-update also failed
                decode_suffix = (
                    "_decode_update_issue"
                    if (du_has and not du_pass)
                    else "_under_teacher_forced_drift"
                )
                if name == "stable_k8_v5_gs64":
                    label = "stable_default" + decode_suffix
                    classifications[name] = label
                    notes.append(
                        f"{name}: {label} — "
                        "locked default runtime, teacher-forced drift "
                        "unresolved"
                        + (
                            ", decode-update degraded"
                            if decode_suffix == "_decode_update_issue"
                            else ""
                        )
                    )
                elif name == "stable_k8_v5_gs32":
                    label = "quality_candidate" + decode_suffix
                    classifications[name] = label
                    notes.append(
                        f"{name}: {label} — "
                        "proven at 0.5B, teacher-forced drift unresolved"
                        + (
                            ", decode-update degraded"
                            if decode_suffix == "_decode_update_issue"
                            else ""
                        )
                    )
                else:
                    label = "stable_baseline" + decode_suffix
                    classifications[name] = label
                    notes.append(
                        f"{name}: {label} — "
                        "proven at 0.5B, teacher-forced drift unresolved"
                    )
                continue
            if name == "stable_k8_v5_gs64":
                classifications[name] = "stable_default"
                notes.append(
                    f"{name}: stable_default — locked default runtime"
                )
            elif name == "stable_k8_v5_gs32":
                classifications[name] = "quality_candidate"
                notes.append(
                    f"{name}: quality_candidate — proven at 0.5B"
                )
            else:
                classifications[name] = "stable_baseline"
                notes.append(
                    f"{name}: stable_baseline — proven at 0.5B"
                )
            continue

        # QJL hard disable
        if "qjl" in name.lower():
            classifications[name] = "disabled"
            notes.append(
                f"{name}: disabled — QJL attention score benchmark "
                "failed"
            )
            continue

        # Check 1.5B evidence
        qwen_pass = False
        if qwen_real and qwen_long:
            for cfg in qwen_real.get("configs", []):
                cfg_name = cfg.get("name", "")
                if cfg_name == name or cfg_name == name.replace(
                    "stable_", ""
                ):
                    qwen_pass = cfg.get("status") == "pass"

        # Check throughput
        tp = tp_configs.get(name, {})
        conclusion = tp_conclusions.get(name, {})
        verdict = conclusion.get("verdict", "unknown")
        has_tp = name in tp_configs

        # >8-bit fallback
        uses_raw_uint32 = False
        bits = row.get("cartesian_bits") or row.get("k_bits")
        if bits is not None and bits > 8:
            uses_raw_uint32 = True

        # Free-running divergence check
        has_fr_divergence = _has_free_running_divergence(
            real_gen_tp, real_gen_name
        )

        # Classification logic
        if not (pass_512 and pass_1024 and pass_2048):
            classifications[name] = "rejected_quality"
            notes.append(
                f"{name}: rejected_quality — fails context validation"
            )
        elif not has_mem:
            classifications[name] = "rejected_memory"
            notes.append(
                f"{name}: rejected_memory — missing compression bytes"
            )
        elif not has_tp:
            classifications[name] = "needs_throughput_data"
            notes.append(
                f"{name}: needs_throughput_data — missing throughput "
                "benchmark"
            )
        elif uses_raw_uint32:
            classifications[name] = "rejected_memory"
            notes.append(
                f"{name}: rejected_memory — uses raw uint32 fallback"
            )
        elif not has_real_gen_tp:
            classifications[name] = "experimental_only"
            notes.append(
                f"{name}: experimental_only — awaiting real generation "
                "data"
            )
        elif has_real_gen_tp and not baseline_valid:
            classifications[name] = "needs_valid_teacher_forced_baseline"
            notes.append(
                f"{name}: needs_valid_teacher_forced_baseline — "
                "baseline_fp16 identity failed, cannot trust comparisons"
            )
        elif tf_has and not tf_pass:
            # Teacher-forced failure overrides throughput verdict
            if fr_has and fr_pass and not has_fr_divergence:
                classifications[name] = (
                    "free_running_match_but_teacher_forced_failed"
                )
                notes.append(
                    f"{name}: "
                    "free_running_match_but_teacher_forced_failed — "
                    "free-running exact match but teacher-forced logits "
                    "fail"
                )
            elif has_fr_divergence:
                classifications[name] = "rejected_generation_divergence"
                notes.append(
                    f"{name}: rejected_generation_divergence — "
                    "free-running divergence observed at short/medium "
                    "prompts"
                )
            else:
                classifications[name] = "rejected_generation_quality"
                notes.append(
                    f"{name}: rejected_generation_quality — "
                    "teacher-forced real-generation failed"
                )
        elif verdict == "rejected_speed":
            classifications[name] = "rejected_speed"
            notes.append(
                f"{name}: rejected_speed — throughput regression"
            )
        elif not qwen_real:
            classifications[name] = "experimental_pending_1_5b"
            notes.append(
                f"{name}: experimental_pending_1_5b — awaiting 1.5B "
                "validation"
            )
        elif not qwen_pass:
            classifications[name] = "rejected_1_5b"
            notes.append(
                f"{name}: rejected_1_5b — fails 1.5B validation"
            )
        elif fr_has and not fr_pass:
            classifications[name] = "generation_divergence_observed"
            notes.append(
                f"{name}: generation_divergence_observed — "
                "teacher-forced passes but free-running diverges"
            )
        else:
            # Determine candidate type for configs that passed all gates
            baseline = next(
                (r for r in rows
                 if r.get("config") == "stable_k8_v5_gs64"),
                None,
            )
            if baseline:
                cm = row.get("cosine_min") or 0
                bcm = baseline.get("cosine_min") or 0
                to = row.get("top5_overlap") or 0
                bto = baseline.get("top5_overlap") or 0
                better_quality = cm >= bcm and to >= bto

                cr = row.get("compression_ratio") or 1.0
                bcr = baseline.get("compression_ratio") or 1.0
                better_memory = cr >= bcr

                baseline_tp = tp_configs.get("stable_k8_v5_gs64", {})
                baseline_tps = baseline_tp.get("tokens_per_second", 0)
                this_tps = tp.get("tokens_per_second", 0)

                if name == "turbo_polar":
                    if this_tps > baseline_tps:
                        classifications[name] = "speed_candidate"
                        notes.append(
                            f"{name}: speed_candidate — fastest "
                            "compressed path"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — not faster "
                            "than baseline"
                        )
                elif name == "adaptive":
                    if better_quality:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — strong quality"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear "
                            "quality win"
                        )
                elif name == "experimental_hybrid":
                    if better_memory and not better_quality:
                        classifications[name] = "memory_candidate"
                        notes.append(
                            f"{name}: memory_candidate — better memory"
                        )
                    elif better_quality and better_memory:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — beats baseline"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear win"
                        )
                else:
                    if better_quality and better_memory:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — beats baseline"
                        )
                    elif better_memory:
                        classifications[name] = "memory_candidate"
                        notes.append(
                            f"{name}: memory_candidate — better memory"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear win"
                        )
            else:
                classifications[name] = "experimental_only"
                notes.append(
                    f"{name}: experimental_only — baseline missing"
                )

    # Build promotion_blocked_by list reflecting current evidence
    promotion_blocked_by: list[str] = ["teacher_forced_logit_drift", "qjl_failed"]
    if has_decode_diagnostics:
        promotion_blocked_by.append("decode_update_failure")
    else:
        promotion_blocked_by.append("decode_update_diagnostics_pending")

    return {
        "release": "experimental",
        "stable_default": "k8_v5_gs64",
        "promoted_to_default": False,
        "teacher_forced_baseline_valid": baseline_valid,
        "promotion_blocked_by": promotion_blocked_by,
        "experimental_status": "research_only",
        "classifications": classifications,
        "notes": notes,
        "evidence": {
            "has_0_5b_validation": bool(comp),
            "has_throughput": bool(throughput),
            "has_real_generation_throughput": has_real_gen_tp,
            "has_decode_diagnostics": has_decode_diagnostics,
            "has_1_5b_validation": bool(qwen_real),
            "has_layer_policy": (exp_dir / "layer_policy.json").exists(),
            "has_sensitivity": (
                exp_dir / "per_layer_sensitivity.json"
            ).exists(),
        },
        "promotion_rules": [
            "stable_default: locked default runtime",
            (
                "stable_default_under_teacher_forced_drift: "
                "locked default runtime, teacher-forced drift unresolved"
            ),
            (
                "stable_default_decode_update_issue: "
                "locked default runtime, decode-update degraded"
            ),
            "quality_candidate: better quality, acceptable throughput",
            (
                "quality_candidate_under_teacher_forced_drift: "
                "proven at 0.5B, teacher-forced drift unresolved"
            ),
            (
                "quality_candidate_decode_update_issue: "
                "proven at 0.5B, decode-update degraded"
            ),
            "stable_baseline: proven stable config",
            (
                "stable_baseline_under_teacher_forced_drift: "
                "proven at 0.5B, teacher-forced drift unresolved"
            ),
            (
                "stable_baseline_decode_update_issue: "
                "proven at 0.5B, decode-update degraded"
            ),
            "stable_needs_data: missing context validation",
            "rejected_quality: fails 512/1024/2048",
            (
                "rejected_memory: missing compressed bytes or "
                "raw uint32 fallback"
            ),
            "rejected_speed: missing throughput or regression",
            "needs_throughput_data: missing throughput benchmark",
            (
                "needs_valid_teacher_forced_baseline: "
                "baseline_fp16 identity failed"
            ),
            (
                "rejected_generation_quality: "
                "teacher-forced real-generation failed"
            ),
            (
                "rejected_generation_divergence: "
                "free-running divergence observed at short/medium prompts"
            ),
            (
                "free_running_match_but_teacher_forced_failed: "
                "free-running exact match but teacher-forced logits fail"
            ),
            (
                "generation_divergence_observed: "
                "teacher-forced passes but free-running diverges"
            ),
            "experimental_pending_1_5b: awaiting 1.5B validation",
            "rejected_1_5b: fails 1.5B validation",
            "experimental_only: passes all gates but no clear win",
            "speed_candidate: fastest compressed path",
            "memory_candidate: better memory, comparable quality",
            "disabled: QJL or hard-blocked config",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify experimental configs"
    )
    parser.add_argument(
        "--experimental-dir", default="artifacts/proof/experimental"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/config_classification.json",
    )
    args = parser.parse_args()

    result = classify_configs(Path(args.experimental_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote classification to {out_path}")
    for note in result["notes"]:
        print(f"  {note}")
    print(f"\nEvidence status: {result['evidence']}")


if __name__ == "__main__":
    main()
