"""TurboPolar promotion gate evaluation.

Enforces the Phase 10 checklist:
  teacher_forced_logit_gate = PASS
  memory_gate = PASS
  attention_output_gate = PASS
  real_cache_used = true
  metal_kernel_used = true
  fallback_used = false
  token_sequence_hash non-empty
  runtime trace is instrumented, not estimated
  cache_bytes_written_actual > 0
  cache_bytes_read_actual > 0
  patch restored / no global contamination
  at least 2 model sizes tested

Until all are true:
  promotion_eligible = false
"""
from __future__ import annotations

from typing import Any

from .turbo_polar_trace import TurboPolarTrace
from .turbo_polar_metrics import PolarOfflineMetrics, QJLOfflineMetrics, KernelValidationMetrics


def evaluate_turbo_polar_promotion(
    logit_gate_passed: bool,
    memory_gate_passed: bool,
    attention_output_gate_passed: bool,
    trace: TurboPolarTrace,
    offline_metrics: PolarOfflineMetrics,
    qjl_metrics: QJLOfflineMetrics | None,
    kernel_metrics: KernelValidationMetrics,
    models_tested: list[str],
) -> tuple[bool, str, list[str]]:
    """Evaluate TurboPolar promotion eligibility.

    Returns:
        (eligible, gate_status, reasons)
    """
    reasons: list[str] = []

    if not logit_gate_passed:
        reasons.append("teacher_forced_logit_gate != PASS")
    if not memory_gate_passed:
        reasons.append("memory_gate != PASS")
    if not attention_output_gate_passed:
        reasons.append("attention_output_gate != PASS")

    trace_ok, trace_reason = trace.validate_for_promotion()
    if not trace_ok:
        reasons.append(f"trace validation failed: {trace_reason}")

    if not kernel_metrics.pass_gate()[0]:
        reasons.append("kernel validation gate failed")
    if kernel_metrics.fallback_used:
        reasons.append("kernel fallback was used")

    if not offline_metrics.pass_gate()[0]:
        reasons.append("offline PolarQuant gate failed")

    if qjl_metrics is not None and not qjl_metrics.qjl_kept:
        # QJL disabled is okay, but if it was tried and failed, note it
        pass  # QJL optional

    if len(models_tested) < 2:
        reasons.append(f"only {len(models_tested)} model(s) tested (need >= 2)")

    if reasons:
        return False, "FAIL_PROMOTION_GATE", reasons
    return True, "PASS", []


def create_promotion_artifact(
    eligible: bool,
    gate_status: str,
    reasons: list[str],
    trace: TurboPolarTrace,
    models_tested: list[str],
) -> dict[str, Any]:
    """Generate the promotion artifact dict for JSON serialization."""
    return {
        "candidate": "turbo_polar",
        "promotion_eligible": eligible,
        "gate_status": gate_status,
        "failed_gate_reasons": reasons,
        "trace": trace.as_dict(),
        "models_tested": models_tested,
        "methodology_status": "TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION",
        "benchmark_methodology": "teacher_forced_logit_v1",
        "promotion_allowed": False,  # Always false until explicitly approved
    }
