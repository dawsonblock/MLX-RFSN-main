"""Quality gate evaluation for shootout candidates.

Gates (Alpha 8.3 teacher-forced calibration):
    logit_cosine  >= 0.999
    KL divergence <= 0.1
    top5_overlap  >= 0.85
    top10_overlap >= 0.90
    max_logit_delta <= 10.0

Calibration data (mlx_lm_quantized_kv_b8, 0.5B, teacher-forced):
    cosine=0.99983  KL=0.055  top5=0.889  top10=0.904  delta=7.46  top1=1.0

These thresholds pass the upstream-maintained 8-bit KV baseline while
still rejecting genuinely degraded candidates (TurboQuant V2 b4 on
0.5B: cosine=0.995  top5=0.40  top10=0.40).

Failures are never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds - Single Source of Truth (P2 Unification)
# ---------------------------------------------------------------------------
# P2 Fix: All thresholds are defined in ONE place - LogitGateThresholds dataclass.
# Module-level constants reference the dataclass to ensure consistency.

@dataclass(frozen=True)
class LogitGateThresholds:
    """Single source of truth for full-logit quality gate thresholds.

    P2 Unification: This dataclass is the authoritative definition of all
    quality gate thresholds. All other code must reference these values
    rather than defining their own.

    Calibration data (Alpha 8.3, mlx_lm_quantized_kv_b8, 0.5B, teacher-forced):
        cosine=0.99983  KL=0.055  top5=0.889  top10=0.904  delta=7.46  top1=1.0

    These thresholds pass the upstream-maintained 8-bit KV baseline while
    still rejecting genuinely degraded candidates.
    """

    logit_cosine_min: float = 0.999
    kl_divergence_max: float = 0.1
    top5_overlap_min: float = 0.85
    top10_overlap_min: float = 0.90
    max_logit_delta_max: float = 10.0

    def to_dict(self) -> dict[str, float]:
        return {
            "logit_cosine_min": self.logit_cosine_min,
            "kl_divergence_max": self.kl_divergence_max,
            "top5_overlap_min": self.top5_overlap_min,
            "top10_overlap_min": self.top10_overlap_min,
            "max_logit_delta_max": self.max_logit_delta_max,
        }


# P2: Module-level constants reference the single source of truth
_DEFAULT_THRESHOLDS = LogitGateThresholds()
LOGIT_COSINE_MIN: float = _DEFAULT_THRESHOLDS.logit_cosine_min
KL_DIVERGENCE_MAX: float = _DEFAULT_THRESHOLDS.kl_divergence_max
TOP5_OVERLAP_MIN: float = _DEFAULT_THRESHOLDS.top5_overlap_min
TOP10_OVERLAP_MIN: float = _DEFAULT_THRESHOLDS.top10_overlap_min
MAX_LOGIT_DELTA_MAX: float = _DEFAULT_THRESHOLDS.max_logit_delta_max


# ---------------------------------------------------------------------------
# Allowed gate_status values
# ---------------------------------------------------------------------------

GATE_STATUS_PASS = "PASS"
GATE_STATUS_FAIL = "FAIL"
GATE_STATUS_PENDING_LOGIT_GATE = "PENDING_LOGIT_GATE"
GATE_STATUS_PENDING_MEMORY_METRICS = "PENDING_MEMORY_METRICS"
GATE_STATUS_PENDING_REAL_CACHE_INJECTION = "PENDING_REAL_CACHE_INJECTION"
GATE_STATUS_ERROR = "ERROR"


@dataclass
class QualityGateResult:
    passed: bool
    logit_cosine: float | None
    kl_divergence: float | None
    top1_match: float | None
    top5_overlap: float | None
    top10_overlap: float | None
    max_logit_delta: float | None
    first_divergent_token: int | None
    failure_reasons: list[str]


def logit_quality_metrics(
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> dict[str, float | None]:
    """Compute quality metrics between baseline and candidate logit arrays.

    Parameters
    ----------
    baseline_logits, candidate_logits
        Shape (T, vocab) float32 arrays of log-probabilities or raw logits.
    """
    if baseline_logits.shape != candidate_logits.shape:
        return {
            "logit_cosine": None,
            "kl_divergence": None,
            "top1_match": None,
            "top5_overlap": None,
            "top10_overlap": None,
            "max_logit_delta": None,
            "first_divergent_token": None,
        }

    T = baseline_logits.shape[0]

    # Cosine similarity per token, averaged
    dots = np.sum(baseline_logits * candidate_logits, axis=-1)
    norms_b = np.linalg.norm(baseline_logits, axis=-1) + 1e-12
    norms_c = np.linalg.norm(candidate_logits, axis=-1) + 1e-12
    cosines = dots / (norms_b * norms_c)
    logit_cosine = float(np.mean(cosines))

    # KL divergence: softmax baseline as reference distribution
    b_sm = _safe_softmax(baseline_logits)
    c_sm = _safe_softmax(candidate_logits)
    kl_per_token = np.sum(b_sm * (np.log(b_sm + 1e-12) - np.log(c_sm + 1e-12)), axis=-1)
    kl_divergence = float(np.mean(kl_per_token))

    # Top-k overlaps
    b_top1 = np.argmax(baseline_logits, axis=-1)
    c_top1 = np.argmax(candidate_logits, axis=-1)
    top1_match = float(np.mean(b_top1 == c_top1))

    b_top5 = np.argsort(baseline_logits, axis=-1)[:, -5:]
    c_top5 = np.argsort(candidate_logits, axis=-1)[:, -5:]
    top5_overlap = float(np.mean([
        len(set(b_top5[t]) & set(c_top5[t])) / 5.0 for t in range(T)
    ]))

    b_top10 = np.argsort(baseline_logits, axis=-1)[:, -10:]
    c_top10 = np.argsort(candidate_logits, axis=-1)[:, -10:]
    top10_overlap = float(np.mean([
        len(set(b_top10[t]) & set(c_top10[t])) / 10.0 for t in range(T)
    ]))

    # Max logit delta
    max_logit_delta = float(np.max(np.abs(baseline_logits - candidate_logits)))

    # First divergent token (top-1 differs)
    divergent = np.where(b_top1 != c_top1)[0]
    first_divergent_token = int(divergent[0]) if len(divergent) > 0 else None

    return {
        "logit_cosine": logit_cosine,
        "kl_divergence": kl_divergence,
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "max_logit_delta": max_logit_delta,
        "first_divergent_token": first_divergent_token,
    }


def evaluate_quality_gate(metrics: dict[str, float | None]) -> QualityGateResult:
    """Apply quality gate thresholds and return a structured result."""
    failures: list[str] = []

    cosine = metrics.get("logit_cosine")
    kl = metrics.get("kl_divergence")
    top5 = metrics.get("top5_overlap")
    top10 = metrics.get("top10_overlap")
    max_delta = metrics.get("max_logit_delta")

    if cosine is None or cosine < LOGIT_COSINE_MIN:
        failures.append(
            f"logit_cosine {cosine} < {LOGIT_COSINE_MIN}"
        )
    if kl is None or kl > KL_DIVERGENCE_MAX:
        failures.append(f"KL {kl} > {KL_DIVERGENCE_MAX}")
    if top5 is None or top5 < TOP5_OVERLAP_MIN:
        failures.append(f"top5_overlap {top5} < {TOP5_OVERLAP_MIN}")
    if top10 is None or top10 < TOP10_OVERLAP_MIN:
        failures.append(f"top10_overlap {top10} < {TOP10_OVERLAP_MIN}")
    if max_delta is None or max_delta > MAX_LOGIT_DELTA_MAX:
        failures.append(f"max_logit_delta {max_delta} > {MAX_LOGIT_DELTA_MAX}")

    return QualityGateResult(
        passed=len(failures) == 0,
        logit_cosine=cosine,
        kl_divergence=kl,
        top1_match=metrics.get("top1_match"),
        top5_overlap=top5,
        top10_overlap=top10,
        max_logit_delta=max_delta,
        first_divergent_token=metrics.get("first_divergent_token"),
        failure_reasons=failures,
    )


def compute_promotion_eligibility(
    logit_gate_passed: bool | None,
    memory_gate_passed: bool | None,
    actual_kv_memory_mb: float | None,
    working_set_memory_mb: float | None,
    size_ratio: float | None,
    compression_factor: float | None,
) -> tuple[bool, str]:
    """Return (promotion_eligible, gate_status)."""
    if logit_gate_passed is not True:
        return False, GATE_STATUS_PENDING_LOGIT_GATE
    if memory_gate_passed is not True:
        return False, GATE_STATUS_PENDING_MEMORY_METRICS
    if actual_kv_memory_mb is None:
        return False, GATE_STATUS_PENDING_MEMORY_METRICS
    if working_set_memory_mb is None:
        return False, GATE_STATUS_PENDING_MEMORY_METRICS
    if size_ratio is None:
        return False, GATE_STATUS_PENDING_MEMORY_METRICS
    if compression_factor is None:
        return False, GATE_STATUS_PENDING_MEMORY_METRICS
    return True, GATE_STATUS_PASS


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)
