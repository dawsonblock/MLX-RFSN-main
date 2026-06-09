"""RFSN v10 — Runtime audit mode.

Audit every N decode steps by running both the compressed path and a
reference FP16 (or otherwise stable) path on the same state, comparing
logits, logging drift, and recommending fallbacks when quality degrades.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    from rfsn_v10.compat import mx

DEFAULT_AUDIT_INTERVAL: int = 32
DEFAULT_AUDIT_LOG_PATH: Path = Path("artifacts/runtime_logs/audit.jsonl")

# Hardened thresholds per Phase 8 — experimental modes must not
# drift silently
DEFAULT_MIN_LOGIT_COSINE: float = 0.999
DEFAULT_MAX_KL: float = 0.001
DEFAULT_MIN_TOP5_OVERLAP: float = 0.95
DEFAULT_FALLBACK_MODE: str = "k8_v5_gs64"


@dataclass
class AuditMetrics:
    """Drift metrics computed between compressed and reference logits."""

    logit_cosine: float = 1.0
    top5_overlap: float = 1.0
    kl_divergence: float = 0.0
    nll_delta: float | None = None
    has_nan_inf: bool = False
    step_num: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuditEvent:
    """Structured audit log entry."""

    event_type: str = "audit_decode_step"
    step_num: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    fallback_recommendation: str | None = None
    timestamp: float = field(default_factory=time.time)


def _softmax_stable(x: mx.array) -> mx.array:
    """Numerically stable softmax over the last axis."""
    x_max = mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x - x_max)
    return e / mx.sum(e, axis=-1, keepdims=True)


def _kl_div(p: mx.array, q: mx.array) -> float:
    """KL(p || q) averaged over the batch / sequence dimensions."""
    # Add epsilon for stability
    eps = 1e-10
    p_safe = mx.maximum(p, eps)
    q_safe = mx.maximum(q, eps)
    kl = mx.sum(p_safe * (mx.log(p_safe) - mx.log(q_safe)), axis=-1)
    return mx.mean(kl).item()


def _compute_top5_overlap(
    compressed_logits: mx.array,
    reference_logits: mx.array,
) -> float:
    """Compute the mean Jaccard overlap of the top-5 tokens per position."""
    # compressed_logits / reference_logits: [B, T, V] or [B, V]
    # For simplicity we assume the last axis is vocab.
    c_top5 = mx.argsort(-compressed_logits, axis=-1)[..., :5]
    r_top5 = mx.argsort(-reference_logits, axis=-1)[..., :5]

    # For each position, count how many of c_top5 appear in r_top5.
    c_top5_exp = mx.expand_dims(c_top5, axis=-1)  # [..., K, 1]
    r_top5_exp = mx.expand_dims(r_top5, axis=-2)  # [..., 1, R]
    in_ref = mx.any(c_top5_exp == r_top5_exp, axis=-1)  # [..., K]
    intersection = mx.sum(in_ref, axis=-1)  # [...]
    # Jaccard = intersection / union ; union = |A| + |B| - intersection
    k_len = c_top5.shape[-1]
    r_len = r_top5.shape[-1]
    union = k_len + r_len - intersection
    jaccard = intersection / mx.maximum(union, 1)
    return mx.mean(jaccard).item()


def _compute_logit_cosine(
    compressed_logits: mx.array,
    reference_logits: mx.array,
) -> float:
    """Cosine similarity between flattened logit tensors."""
    a = compressed_logits.flatten().astype(mx.float32)
    b = reference_logits.flatten().astype(mx.float32)
    dot = mx.sum(a * b)
    norm = mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def _check_nan_inf(x: mx.array) -> bool:
    """Return True if any NaN or Inf values exist in *x*."""
    return bool(
        mx.any(mx.isnan(x)).item() or mx.any(mx.isinf(x)).item()
    )


def audit_decode_step(
    compressed_logits: mx.array,
    reference_logits: mx.array,
    step_num: int,
    audit_interval: int = DEFAULT_AUDIT_INTERVAL,
    labels: mx.array | None = None,
    log_path: Path | None = None,
) -> AuditMetrics | None:
    """Compare compressed-path logits against a reference and log drift.

    Args:
        compressed_logits:  Logits from the compressed / quantized path.
        reference_logits:   Logits from the FP16 or stable reference path.
        step_num:           Current decode step index.
        audit_interval:     Only compute metrics when
                            ``step_num % audit_interval == 0``.
        labels:             Optional ground-truth token ids for NLL delta.
        log_path:           Optional destination for the audit JSONL event.
                            Defaults to ``DEFAULT_AUDIT_LOG_PATH``.

    Returns:
        :class:`AuditMetrics` when an audit is performed, otherwise ``None``.
    """
    if step_num % audit_interval != 0:
        return None

    has_nan_inf = _check_nan_inf(compressed_logits) or _check_nan_inf(
        reference_logits
    )

    logit_cosine = _compute_logit_cosine(compressed_logits, reference_logits)
    top5_overlap = _compute_top5_overlap(compressed_logits, reference_logits)

    # KL divergence: KL(reference || compressed)
    p_ref = _softmax_stable(reference_logits)
    p_comp = _softmax_stable(compressed_logits)
    kl = _kl_div(p_ref, p_comp)

    nll_delta: float | None = None
    if labels is not None:
        # NLL for a single next-token prediction
        # labels shape assumed to be [B] or [B, 1] with token ids
        lbl = labels.flatten().astype(mx.int32)
        # Gather log-probs from reference
        log_probs_ref = mx.log(mx.maximum(p_ref, 1e-10))
        # If shapes differ, try to squeeze / align
        if log_probs_ref.ndim == 3 and lbl.ndim == 1:
            # [B, 1, V] case -> squeeze time dim
            log_probs_ref = log_probs_ref[:, 0, :]
            log_probs_comp = mx.log(mx.maximum(p_comp[:, 0, :], 1e-10))
        else:
            log_probs_comp = mx.log(mx.maximum(p_comp, 1e-10))
        nll_ref = -mx.mean(log_probs_ref[mx.arange(lbl.shape[0]), lbl]).item()
        nll_comp = -mx.mean(
            log_probs_comp[mx.arange(lbl.shape[0]), lbl]
        ).item()
        nll_delta = nll_comp - nll_ref

    metrics = AuditMetrics(
        logit_cosine=logit_cosine,
        top5_overlap=top5_overlap,
        kl_divergence=kl,
        nll_delta=nll_delta,
        has_nan_inf=has_nan_inf,
        step_num=step_num,
    )

    # Determine fallback recommendation immediately
    fallback = check_drift(metrics)

    event = AuditEvent(
        event_type="audit_decode_step",
        step_num=step_num,
        metrics={
            "logit_cosine": metrics.logit_cosine,
            "top5_overlap": metrics.top5_overlap,
            "kl_divergence": metrics.kl_divergence,
            "nll_delta": metrics.nll_delta,
            "has_nan_inf": metrics.has_nan_inf,
        },
        fallback_recommendation=fallback,
    )
    log_audit_event(event, log_path=log_path)
    return metrics


def check_drift(
    metrics: AuditMetrics,
    min_logit_cosine: float = DEFAULT_MIN_LOGIT_COSINE,
    max_kl: float = DEFAULT_MAX_KL,
    min_top5_overlap: float = DEFAULT_MIN_TOP5_OVERLAP,
    fallback_mode: str = DEFAULT_FALLBACK_MODE,
) -> str | None:
    """Return a fallback recommendation based on drift thresholds.

    Rules (in priority order):

    1. **NaN / Inf** → switch to ``FP16``
    2. **logit_cosine < min_logit_cosine** → switch to *fallback_mode*
    3. **top5_overlap < min_top5_overlap** → switch to *fallback_mode*
    4. **KL > max_kl** → switch to *fallback_mode*

    Args:
        metrics: Computed drift metrics.
        min_logit_cosine: Minimum cosine similarity (default 0.999).
        max_kl: Maximum KL divergence (default 0.001).
        min_top5_overlap: Minimum top-5 overlap (default 0.95).
        fallback_mode: Mode to fall back to (default ``k8_v5_gs64``).

    Returns:
        Fallback mode string or ``None``
        when no action is required.
    """
    if metrics.has_nan_inf:
        return "FP16"
    if metrics.logit_cosine < min_logit_cosine:
        return fallback_mode
    if metrics.top5_overlap < min_top5_overlap:
        return fallback_mode
    if metrics.kl_divergence > max_kl:
        return fallback_mode
    return None


def log_audit_event(
    event: AuditEvent, log_path: Path | None = None
) -> None:
    """Append an audit event to the JSONL audit log.

    Args:
        event:    The :class:`AuditEvent` to persist.
        log_path: Destination file path. Defaults to
                  ``artifacts/runtime_logs/audit.jsonl``.
    """
    path = log_path or DEFAULT_AUDIT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        record = {
            "event_type": event.event_type,
            "step_num": event.step_num,
            "metrics": event.metrics,
            "fallback_recommendation": event.fallback_recommendation,
            "timestamp": event.timestamp,
        }
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
