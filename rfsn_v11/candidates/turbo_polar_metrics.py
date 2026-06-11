"""TurboPolar-specific offline and online metrics.

These supplement the generic CandidateResult with PolarQuant/QJL-specific
quality measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolarOfflineMetrics:
    """Metrics from offline PolarQuant encoder/decoder validation."""

    reconstruction_cosine: float | None = None
    reconstruction_mse: float | None = None
    attention_score_cosine: float | None = None
    attention_top5_overlap: float | None = None
    attention_top10_overlap: float | None = None
    size_ratio: float | None = None
    compression_factor: float | None = None
    nan_inf_detected: bool = False

    def pass_gate(self) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if self.attention_score_cosine is None or self.attention_score_cosine < 0.999:
            failures.append(
                f"attention_score_cosine {self.attention_score_cosine} < 0.999"
            )
        if self.attention_top10_overlap is None or self.attention_top10_overlap < 0.98:
            failures.append(
                f"attention_top10_overlap {self.attention_top10_overlap} < 0.98"
            )
        if self.nan_inf_detected:
            failures.append("NaN or Inf detected")
        return len(failures) == 0, failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "reconstruction_cosine": self.reconstruction_cosine,
            "reconstruction_mse": self.reconstruction_mse,
            "attention_score_cosine": self.attention_score_cosine,
            "attention_top5_overlap": self.attention_top5_overlap,
            "attention_top10_overlap": self.attention_top10_overlap,
            "size_ratio": self.size_ratio,
            "compression_factor": self.compression_factor,
            "nan_inf_detected": self.nan_inf_detected,
        }


@dataclass
class QJLOfflineMetrics:
    """Metrics from offline QJL residual score correction validation."""

    without_qjl_score_error: float | None = None
    with_qjl_score_error: float | None = None
    without_qjl_top10_overlap: float | None = None
    with_qjl_top10_overlap: float | None = None
    qjl_kept: bool = False
    disable_reason: str = ""

    def evaluate(self) -> tuple[bool, str]:
        """Return (kept, reason)."""
        if self.with_qjl_score_error is None or self.without_qjl_score_error is None:
            return False, "missing score_error values"
        if self.with_qjl_score_error >= self.without_qjl_score_error:
            return False, "QJL did not reduce score error"
        if self.with_qjl_top10_overlap is None or self.without_qjl_top10_overlap is None:
            return False, "missing top10 overlap values"
        if self.with_qjl_top10_overlap <= self.without_qjl_top10_overlap:
            return False, "QJL did not improve top10 overlap"
        return True, "QJL improved both score error and top10 overlap"

    def as_dict(self) -> dict[str, Any]:
        return {
            "without_qjl_score_error": self.without_qjl_score_error,
            "with_qjl_score_error": self.with_qjl_score_error,
            "without_qjl_top10_overlap": self.without_qjl_top10_overlap,
            "with_qjl_top10_overlap": self.with_qjl_top10_overlap,
            "qjl_kept": self.qjl_kept,
            "disable_reason": self.disable_reason,
        }


@dataclass
class KernelValidationMetrics:
    """Metrics from fused Metal kernel validation against Python reference."""

    kernel_name: str = ""
    score_cosine: float | None = None
    top10_overlap: float | None = None
    max_abs_error: float | None = None
    nan_inf_detected: bool = False
    fallback_used: bool = False

    def pass_gate(self, tolerance: float = 1e-3) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if self.score_cosine is None or self.score_cosine < 0.9999:
            failures.append(f"score_cosine {self.score_cosine} < 0.9999")
        if self.top10_overlap is None or self.top10_overlap < 0.99:
            failures.append(f"top10_overlap {self.top10_overlap} < 0.99")
        if self.max_abs_error is None or self.max_abs_error > tolerance:
            failures.append(f"max_abs_error {self.max_abs_error} > {tolerance}")
        if self.nan_inf_detected:
            failures.append("NaN or Inf detected")
        if self.fallback_used:
            failures.append("fallback was used")
        return len(failures) == 0, failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "score_cosine": self.score_cosine,
            "top10_overlap": self.top10_overlap,
            "max_abs_error": self.max_abs_error,
            "nan_inf_detected": self.nan_inf_detected,
            "fallback_used": self.fallback_used,
        }
