"""Promotion criteria and levels for polar_fused candidates.

A Polar candidate may be promoted only when all criteria are met:
1. All correctness tests pass
2. All quality gates pass
3. Actual cache compression measured
4. No host NumPy packing
5. No permanent FP16 duplicate
6. No global monkey-patching
7. Unsupported layers fall back correctly
8. Long-context retrieval intact
9. End-to-end decode not slower below threshold
10. Measurable speed or capacity advantage above threshold
11. Results include model, device, MLX version, context, config
12. Benchmark artifacts reproducible
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PromotionLevel:
    """Promotion levels for polar_fused candidates."""
    LAB = "LAB"              # Kernel-correct, not model-approved
    CANDIDATE = "CANDIDATE"  # Quality-approved on named models
    SUPPORTED = "SUPPORTED"  # Quality and reliability approved
    DEFAULT = "DEFAULT"      # Faster or more capable than RFSN stable


@dataclass
class PromotionStatus:
    """Current promotion status of a polar_fused candidate."""
    candidate_name: str
    level: str
    can_promote: bool
    missing_criteria: list[str] = field(default_factory=list)
    completed_criteria: list[str] = field(default_factory=list)
    model_results: dict[str, Any] = field(default_factory=dict)
    benchmark_summary: dict[str, Any] = field(default_factory=dict)


class PromotionEngine:
    """Evaluate whether a polar_fused candidate meets promotion criteria."""

    CRITERIA = [
        "correctness_tests_pass",
        "quality_gates_pass",
        "compression_measured",
        "no_numpy_packing",
        "no_fp16_duplicate",
        "no_global_monkeypatch",
        "fallback_works",
        "long_context_intact",
        "not_slower_below_threshold",
        "reproducible_benchmarks",
    ]

    def __init__(self) -> None:
        self._status: dict[str, PromotionStatus] = {}

    def evaluate(self, candidate_name: str, evidence: dict[str, Any]) -> PromotionStatus:
        """Evaluate candidate against all promotion criteria.

        Parameters
        ----------
        evidence
            Dict mapping criterion name to result (bool or dict).
        """
        completed = []
        missing = []

        for criterion in self.CRITERIA:
            result = evidence.get(criterion, False)
            if isinstance(result, dict):
                result = result.get("pass", False)
            if result:
                completed.append(criterion)
            else:
                missing.append(criterion)

        can_promote = len(missing) == 0

        # Determine level
        if can_promote:
            # Check if beats RFSN stable (must be explicitly True)
            beats_stable = evidence.get("faster_above_threshold", False)
            if isinstance(beats_stable, dict):
                beats_stable = beats_stable.get("pass", False)
            if beats_stable is True:
                level = PromotionLevel.DEFAULT
            else:
                level = PromotionLevel.SUPPORTED
        else:
            quality_ok = evidence.get("quality_gates_pass", False)
            if isinstance(quality_ok, dict):
                quality_ok = quality_ok.get("pass", False)
            if quality_ok:
                level = PromotionLevel.CANDIDATE
            else:
                level = PromotionLevel.LAB

        status = PromotionStatus(
            candidate_name=candidate_name,
            level=level,
            can_promote=can_promote,
            missing_criteria=missing,
            completed_criteria=completed,
            model_results=evidence.get("model_results", {}),
            benchmark_summary=evidence.get("benchmark_summary", {}),
        )
        self._status[candidate_name] = status
        return status

    def get_status(self, candidate_name: str) -> PromotionStatus | None:
        return self._status.get(candidate_name)

    def all_statuses(self) -> dict[str, PromotionStatus]:
        return dict(self._status)
