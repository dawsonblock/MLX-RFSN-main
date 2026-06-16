"""Release level definitions and gating for RFSN candidates.

Phase 10: Three release levels with strict promotion criteria.

Level 1 — alpha
  * Exactly one active candidate: K8/V8 GS64
  * No placeholder artifacts
  * Native gate passes on Apple Silicon
  * Token match to dense baseline
  * Zero fallback in strict mode

Level 2 — beta
  * K8/V8 proven on at least 3 context lengths (128, 512, 2048)
  * K8/V6 explored (reference path OK, Metal kernel gated)
  * No vendored repo dependencies in default registry
  * All experimental artifacts replaced with native gate artifacts

Level 3 — stable
  * All bit-widths (K8/V8, K8/V6, K8/V5) validated
  * Metal kernel supports mixed bit-widths
  * Performance within 2x of dense baseline
  * Memory reduction > 25% measured
  * No full-history materialization in any path
"""
from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from typing import Any


class ReleaseLevel(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    STABLE = "stable"


@dataclass
class ReleaseCriteria:
    """Criteria for a given release level."""

    level: ReleaseLevel
    allowed_candidates: list[str]
    require_strict_gate: bool
    require_token_match: bool
    require_zero_fallback: bool
    require_native_artifacts: bool
    allow_vendored_repos: bool
    allow_experimental_registry: bool
    min_context_lengths: int
    max_bit_width: int
    # Quality thresholds (None = no enforcement)
    max_kl_divergence: float | None = None
    min_top1_match: float | None = None
    min_logit_cosine: float | None = None
    max_speed_ratio: float | None = None  # packed_time / dense_time
    min_memory_reduction_ratio: float | None = None  # dense_mb / packed_mb

    def check(self, manifest: dict[str, Any]) -> list[str]:
        """Return list of violations (empty if all pass)."""
        violations = []

        candidate = manifest.get("candidate", "")
        if candidate not in self.allowed_candidates:
            violations.append(
                f"candidate {candidate!r} not in {self.allowed_candidates}"
            )

        runs = manifest.get("runs", [])
        if len(runs) < self.min_context_lengths:
            violations.append(
                f"only {len(runs)} context lengths, need {self.min_context_lengths}"
            )

        for run in runs:
            packed = run.get("packed", {})
            counters = packed.get("counters", {})

            if self.require_token_match:
                if not run.get("token_match"):
                    violations.append(
                        f"context {run['context_length']}: token mismatch"
                    )

            if self.require_strict_gate:
                if not counters.get("requested_strict_mode"):
                    violations.append(
                        f"context {run['context_length']}: requested_strict_mode is false"
                    )
                if not counters.get("effective_strict_mode"):
                    violations.append(
                        f"context {run['context_length']}: effective_strict_mode is false"
                    )

            if self.require_zero_fallback:
                if counters.get("dense_fallback_calls", 0) > 0:
                    violations.append(
                        f"context {run['context_length']}: dense_fallback > 0"
                    )
                if counters.get("full_history_materialization_calls", 0) > 0:
                    violations.append(
                        f"context {run['context_length']}: materialization > 0"
                    )
                if counters.get("packed_attention_calls", 0) == 0:
                    violations.append(
                        f"context {run['context_length']}: packed_attention_calls == 0"
                    )

            # Quality thresholds
            quality = run.get("quality")
            if quality and isinstance(quality, dict):
                if self.max_kl_divergence is not None:
                    kl = quality.get("kl_divergence")
                    if kl is not None and kl > self.max_kl_divergence:
                        violations.append(
                            f"context {run['context_length']}: KL {kl} > {self.max_kl_divergence}"
                        )
                if self.min_top1_match is not None:
                    tm = quality.get("top1_match")
                    if tm is not None and tm < self.min_top1_match:
                        violations.append(
                            f"context {run['context_length']}: top1_match {tm} < {self.min_top1_match}"
                        )
                if self.min_logit_cosine is not None:
                    cos = quality.get("logit_cosine")
                    if cos is not None and cos < self.min_logit_cosine:
                        violations.append(
                            f"context {run['context_length']}: cosine {cos} < {self.min_logit_cosine}"
                        )

            # Speed threshold — compare free-running times (comparable across candidates)
            if self.max_speed_ratio is not None:
                dense_ms = run.get("dense", {}).get("free_running_elapsed_ms")
                packed_ms = run.get("packed", {}).get("free_running_elapsed_ms")
                if dense_ms and packed_ms and dense_ms > 0:
                    ratio = packed_ms / dense_ms
                    if ratio > self.max_speed_ratio:
                        violations.append(
                            f"context {run['context_length']}: speed ratio {ratio:.1f}x > {self.max_speed_ratio}x"
                        )

            # Memory threshold
            if self.min_memory_reduction_ratio is not None:
                dense_mb = run.get("dense", {}).get("memory", {}).get("total_accounted_mb")
                packed_mb = run.get("packed", {}).get("memory", {}).get("total_accounted_mb")
                if dense_mb and packed_mb and packed_mb > 0:
                    ratio = dense_mb / packed_mb
                    if ratio < self.min_memory_reduction_ratio:
                        violations.append(
                            f"context {run['context_length']}: memory ratio {ratio:.2f}x < {self.min_memory_reduction_ratio}x"
                        )

        return violations


# ---------------------------------------------------------------------------
# Canonical criteria per level
# ---------------------------------------------------------------------------

ALPHA_CRITERIA = ReleaseCriteria(
    level=ReleaseLevel.ALPHA,
    allowed_candidates=[
        "rfsn_direct_packed_k8v8_gs64",
        "dense_mlx_baseline",
    ],
    require_strict_gate=True,
    require_token_match=True,
    require_zero_fallback=True,
    require_native_artifacts=True,
    allow_vendored_repos=True,
    allow_experimental_registry=False,
    min_context_lengths=1,
    max_bit_width=8,
)

BETA_CRITERIA = ReleaseCriteria(
    level=ReleaseLevel.BETA,
    allowed_candidates=[
        "rfsn_direct_packed_k8v8_gs64",
        "rfsn_direct_packed_k8v6_gs64",
        "dense_mlx_baseline",
    ],
    require_strict_gate=True,
    require_token_match=True,
    require_zero_fallback=True,
    require_native_artifacts=True,
    allow_vendored_repos=False,
    allow_experimental_registry=True,
    min_context_lengths=3,
    max_bit_width=8,
    max_kl_divergence=0.1,
    min_top1_match=0.95,
    min_logit_cosine=0.99,
)

STABLE_CRITERIA = ReleaseCriteria(
    level=ReleaseLevel.STABLE,
    allowed_candidates=[
        "rfsn_direct_packed_k8v8_gs64",
        "rfsn_direct_packed_k8v6_gs64",
        "rfsn_direct_packed_k8v5_gs64",
        "dense_mlx_baseline",
    ],
    require_strict_gate=True,
    require_token_match=True,
    require_zero_fallback=True,
    require_native_artifacts=True,
    allow_vendored_repos=False,
    allow_experimental_registry=True,
    min_context_lengths=3,
    max_bit_width=8,
    max_kl_divergence=0.05,
    min_top1_match=0.99,
    min_logit_cosine=0.995,
    max_speed_ratio=2.0,
    min_memory_reduction_ratio=1.25,
)


def get_criteria(level: ReleaseLevel) -> ReleaseCriteria:
    """Return criteria for the given release level."""
    return {
        ReleaseLevel.ALPHA: ALPHA_CRITERIA,
        ReleaseLevel.BETA: BETA_CRITERIA,
        ReleaseLevel.STABLE: STABLE_CRITERIA,
    }[level]
