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
)


def get_criteria(level: ReleaseLevel) -> ReleaseCriteria:
    """Return criteria for the given release level."""
    return {
        ReleaseLevel.ALPHA: ALPHA_CRITERIA,
        ReleaseLevel.BETA: BETA_CRITERIA,
        ReleaseLevel.STABLE: STABLE_CRITERIA,
    }[level]
