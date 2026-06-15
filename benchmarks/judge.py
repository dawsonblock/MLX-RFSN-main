"""RFSN benchmark judge.

Given a CandidateResult and a baseline CandidateResult, emits one of four verdicts:

    PROMOTE          — quality gates pass AND at least one improvement gate passes.
                       Safe to move up the promotion ladder.

    KEEP_EXPERIMENTAL — quality gates pass but no improvement gate passes, OR one
                       or more required metrics are None (unmeasured).

    REJECT           — any quality gate fails.  Method damages output quality.

    REGRESSION       — all quality gates pass and there is improvement, but the
                       candidate is worse than a previously stable method on a metric
                       where the stable method passed.

The judge never excuses quality damage because the method is "more advanced."

Promotion gates (all must pass for PROMOTE):
    logit_cosine              >= LOGIT_COSINE_MIN
    top5_overlap              >= TOP5_OVERLAP_MIN
    attention_score_cosine    >= ATTN_COSINE_MIN
    attention_top5_overlap    >= ATTN_TOP5_MIN
    perplexity_delta          <= PERPLEXITY_DELTA_MAX   (lower is better)
    visible_output_drift_score <= DRIFT_MAX

Improvement gates (at least one must pass for PROMOTE):
    KV memory reduced by >= KV_MEMORY_IMPROVEMENT_MIN (fraction, e.g. 0.30)
    peak memory reduced by >= PEAK_MEMORY_IMPROVEMENT_MIN
    decode TPS improved by >= DECODE_TPS_IMPROVEMENT_MIN
    enables longer context (snapkv_enabled=True and memory saved is positive)

Required fields for promotion (None → KEEP_EXPERIMENTAL):
    logit_cosine, top5_overlap, attention_score_cosine, attention_top5_overlap,
    perplexity_delta, visible_output_drift_score,
    peak_memory_mb, kv_cache_memory_mb, compressed_kv_memory_mb, decode_tps
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schemas import CandidateResult

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LOGIT_COSINE_MIN: float = 0.995
TOP5_OVERLAP_MIN: float = 0.95
ATTN_COSINE_MIN: float = 0.995
ATTN_TOP5_MIN: float = 0.95
PERPLEXITY_DELTA_MAX: float = 0.02          # candidate ppl - baseline ppl
DRIFT_MAX: float = 0.05                     # visible output drift score

KV_MEMORY_IMPROVEMENT_MIN: float = 0.30    # KV memory reduced by >= 30 %
PEAK_MEMORY_IMPROVEMENT_MIN: float = 0.20  # peak memory reduced by >= 20 %
DECODE_TPS_IMPROVEMENT_MIN: float = 0.10   # decode TPS improved by >= 10 %

# Required quality metrics whose absence blocks promotion
_REQUIRED_QUALITY_FIELDS = (
    "logit_cosine",
    "top5_overlap",
    "attention_score_cosine",
    "attention_top5_overlap",
    "perplexity_delta",
    "visible_output_drift_score",
)

# Required memory/speed fields whose absence blocks promotion
_REQUIRED_PERF_FIELDS = (
    "peak_memory_mb",
    "kv_cache_memory_mb",
    "compressed_kv_memory_mb",
    "decode_tps",
)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class VerdictLabel(str, Enum):
    PROMOTE = "PROMOTE"
    KEEP_EXPERIMENTAL = "KEEP_EXPERIMENTAL"
    REJECT = "REJECT"
    REGRESSION = "REGRESSION"
    SMOKE_PASS = "SMOKE_PASS"
    SMOKE_FAIL = "SMOKE_FAIL"


@dataclass
class Verdict:
    label: VerdictLabel
    candidate_name: str
    model_id: str
    prompt_id: str

    # Gates that failed (empty if PROMOTE or KEEP_EXPERIMENTAL for correct reasons)
    quality_failures: list[str]
    missing_required: list[str]
    improvement_met: bool
    improvement_notes: list[str]

    # Human-readable summary
    reason: str

    def __str__(self) -> str:
        lines = [
            f"Verdict: {self.label.value}",
            f"  candidate  : {self.candidate_name}",
            f"  model      : {self.model_id}",
            f"  prompt     : {self.prompt_id}",
            f"  reason     : {self.reason}",
        ]
        if self.quality_failures:
            lines.append("  quality failures:")
            for f in self.quality_failures:
                lines.append(f"    - {f}")
        if self.missing_required:
            lines.append("  missing required metrics:")
            for f in self.missing_required:
                lines.append(f"    - {f}")
        if self.improvement_notes:
            lines.append("  improvement gates:")
            for n in self.improvement_notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class Judge:
    """Evaluate a CandidateResult against a dense baseline and emit a Verdict.

    Parameters
    ----------
    stable_reference : CandidateResult, optional
        A previously promoted stable method.  If provided, the judge will
        emit REGRESSION instead of PROMOTE when the candidate is worse than
        the stable reference on any critical metric.
    strict : bool
        If ``True``, governance and proof-counter violations become hard
        ``REJECT`` verdicts instead of ``KEEP_EXPERIMENTAL``.  Used for
        benchmark harnesses that must fail closed.
    """

    def __init__(
        self,
        stable_reference: CandidateResult | None = None,
        strict: bool = False,
    ) -> None:
        self.stable_reference = stable_reference
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: CandidateResult,
        baseline: CandidateResult,
    ) -> Verdict:
        """Return a Verdict for *candidate* relative to *baseline*.

        baseline should be the dense FP16 result for the same model/prompt.
        Smoke runs are remapped to SMOKE_PASS / SMOKE_FAIL so they never
        produce a real PROMOTE verdict.
        """
        verdict = self._evaluate_inner(candidate, baseline)
        return self._remap_smoke(verdict, candidate)

    def _evaluate_inner(
        self,
        candidate: CandidateResult,
        baseline: CandidateResult,
    ) -> Verdict:
        """Core evaluation logic (no smoke remapping)."""
        quality_failures: list[str] = []
        missing_required: list[str] = []
        improvement_notes: list[str] = []

        # ---- Phase 1 governance checks (before any quality gates) ----
        # Governance is fail-closed: violations always produce REJECT.
        gov = self._check_governance(candidate)
        if gov:
            return Verdict(
                label=VerdictLabel.REJECT,
                candidate_name=candidate.candidate_name,
                model_id=candidate.model_id,
                prompt_id=candidate.prompt_id,
                quality_failures=[],
                missing_required=[],
                improvement_met=False,
                improvement_notes=[],
                reason=f"governance block: {gov}",
            )

        # ---- Phase 1b strict proof-counter checks ----
        if self.strict:
            proof_failures = self._check_proof_counters(candidate)
            if proof_failures:
                return Verdict(
                    label=VerdictLabel.REJECT,
                    candidate_name=candidate.candidate_name,
                    model_id=candidate.model_id,
                    prompt_id=candidate.prompt_id,
                    quality_failures=[],
                    missing_required=[],
                    improvement_met=False,
                    improvement_notes=[],
                    reason=f"proof failure: {'; '.join(proof_failures)}",
                )

        # 1. Check for missing required metrics
        for fname in _REQUIRED_QUALITY_FIELDS + _REQUIRED_PERF_FIELDS:
            val = getattr(candidate, fname, None)
            if val is None:
                missing_required.append(fname)

        # 2. Quality gates (only if all fields present)
        if not missing_required:
            qf = _check_quality_gates(candidate)
            quality_failures.extend(qf)

        # 3. Determine verdict
        if missing_required:
            return Verdict(
                label=VerdictLabel.KEEP_EXPERIMENTAL,
                candidate_name=candidate.candidate_name,
                model_id=candidate.model_id,
                prompt_id=candidate.prompt_id,
                quality_failures=[],
                missing_required=missing_required,
                improvement_met=False,
                improvement_notes=[],
                reason=f"missing required metric(s): {', '.join(missing_required)}",
            )

        if quality_failures:
            return Verdict(
                label=VerdictLabel.REJECT,
                candidate_name=candidate.candidate_name,
                model_id=candidate.model_id,
                prompt_id=candidate.prompt_id,
                quality_failures=quality_failures,
                missing_required=[],
                improvement_met=False,
                improvement_notes=[],
                reason=f"quality gate(s) failed: {'; '.join(quality_failures)}",
            )

        # 4. Improvement gates
        improvement_met, improvement_notes = _check_improvement_gates(
            candidate, baseline
        )

        if not improvement_met:
            return Verdict(
                label=VerdictLabel.KEEP_EXPERIMENTAL,
                candidate_name=candidate.candidate_name,
                model_id=candidate.model_id,
                prompt_id=candidate.prompt_id,
                quality_failures=[],
                missing_required=[],
                improvement_met=False,
                improvement_notes=improvement_notes,
                reason="quality-safe but no improvement gate met",
            )

        # 5. Regression check against stable reference
        if self.stable_reference is not None:
            regression_notes = _check_regression(candidate, self.stable_reference)
            if regression_notes:
                return Verdict(
                    label=VerdictLabel.REGRESSION,
                    candidate_name=candidate.candidate_name,
                    model_id=candidate.model_id,
                    prompt_id=candidate.prompt_id,
                    quality_failures=[],
                    missing_required=[],
                    improvement_met=True,
                    improvement_notes=regression_notes,
                    reason=f"regresses vs stable reference: {'; '.join(regression_notes)}",
                )

        return Verdict(
            label=VerdictLabel.PROMOTE,
            candidate_name=candidate.candidate_name,
            model_id=candidate.model_id,
            prompt_id=candidate.prompt_id,
            quality_failures=[],
            missing_required=[],
            improvement_met=True,
            improvement_notes=improvement_notes,
            reason="all quality gates and at least one improvement gate passed",
        )

    def _remap_smoke(self, verdict: Verdict, candidate: CandidateResult) -> Verdict:
        """Remap smoke-run verdicts so they never produce real PROMOTE/REJECT."""
        if getattr(candidate, "run_type", None) != "smoke":
            return verdict

        if verdict.label in (VerdictLabel.PROMOTE, VerdictLabel.KEEP_EXPERIMENTAL):
            new_label = VerdictLabel.SMOKE_PASS
            new_reason = "smoke run — harness validated (not promotion evidence)"
        else:
            new_label = VerdictLabel.SMOKE_FAIL
            new_reason = "smoke run — harness failure"

        return Verdict(
            label=new_label,
            candidate_name=verdict.candidate_name,
            model_id=verdict.model_id,
            prompt_id=verdict.prompt_id,
            quality_failures=verdict.quality_failures,
            missing_required=verdict.missing_required,
            improvement_met=verdict.improvement_met,
            improvement_notes=verdict.improvement_notes,
            reason=new_reason,
        )

    def _check_governance(self, candidate: CandidateResult) -> str:
        """Return a failure reason string, or empty string if clean.

        Governance is **always** enforced (fail-closed).  Violations produce
        ``REJECT`` regardless of *strict* mode.
        """
        if candidate.run_type == "synthetic":
            return "synthetic runs are ineligible for promotion"
        if candidate.run_type not in ("real_model", "smoke"):
            return f"unknown run_type '{candidate.run_type}' is ineligible"
        if candidate.fallback_used:
            return "fallback execution is ineligible for promotion"
        if not candidate.measured_memory:
            return "measured_memory must be True for promotion"
        if candidate.source_type != "installed_wheel":
            return "installed-wheel execution required for promotion"
        if not candidate.requested_backend or candidate.requested_backend == "unknown":
            return "requested_backend must be a known, typed value"
        if not candidate.executed_backend or candidate.executed_backend == "unknown":
            return "executed_backend must be a known, typed value"
        if candidate.requested_backend != candidate.executed_backend:
            return (
                f"backend mismatch: requested '{candidate.requested_backend}' "
                f"but executed '{candidate.executed_backend}'"
            )
        executed_lower = candidate.executed_backend.lower()
        if "reference" in executed_lower or "dense" in executed_lower:
            return "dense-reference backend aliases are ineligible for promotion"
        if "fallback" in executed_lower:
            return "fallback backend is ineligible for promotion"

        # Proof counters that must hold in all paths (not just strict mode)
        counters = candidate.proof_counters or {}
        if counters.get("requantized_tokens", 0) > 0:
            return f"requantized_tokens={counters['requantized_tokens']} > 0"
        if counters.get("fallback_attention_calls", 0) > 0:
            return f"fallback_attention_calls={counters['fallback_attention_calls']} > 0"
        if counters.get("dense_shadow_bytes", 0) > 0:
            return f"dense_shadow_bytes={counters['dense_shadow_bytes']} > 0"
        if counters.get("unknown_layer_events", 0) > 0:
            return f"unknown_layer_events={counters['unknown_layer_events']} > 0"

        # Canonical format enforcement (K8/V5/gs64 for production)
        # Benchmark-only candidates (e.g., A1 with MLX limitations) are exempt
        if candidate.key_bits is not None and candidate.key_bits != 8:
            return f"noncanonical key_bits={candidate.key_bits} (required 8)"
        if candidate.value_bits is not None:
            # Benchmark-only candidates are allowed to use V4 due to MLX limitations
            if not candidate.is_benchmark_only and candidate.value_bits != 5:
                return f"noncanonical value_bits={candidate.value_bits} (required 5 for production, V4 only allowed for benchmark-only candidates)"
            if candidate.is_benchmark_only and candidate.value_bits not in (4, 5):
                return f"noncanonical value_bits={candidate.value_bits} (benchmark-only requires 4 or 5)"
        if candidate.group_size is not None and candidate.group_size != 64:
            return f"noncanonical group_size={candidate.group_size} (required 64)"

        # Evidence hash completeness
        if not candidate.commit_hash:
            return "empty commit_hash is ineligible"
        if not candidate.corpus_hash:
            return "empty corpus_hash is ineligible"
        if not candidate.token_sequence_hash:
            return "empty token_sequence_hash is ineligible"

        # Canonical identity fields must be present for compression candidates
        # Baseline candidates (no quantization) are exempt
        is_baseline = candidate.preconditioner == "none" and candidate.quantizer == "none"
        if not is_baseline:
            if candidate.key_bits is None:
                return "key_bits is required for promotion"
            if candidate.value_bits is None:
                return "value_bits is required for promotion"
            if candidate.group_size is None:
                return "group_size is required for promotion"

        # Smoke data is not eligible for production promotion
        if not candidate.promotion_eligible:
            return "candidate is not eligible for production promotion"

        # Benchmark-only candidates are not eligible for production promotion
        if candidate.is_benchmark_only:
            return "benchmark-only candidates are not eligible for production promotion"

        return ""

    def _check_proof_counters(self, candidate: CandidateResult) -> list[str]:
        """In strict mode, proof counters must prove correct runtime structure.

        Returns a list of failure reason strings; empty list means clean.
        """
        failures: list[str] = []
        counters = candidate.proof_counters or {}

        if candidate.executed_backend == "fallback":
            failures.append("executed_backend is fallback")

        if (
            candidate.requested_backend == "metal"
            and not candidate.metal_executed
        ):
            failures.append("metal requested but not executed")

        if counters.get("requantized_tokens", 0) > 0:
            failures.append(
                f"requantized_tokens={counters['requantized_tokens']} > 0"
            )

        if counters.get("fallback_attention_calls", 0) > 0:
            failures.append(
                f"fallback_attention_calls={counters['fallback_attention_calls']} > 0"
            )

        if counters.get("dense_shadow_bytes", 0) > 0:
            failures.append(
                f"dense_shadow_bytes={counters['dense_shadow_bytes']} > 0"
            )

        if counters.get("unknown_layer_events", 0) > 0:
            failures.append(
                f"unknown_layer_events={counters['unknown_layer_events']} > 0"
            )

        return failures

    def evaluate_batch(
        self,
        candidates: list[CandidateResult],
        baseline: CandidateResult,
    ) -> list[Verdict]:
        """Evaluate a list of candidates and return verdicts in order."""
        return [self.evaluate(c, baseline) for c in candidates]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_quality_gates(c: CandidateResult) -> list[str]:
    failures: list[str] = []

    if c.logit_cosine is not None and c.logit_cosine < LOGIT_COSINE_MIN:
        failures.append(
            f"logit_cosine {c.logit_cosine:.5f} < {LOGIT_COSINE_MIN}"
        )
    if c.top5_overlap is not None and c.top5_overlap < TOP5_OVERLAP_MIN:
        failures.append(
            f"top5_overlap {c.top5_overlap:.3f} < {TOP5_OVERLAP_MIN}"
        )
    if c.attention_score_cosine is not None and c.attention_score_cosine < ATTN_COSINE_MIN:
        failures.append(
            f"attention_score_cosine {c.attention_score_cosine:.5f} < {ATTN_COSINE_MIN}"
        )
    if c.attention_top5_overlap is not None and c.attention_top5_overlap < ATTN_TOP5_MIN:
        failures.append(
            f"attention_top5_overlap {c.attention_top5_overlap:.3f} < {ATTN_TOP5_MIN}"
        )
    if c.perplexity_delta is not None and c.perplexity_delta > PERPLEXITY_DELTA_MAX:
        failures.append(
            f"perplexity_delta {c.perplexity_delta:.4f} > {PERPLEXITY_DELTA_MAX}"
        )
    if (
        c.visible_output_drift_score is not None
        and c.visible_output_drift_score > DRIFT_MAX
    ):
        failures.append(
            f"visible_output_drift_score {c.visible_output_drift_score:.3f} > {DRIFT_MAX}"
        )
    return failures


def _check_improvement_gates(
    candidate: CandidateResult,
    baseline: CandidateResult,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    met = False

    # KV memory reduction
    if (
        candidate.compressed_kv_memory_mb is not None
        and baseline.kv_cache_memory_mb is not None
        and baseline.kv_cache_memory_mb > 0
    ):
        kv_reduction = 1.0 - (
            candidate.compressed_kv_memory_mb / baseline.kv_cache_memory_mb
        )
        if kv_reduction >= KV_MEMORY_IMPROVEMENT_MIN:
            notes.append(
                f"KV memory reduced by {kv_reduction * 100:.1f}% (>= {KV_MEMORY_IMPROVEMENT_MIN * 100:.0f}%)"
            )
            met = True
        else:
            notes.append(
                f"KV memory reduced by only {kv_reduction * 100:.1f}% (need >= {KV_MEMORY_IMPROVEMENT_MIN * 100:.0f}%)"
            )

    # Peak memory reduction
    if (
        candidate.peak_memory_mb is not None
        and baseline.peak_memory_mb is not None
        and baseline.peak_memory_mb > 0
    ):
        peak_reduction = 1.0 - (candidate.peak_memory_mb / baseline.peak_memory_mb)
        if peak_reduction >= PEAK_MEMORY_IMPROVEMENT_MIN:
            notes.append(
                f"peak memory reduced by {peak_reduction * 100:.1f}% (>= {PEAK_MEMORY_IMPROVEMENT_MIN * 100:.0f}%)"
            )
            met = True
        else:
            notes.append(
                f"peak memory reduced by only {peak_reduction * 100:.1f}% (need >= {PEAK_MEMORY_IMPROVEMENT_MIN * 100:.0f}%)"
            )

    # Decode TPS improvement
    if (
        candidate.decode_tps is not None
        and baseline.decode_tps is not None
        and baseline.decode_tps > 0
    ):
        tps_gain = (candidate.decode_tps - baseline.decode_tps) / baseline.decode_tps
        if tps_gain >= DECODE_TPS_IMPROVEMENT_MIN:
            notes.append(
                f"decode TPS improved by {tps_gain * 100:.1f}% (>= {DECODE_TPS_IMPROVEMENT_MIN * 100:.0f}%)"
            )
            met = True
        else:
            notes.append(
                f"decode TPS improved by only {tps_gain * 100:.1f}% (need >= {DECODE_TPS_IMPROVEMENT_MIN * 100:.0f}%)"
            )

    # Long-context enablement (SnapKV)
    if candidate.snapkv_enabled and (
        candidate.snapkv_memory_saved_mb is not None
        and candidate.snapkv_memory_saved_mb > 0
    ):
        notes.append(
            f"enables longer context via SnapKV (memory saved: {candidate.snapkv_memory_saved_mb:.1f} MB)"
        )
        met = True

    if not notes:
        notes.append("no improvement gate measured")

    return met, notes


def _check_regression(
    candidate: CandidateResult,
    stable: CandidateResult,
) -> list[str]:
    """Return non-empty list if candidate is worse than the stable reference on key metrics."""
    issues: list[str] = []

    def _worse(c_val: float | None, s_val: float | None, name: str, higher_is_better: bool = True) -> None:
        if c_val is None or s_val is None:
            return
        if higher_is_better and c_val < s_val - 1e-6:
            issues.append(f"{name} {c_val:.5f} < stable {s_val:.5f}")
        elif not higher_is_better and c_val > s_val + 1e-6:
            issues.append(f"{name} {c_val:.5f} > stable {s_val:.5f}")

    _worse(candidate.logit_cosine, stable.logit_cosine, "logit_cosine")
    _worse(candidate.top5_overlap, stable.top5_overlap, "top5_overlap")
    _worse(candidate.attention_score_cosine, stable.attention_score_cosine, "attention_score_cosine")
    _worse(candidate.perplexity_delta, stable.perplexity_delta, "perplexity_delta", higher_is_better=False)
    return issues
