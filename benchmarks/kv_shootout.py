#!/usr/bin/env python3
"""KV-cache compression shootout benchmark.

Compares all compression candidates on the same models and prompts,
applies quality gates, and selects the winner.

Usage
-----
    # Quick sanity run (fewer prompts, small model only)
    python benchmarks/kv_shootout.py --quick

    # Full run with real logit gate
    python benchmarks/kv_shootout.py --full-logit-gate

    # Memory report
    python benchmarks/kv_shootout.py --memory-report

    # Promotion report (only promotion-eligible candidates)
    python benchmarks/kv_shootout.py --promotion-report

    # Governance-only mode (no MLX required, for CI smoke testing)
    python benchmarks/kv_shootout.py --governance-only

    # Specific model only
    python benchmarks/kv_shootout.py --model Qwen/Qwen2.5-1.5B-Instruct

Outputs
-------
    artifacts/bench/shootout/quick/results.json
    artifacts/bench/shootout/full_logit/results.json
    artifacts/bench/shootout/memory/results.json
    artifacts/bench/shootout/promotion/results.json

Decision rule
-------------
The candidate with the best quality-gated tokens/sec wins.
If no candidate beats dense_mlx_baseline in quality, the baseline wins.

Metric definitions
------------------
size_ratio        = compressed_size / baseline_size   (lower is better)
compression_factor = baseline_size / compressed_size  (higher is better)

Do NOT say "0.265x compression". Say:
    Compressed size: 26.5% of FP16  (3.77x smaller)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Suppress mlx-lm legacy sampling-arg deprecation warnings emitted during
# generation. These are known and do not affect results.
warnings.filterwarnings(
    "ignore", message="Specifying sampling arguments", category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Specifying ``repetition_penalty``",
    category=UserWarning,
)

# Add repo root to path so rfsn_v11 is importable without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfsn_v11.candidates.artifact_utils import (  # noqa: E402
    _build_honest_markdown_table,
    _export_rfsn_v10_proof_trace,
    _export_winner,
)
from rfsn_v11.candidates.base import (  # noqa: E402
    CandidateResult,
    KVCompressionCandidate,
)
from rfsn_v11.candidates.candidate_status import (  # noqa: E402
    CandidateStatus,
)
from rfsn_v11.candidates.json_utils import dump_json_strict  # noqa: E402
from rfsn_v11.candidates.logit_capture import (  # noqa: E402
    capture_teacher_forced_logprobs,
    compute_logit_metrics_from_logprobs,
)
from rfsn_v11.candidates.promotion_policy import (  # noqa: E402
    evaluate_promotion_eligibility as evaluate_promotion_policy,
)
from rfsn_v11.candidates.quality_gates import (  # noqa: E402
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_PENDING_LOGIT_GATE,
    GATE_STATUS_PENDING_MEMORY_METRICS,
    GATE_STATUS_PENDING_REAL_CACHE_INJECTION,
    LogitGateThresholds,
    evaluate_quality_gate,
)
from rfsn_v11.candidates.quality_gates import (
    compute_promotion_eligibility as compute_quality_gate_eligibility,
)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# primary: head_dim=128, ideal for TQ rotation
MODELS_FULL = [
    "Qwen/Qwen2.5-1.5B-Instruct",
]
MODELS_QUICK = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # quick iteration: head_dim=64, fast load
]

PROMPTS_QUICK = [
    "Hello",
    "Write a Python function that adds two numbers.",
]

MAX_TOKENS_FULL = 50
MAX_TOKENS_QUICK = 50

PROMPT_SUITE: dict[str, list[str]] = {
    "short_chat": [
        "Hello",
        "What is 2 + 2?",
    ],
    "coding": [
        "Write a Python function that adds two numbers.",
        "Write a Python class for a min-heap with push and pop methods.",
        "Implement binary search in Python with type hints.",
    ],
    "summarization": [
        "Summarize this paragraph in one sentence.",
        "In one sentence, what is machine learning?",
    ],
    "long_context": [
        "Explain the difference between RAM and storage in detail.",
        "Describe the history of the internet from ARPANET to modern day.",
    ],
    "math": [
        "Solve step by step: if x^2 - 5x + 6 = 0, what are the values of x?",
        "Explain why 0.1 + 0.2 != 0.3 in floating point arithmetic.",
    ],
    "multi_turn": [
        (
            "User: What is the capital of France?\n"
            "Assistant: Paris.\n"
            "User: And what language do they speak there?"
        ),
    ],
}

# Flat list for non-quick full runs (one prompt per category)
PROMPTS_FULL = [prompts[0] for prompts in PROMPT_SUITE.values()]

# Temperature=0.0 for all candidates to make text comparable across methods.
# Without greedy decoding, stochastic sampling causes false
# text-heuristic FAILs.
GENERATION_TEMP = 0.0

ARTIFACTS_ROOT = Path("artifacts/bench/shootout")


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file for provenance tracking."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------

def _build_candidates(
    quick: bool = False,
    include_legacy: bool = False,
    bit_width_config: str = "k8v8",
    check_available: bool = True,
    canonical: bool = False,
    experimental: bool = False,
) -> list[KVCompressionCandidate]:
    """Instantiate candidates using the authoritative registry.

    Fix #8: Delegates to candidate_registry.  Phase 0: default returns exactly
    three canonical candidates.  experimental=True unlocks the full registry.

    Args:
        quick: If True, use quick-mode candidate selection (smoke BS8).
        include_legacy: Legacy flag; ignored in default (frozen) mode.
        bit_width_config: Bit-width for experimental ladder (ignored in default mode).
        check_available: If True, filter by is_available(). If False, return declared.
        canonical: If True, force canonical BS64 (overrides quick mode).
        experimental: Phase 0: If True, use experimental registry with all candidates.
    """
    from benchmarks.candidate_registry import (
        get_experimental_registry,
        get_registry,
    )

    if experimental:
        registry = get_experimental_registry()
        # Experimental mode: baseline + MLX-LM 8-bit + selected bit-width + legacy
        baseline = registry.get("dense_mlx_baseline")
        bit_config_to_name = {
            "k16v16": "rfsn_direct_packed_k16v16",
            "k8v16": "rfsn_direct_packed_k8v16",
            "k16v8": "rfsn_direct_packed_k16v8",
            "k8v8": "rfsn_direct_packed_k8v8",
            "k8v6": "rfsn_direct_packed_k8v6",
            "k8v5": "rfsn_direct_packed_k8v5",
        }
        if quick and not canonical:
            bit_config_to_name["k8v8"] = "rfsn_direct_packed_k8v8_smoke"
        candidate_name = bit_config_to_name.get(bit_width_config, "rfsn_direct_packed_k8v8")
        all_candidates: list[KVCompressionCandidate] = [
            baseline,
            registry.get("mlx_lm_8bit_kv"),
            registry.get(candidate_name),
        ]
        if include_legacy:
            legacy_names = [
                "A1_wht_grouped_k8v4_gs64",
                "A1b_wht_asym_k8v4",
                "A2_wht_polar_4bit",
                "A3_wht_turboquant_mse_4bit",
                "A4_wht_turboquant_qjl_4bit",
                "B1_sparsejl_grouped_k8v4_gs64",
                "R1_wht_grouped_residual128",
                "R2_turboquant_mse_residual128",
                "S1_snapkv_prune_only",
                "S2_snapkv_plus_grouped",
                "S3_snapkv_plus_turboquant_mse_residual128",
                "rfsn_v10_k8v5",
                "rfsn_v10_k8v8",
            ]
            for name in legacy_names:
                try:
                    all_candidates.append(registry.get(name))
                except KeyError:
                    pass
    else:
        # Phase 0 default: exactly three canonical candidates
        registry = get_registry()
        # Map quick mode to smoke variant inside the canonical candidate
        if quick and not canonical:
            # Smoke variant: use the smoke factory directly from experimental registry
            # but default registry only has canonical.  We temporarily fetch smoke.
            from benchmarks.candidate_registry import get_experimental_registry
            exp_reg = get_experimental_registry()
            direct_packed = exp_reg.get("rfsn_direct_packed_k8v8_smoke")
        else:
            direct_packed = registry.get("rfsn_direct_packed_k8v8")
        all_candidates = [
            registry.get("dense_mlx_baseline"),
            registry.get("mlx_lm_8bit_kv"),
            direct_packed,
        ]

    if not check_available:
        return all_candidates

    available = []
    for c in all_candidates:
        if c.is_available():
            available.append(c)
        else:
            print(f"  [skip] {c.name}: not available in this environment")
    return available


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load model and tokenizer via mlx_lm."""
    try:
        import mlx_lm
        print(f"\nLoading {model_id} ...")
        model, tokenizer = mlx_lm.load(model_id)
        return model, tokenizer
    except Exception as exc:
        print(f"  ERROR loading {model_id}: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _peak_memory_mb() -> float | None:
    """Return peak MLX memory usage in MB if available."""
    try:
        import mlx.core as mx
        return mx.metal.get_peak_memory() / (1024 ** 2)
    except Exception:
        return None


def _reset_peak_memory() -> None:
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_once(
    candidate: KVCompressionCandidate,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    baseline_result: CandidateResult | None,
    temp: float = GENERATION_TEMP,
    mode: str = "quick",
    require_compressed_execution: bool = False,
) -> CandidateResult:
    """Run one candidate on one prompt and apply quality gate.
    
    Fix #5: Added require_compressed_execution parameter for hard compressed-execution gate.
    """
    _reset_peak_memory()

    # Wrap candidate.run() with structured error handling
    try:
        result = candidate.run(model, tokenizer, prompt, max_tokens, temp=temp)
    except Exception as exc:
        import traceback
        # Return structured error result instead of letting exception propagate
        result = CandidateResult(
            name=candidate.name,
            model_id=getattr(model, "name_or_path", "unknown"),
            prompt=prompt,
            gate_status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
            promotion_eligible=False,
        )
        # Store detailed error info in notes field for now
        result.notes = f"Exception in {type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    peak_mb = _peak_memory_mb()
    if peak_mb is not None:
        result.working_set_memory_mb = peak_mb

    # Fix #5: Hard compressed-execution gate
    # Only applies to packed candidates, not baseline.
    # NOTE: full_history_materialization_calls > 0 is allowed because the
    # current Metal path honestly records dense reconstruction. The promotion
    # policy will reject such candidates; the gate only checks that counters
    # are present and no silent dense fallback occurred.
    if require_compressed_execution and candidate.name != "dense_mlx_baseline":
        if not (
            result.packed_blocks_created > 0
            and result.packed_blocks_read > 0
            and result.packed_attention_calls > 0
            and result.packed_bytes_written > 0
            and result.packed_bytes_read > 0
            and result.dense_fallback_calls == 0
        ):
            result.gate_status = "ERROR"
            result.promotion_eligible = False
            result.error = (
                f"Compressed-execution gate failed: "
                f"packed_blocks_created={result.packed_blocks_created}, "
                f"packed_blocks_read={result.packed_blocks_read}, "
                f"packed_attention_calls={result.packed_attention_calls}, "
                f"packed_bytes_written={result.packed_bytes_written}, "
                f"packed_bytes_read={result.packed_bytes_read}, "
                f"dense_fallback_calls={result.dense_fallback_calls}"
            )
            return result

    # Error gate
    if result.error or result.gate_status == "ERROR":
        result.gate_status = "ERROR"
        result.promotion_eligible = False
        return result

    # Preserve adapter-specific pending statuses
    # (more specific than generic logic)
    if result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION:
        result.promotion_eligible = False
        return result

    # Baseline always passes logit gate by definition, but CONTROL never
    # promotes — it is the comparison target, not a candidate.
    if candidate.name == "dense_mlx_baseline":
        result.logit_cosine = 1.0
        result.kl_divergence = 0.0
        result.top1_match = 1.0
        result.top5_overlap = 1.0
        result.top10_overlap = 1.0
        result.max_logit_delta = 0.0
        result.first_divergent_token = None
        result.logit_gate_passed = True
        result.memory_gate_passed = True
        result.gate_status = "PASS_NO_PROMOTE"
        result.promotion_eligible = False
        result.candidate_status = "CONTROL"
        # Set packed fields to 0 for baseline (no compression)
        result.packed_blocks_created = 0
        result.packed_blocks_read = 0
        result.packed_attention_calls = 0
        result.dense_fallback_calls = 0
        result.full_history_materialization_calls = 0
        result.packed_bytes_written = 0
        result.packed_bytes_read = 0
        return result

    # In quick mode, we only have text heuristic — no real logit gate
    if mode == "quick":
        if baseline_result is not None and baseline_result.generated_text:
            result = _text_quality_heuristic(result, baseline_result)
        else:
            result.text_heuristic_passed = None
            result.logit_gate_passed = None
            result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
            result.promotion_eligible = False
            result.notes += "  [quick mode: logit gate pending]"
        return result

    # full-logit-gate mode: teacher-forced logit comparison.
    # We run the baseline greedy decode once, then force-feed the exact
    # same token sequence through the candidate to get comparable logits.
    # This avoids the cascade-divergence problem of independent greedy
    # decodes (where a single differing token makes all subsequent logits
    # incomparable).
    if mode == "full_logit":
        baseline_text = (
            baseline_result.generated_text
            if baseline_result is not None else ""
        )
        baseline_logprobs = None
        candidate_logprobs = None

        if candidate.name == "dense_mlx_baseline":
            # Baseline is the reference — perfect by definition
            result.logit_cosine = 1.0
            result.kl_divergence = 0.0
            result.top1_match = 1.0
            result.top5_overlap = 1.0
            result.top10_overlap = 1.0
            result.max_logit_delta = 0.0
            result.first_divergent_token = None
            result.logit_gate_passed = True
            result.memory_gate_passed = True
            result.gate_status = "PASS_NO_PROMOTE"
            result.promotion_eligible = False
            result.candidate_status = "CONTROL"
            # Set packed fields to 0 for baseline (no compression)
            result.packed_blocks_created = 0
            result.packed_blocks_read = 0
            result.packed_attention_calls = 0
            result.dense_fallback_calls = 0
            result.full_history_materialization_calls = 0
            result.packed_bytes_written = 0
            result.packed_bytes_read = 0
            return result

        if not baseline_text:
            result.logit_gate_passed = None
            result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
            result.promotion_eligible = False
            result.notes += (
                "  [full-logit-gate: no baseline text for comparison]"
            )
            return result

        # Baseline teacher-forced logprobs (standard FP16 cache)
        baseline_logprobs = capture_teacher_forced_logprobs(
            model, tokenizer, prompt, baseline_text,
        )

        # Candidate teacher-forced logprobs
        if candidate.name == "mlx_lm_quantized_kv_b8":
            candidate_logprobs = capture_teacher_forced_logprobs(
                model, tokenizer, prompt, baseline_text,
                kv_bits=8, kv_group_size=64,
            )
        elif candidate.name in (
            "turboquant_v2_b4_gs64_rot",
            "turboquant_v2_b4_gs64_norot",
            "turboquant_v2_b6_gs64_rot",
            "turboquant_v2_b6_gs64_norot",
        ):
            # TurboQuant V2 builds its own cache and patches SDPA
            tq_candidate = candidate
            try:
                import sys

                import turboquant.patch as tq_patch

                head_dim = tq_candidate._detect_head_dim(model)
                use_rotation = head_dim >= 128
                caches = tq_candidate._build_cache(
                    model, head_dim, use_rotation,
                )
                tq_patch.apply()
                import mlx_lm.models.base as _base
                _patched_fn = _base.scaled_dot_product_attention
                for _mod_name, _mod in list(sys.modules.items()):
                    if (
                        _mod_name.startswith("mlx_lm.models.")
                        and _mod is not None
                    ):
                        if hasattr(_mod, "scaled_dot_product_attention"):
                            _mod.scaled_dot_product_attention = _patched_fn
                try:
                    candidate_logprobs = capture_teacher_forced_logprobs(
                        model, tokenizer, prompt, baseline_text,
                        prompt_cache=caches,
                    )
                finally:
                    tq_patch.revert()
                    _orig_fn = _base.scaled_dot_product_attention
                    for _mod_name, _mod in list(sys.modules.items()):
                        if (
                            _mod_name.startswith("mlx_lm.models.")
                            and _mod is not None
                        ):
                            if hasattr(_mod, "scaled_dot_product_attention"):
                                _mod.scaled_dot_product_attention = _orig_fn
            except Exception:
                candidate_logprobs = None
        elif candidate.name in (
            "polar_reference_offline_b4_d128",
        ):
            # Polar builds its own cache and patches SDPA
            polar_candidate = candidate
            try:
                from mlx_turboquant.cache import TurboQuantKVCache

                from rfsn_v11.candidates.polar_reference_adapter import (
                    _apply_polar_patch,
                    _revert_polar_patch,
                )

                head_dim = polar_candidate._detect_head_dim(model)
                n_layers = len(model.layers)
                caches = [
                    TurboQuantKVCache(
                        bits=polar_candidate.bits,
                        head_dim=head_dim,
                        key_seed=polar_candidate.seed + i,
                        value_seed=polar_candidate.seed + i + 1000,
                    )
                    for i in range(n_layers)
                ]
                _apply_polar_patch()
                try:
                    candidate_logprobs = capture_teacher_forced_logprobs(
                        model, tokenizer, prompt, baseline_text,
                        prompt_cache=caches,
                    )
                finally:
                    _revert_polar_patch()
            except Exception:
                candidate_logprobs = None
        elif getattr(candidate, "supports_teacher_forced_capture", False):
            # Capability-based dispatch: any candidate that declares support
            # for teacher-forced capture can be called here.
            candidate_logprobs = candidate.capture_logprobs(
                model, tokenizer, prompt, baseline_text,
            )
            if candidate_logprobs is not None and candidate.name.startswith("rfsn_v10"):
                runtime_counters = getattr(
                    candidate, "_last_runtime_counters", None
                )
                _export_rfsn_v10_proof_trace(
                    candidate_name=candidate.name,
                    model=model,
                    config_name=getattr(candidate, "config_name", ""),
                    actual_kv_memory_mb=result.actual_kv_memory_mb,
                    runtime_counters=runtime_counters,
                )
        else:
            # Any other candidate without a capture path
            result.logit_gate_passed = None
            result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
            result.promotion_eligible = False
            result.notes += (
                "  [full-logit-gate: teacher-forced capture not available]"
            )
            return result

        if baseline_logprobs is not None and candidate_logprobs is not None:
            metrics = compute_logit_metrics_from_logprobs(
                baseline_logprobs, candidate_logprobs,
            )
            result.logit_cosine = metrics.get("logit_cosine")
            result.kl_divergence = metrics.get("kl_divergence")
            result.top1_match = metrics.get("top1_match")
            result.top5_overlap = metrics.get("top5_overlap")
            result.top10_overlap = metrics.get("top10_overlap")
            result.max_logit_delta = metrics.get("max_logit_delta")
            result.first_divergent_token = metrics.get("first_divergent_token")
            gate = evaluate_quality_gate(metrics)
            result.logit_gate_passed = gate.passed
            if not gate.passed:
                result.gate_status = GATE_STATUS_FAIL
                result.promotion_eligible = False
                result.failed_gate_reasons = gate.failure_reasons
                reasons = "; ".join(gate.failure_reasons)
                result.notes += (
                    f"  [logit gate failed: {reasons}]"
                )
            else:
                result.memory_gate_passed = (
                    result.actual_kv_memory_mb is not None
                    and result.working_set_memory_mb is not None
                    and result.size_ratio is not None
                    and result.compression_factor is not None
                )
                promotion_eligible, gate_status = (
                    compute_quality_gate_eligibility(
                        logit_gate_passed=result.logit_gate_passed,
                        memory_gate_passed=result.memory_gate_passed,
                        actual_kv_memory_mb=result.actual_kv_memory_mb,
                        working_set_memory_mb=result.working_set_memory_mb,
                        size_ratio=result.size_ratio,
                        compression_factor=result.compression_factor,
                    )
                )
                result.promotion_eligible = promotion_eligible
                result.gate_status = gate_status
        else:
            result.logit_gate_passed = None
            result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
            result.promotion_eligible = False
            result.notes += (
                "  [full-logit-gate: teacher-forced capture failed]"
            )
    elif result.logit_cosine is None:
        result.logit_gate_passed = None
        result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
        result.promotion_eligible = False
        result.notes += "  [full-logit-gate: no logits captured]"
    else:
        metrics = {
            "logit_cosine": result.logit_cosine,
            "kl_divergence": result.kl_divergence,
            "top5_overlap": result.top5_overlap,
            "top10_overlap": result.top10_overlap,
            "max_logit_delta": result.max_logit_delta,
            "first_divergent_token": result.first_divergent_token,
        }
        gate = evaluate_quality_gate(metrics)
        result.logit_gate_passed = gate.passed
        if not gate.passed:
            result.gate_status = GATE_STATUS_FAIL
            result.promotion_eligible = False
            result.failed_gate_reasons = gate.failure_reasons
            reasons = "; ".join(gate.failure_reasons)
            result.notes += f"  [logit gate failed: {reasons}]"
        else:
            result.memory_gate_passed = (
                result.actual_kv_memory_mb is not None
                and result.working_set_memory_mb is not None
                and result.size_ratio is not None
                and result.compression_factor is not None
            )
            promotion_eligible, gate_status = compute_quality_gate_eligibility(
                logit_gate_passed=result.logit_gate_passed,
                memory_gate_passed=result.memory_gate_passed,
                actual_kv_memory_mb=result.actual_kv_memory_mb,
                working_set_memory_mb=result.working_set_memory_mb,
                size_ratio=result.size_ratio,
                compression_factor=result.compression_factor,
            )
            result.promotion_eligible = promotion_eligible
            result.gate_status = gate_status

    # Memory-report mode: every candidate must report memory metrics
    if mode == "memory":
        if result.actual_kv_memory_mb is None:
            result.memory_gate_passed = False
            result.gate_status = GATE_STATUS_PENDING_MEMORY_METRICS
            result.promotion_eligible = False
            result.notes += "  [memory report: actual_kv_memory_mb missing]"
        elif result.working_set_memory_mb is None:
            result.memory_gate_passed = False
            result.gate_status = GATE_STATUS_PENDING_MEMORY_METRICS
            result.promotion_eligible = False
            result.notes += "  [memory report: working_set_memory_mb missing]"
        elif result.size_ratio is None or result.compression_factor is None:
            result.memory_gate_passed = False
            result.gate_status = GATE_STATUS_PENDING_MEMORY_METRICS
            result.promotion_eligible = False
            result.notes += (
                "  [memory report: size_ratio/"
                "compression_factor missing]"
            )
        else:
            result.memory_gate_passed = True
            # Even with memory gate passed, logit gate may still be pending
            promotion_eligible, gate_status = compute_quality_gate_eligibility(
                logit_gate_passed=result.logit_gate_passed,
                memory_gate_passed=True,
                actual_kv_memory_mb=result.actual_kv_memory_mb,
                working_set_memory_mb=result.working_set_memory_mb,
                size_ratio=result.size_ratio,
                compression_factor=result.compression_factor,
            )
            result.promotion_eligible = promotion_eligible
            result.gate_status = gate_status

    # Enforce promotion rules based on candidate status
    result = _apply_promotion_rules(result, candidate)
    return result


def _apply_promotion_rules(
    result: CandidateResult, candidate: KVCompressionCandidate
) -> CandidateResult:
    """Enforce candidate-status-based promotion rules.

    Only EXPERIMENTAL or BASELINE candidates can become PROMOTED.
    OFFLINE_ONLY cannot promote until real cache injection exists.
    REFERENCE_ONLY cannot promote unless upgraded into a real
    runtime candidate.
    CONTROL does not promote; it is the comparison target.
    """
    status = result.candidate_status or candidate.candidate_status
    if status in (CandidateStatus.CONTROL, CandidateStatus.REFERENCE_ONLY):
        result.promotion_eligible = False
        if result.gate_status == GATE_STATUS_PASS:
            result.gate_status = "PASS_NO_PROMOTE"
    elif status == CandidateStatus.OFFLINE_ONLY:
        result.promotion_eligible = False
        result.gate_status = GATE_STATUS_PENDING_REAL_CACHE_INJECTION
    elif status == CandidateStatus.FAILED:
        result.promotion_eligible = False
    elif status in (
        CandidateStatus.PROMOTED,
        CandidateStatus.PROMOTION_ELIGIBLE,
    ):
        # Already elevated; preserve computed eligibility
        pass
    # EXPERIMENTAL and BASELINE keep their computed eligibility
    return result


def _text_quality_heuristic(
    result: CandidateResult,
    baseline: CandidateResult,
) -> CandidateResult:
    """Compare generated text to baseline without real logits.

    This is a heuristic only. A candidate that passes here is NOT
    promotion eligible until the real logit gate runs.
    """
    if not baseline.generated_text or not result.generated_text:
        result.text_heuristic_passed = None
        result.logit_gate_passed = None
        result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
        result.promotion_eligible = False
        result.notes += "  [text heuristic: no text to compare]"
        return result

    baseline_tokens = baseline.generated_text.split()
    result_tokens = result.generated_text.split()

    # Exact match check
    if baseline.generated_text == result.generated_text:
        result.text_heuristic_passed = True
        result.logit_gate_passed = None
        result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
        result.promotion_eligible = False
        result.notes += (
            "  [text heuristic: exact match — "
            "logit gate still pending]"
        )
        return result

    # Divergence at token level
    first_diff = None
    for i, (b, r) in enumerate(zip(baseline_tokens, result_tokens)):
        if b != r:
            first_diff = i
            break

    if first_diff is None and len(baseline_tokens) != len(result_tokens):
        first_diff = min(len(baseline_tokens), len(result_tokens))

    result.first_divergent_token = first_diff
    result.text_heuristic_passed = False
    result.logit_gate_passed = None
    result.gate_status = GATE_STATUS_PENDING_LOGIT_GATE
    result.promotion_eligible = False
    result.notes += (
        f"  [text heuristic: diverged at token {first_diff}"
        " — logit gate pending]"
    )
    return result


# ---------------------------------------------------------------------------
# Aggregated reporting
# ---------------------------------------------------------------------------

def _aggregate(results: list[CandidateResult]) -> dict[str, Any]:
    """Aggregate a list of per-prompt results into one summary row."""
    if not results:
        return {}

    name = results[0].name
    model_id = results[0].model_id
    candidate_status = results[0].candidate_status

    # Average numeric fields
    numeric = [
        "total_ms",
        "tokens_per_sec",
        "actual_kv_memory_mb",
        "working_set_memory_mb",
        "size_ratio",
        "compression_factor",
        "logit_cosine",
        "kl_divergence",
        "top1_match",
        "top5_overlap",
        "top10_overlap",
        "max_logit_delta",
    ]
    agg: dict[str, Any] = {
        "name": name,
        "model_id": model_id,
        "candidate_status": str(candidate_status),
    }
    for field in numeric:
        vals = [
            getattr(r, field) for r in results
            if getattr(r, field) is not None
        ]
        agg[field] = float(np.mean(vals)) if vals else None

    # Preserve required promotion fields (do not average, use first or sum)
    # Fix P0 #12, #13: Preserve fields needed by promotion policy
    required_promotion_fields = [
        "packed_attention_calls",
        "dense_fallback_calls",
        "full_history_materialization_calls",
        "packed_blocks_created",
        "packed_blocks_read",
        "packed_bytes_written",
        "packed_bytes_read",
        "measurement_kind",
        "execution_backend",
        "cache_backend_used",
        "error",
    ]
    for field in required_promotion_fields:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if vals:
            # For numeric fields, sum them; for strings, use the first non-empty
            if field in ("packed_attention_calls", "dense_fallback_calls",
                        "full_history_materialization_calls", "packed_blocks_created",
                        "packed_blocks_read", "packed_bytes_written", "packed_bytes_read"):
                agg[field] = sum(v for v in vals if isinstance(v, (int, float)))
            else:
                agg[field] = vals[0]

    # Gate status: if any FAIL → overall FAIL; if all PASS → PASS;
    # otherwise pending
    statuses = [r.gate_status for r in results]
    if any(s == GATE_STATUS_FAIL for s in statuses):
        agg["gate_status"] = GATE_STATUS_FAIL
    elif all(s == GATE_STATUS_PASS for s in statuses):
        agg["gate_status"] = GATE_STATUS_PASS
    else:
        pending = [s for s in statuses if s != GATE_STATUS_PASS]
        agg["gate_status"] = pending[0] if pending else GATE_STATUS_PASS

    # REFERENCE_ONLY candidates are never promotion-eligible
    if candidate_status == CandidateStatus.REFERENCE_ONLY:
        agg["promotion_eligible"] = False
        agg["promotion_blocked_reason"] = (
            "REFERENCE_ONLY candidates are not eligible for speed or memory promotion"
        )
    else:
        agg["promotion_eligible"] = all(r.promotion_eligible for r in results)

    agg["real_cache_used"] = any(
        r.cache_backend_used and r.cache_backend_used != ""
        and r.cache_backend_used != "rfsn_v11_offline"
        for r in results
    )
    agg["count"] = len(results)
    agg["notes"] = " | ".join({r.notes for r in results if r.notes})

    # Aggregate failed_gate_reasons from all per-prompt results
    all_reasons: list[str] = []
    for r in results:
        if r.failed_gate_reasons:
            all_reasons.extend(r.failed_gate_reasons)
    agg["failed_gate_reasons"] = (
        sorted(set(all_reasons)) if all_reasons else []
    )
    return agg


_ARTIFACT_METHODLOGY = "teacher_forced_logit_v1"


_METHODOLOGY_STATUS_STALE = "STALE_PRE_TEACHER_FORCED"
_METHODOLOGY_STATUS_RERUN_NO_PROMO = (
    "TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION"
)
_METHODOLOGY_STATUS_RERUN_PROMO = (
    "TEACHER_FORCED_RERUN_COMPLETE_PROMOTION_ALLOWED"
)
_METHODOLOGY_STATUS_RERUN_INCOMPLETE_NO_PROMO = (
    "TEACHER_FORCED_RERUN_INCOMPLETE_NO_PROMOTION"
)
_METHODOLOGY_STATUS_RERUN_FAILED = "TEACHER_FORCED_RERUN_FAILED"


def _artifact_metadata(
    mode: str,
    token_sequence_hash: str = "",
    promotion_allowed: bool = False,
    gate_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    if promotion_allowed:
        methodology_status = _METHODOLOGY_STATUS_RERUN_PROMO
    elif not token_sequence_hash:
        methodology_status = _METHODOLOGY_STATUS_RERUN_INCOMPLETE_NO_PROMO
    else:
        methodology_status = _METHODOLOGY_STATUS_RERUN_NO_PROMO
    return {
        "benchmark_methodology": _ARTIFACT_METHODLOGY,
        "methodology_status": methodology_status,
        "baseline_decode_mode": "greedy",
        "comparison_mode": "same_token_sequence",
        "token_sequence_hash": token_sequence_hash,
        "promotion_allowed": promotion_allowed,
        "artifact_schema_version": "2.0",
        "mode": mode,
        "gate_thresholds": gate_thresholds or {},
        "note": (
            "Teacher-forced alignment fixed, but promotion remains disabled "
            "pending non-empty token_sequence_hash and "
            "runtime-instrumented proof trace."
            if not token_sequence_hash
            else (
                "Teacher-forced logit gate active. "
                "Artifacts are current and promotion is evaluated "
                "under the corrected gate."
            )
        ),
    }


def _write_artifacts(
    rows: list[dict[str, Any]],
    out_dir: Path,
    mode: str = "quick",
    token_sequence_hash: str = "",
    token_sequence_reference: dict[str, Any] | None = None,
    promotion_allowed: bool = False,
) -> None:
    """Write JSON, CSV, and Markdown artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_thresholds = LogitGateThresholds().to_dict()

    # Build metadata with proper token provenance
    metadata = _artifact_metadata(
        mode=mode,
        token_sequence_hash=token_sequence_hash,
        promotion_allowed=promotion_allowed,
        gate_thresholds=gate_thresholds,
    )

    # Include token sequence reference if available (for non-full-logit modes)
    if token_sequence_reference:
        metadata["token_sequence_provenance"] = token_sequence_reference

    payload = {
        "metadata": metadata,
        "results": rows,
    }

    # JSON (strict — no NaN / Infinity)
    json_path = out_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        dump_json_strict(payload, fh, indent=2, default=str)

    # CSV (use rows only, not metadata wrapper)
    if rows:
        csv_path = out_dir / "results.csv"
        # Union of all keys across rows so extra fields don't crash the writer
        headers = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    # Markdown
    md_path = out_dir / "results.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(
            _build_honest_markdown_table(
                rows, promotion_allowed=promotion_allowed,
            )
        )
        fh.write("\n## Notes\n\n")
        fh.write(
            "**Methodology:** "
            f"`{payload['metadata']['benchmark_methodology']}`  \n"
            f"**Promotion allowed:** "
            f"{payload['metadata']['promotion_allowed']}  \n"
            f"**Schema version:** "
            f"{payload['metadata']['artifact_schema_version']}  \n"
            "\n"
            "**Working-set memory measurement mode dependency**: "
            "Baseline working-set memory differs between full-logit mode "
            "(~975 MB) and memory-report mode (~1422 MB). This is due to "
            "different run paths, model warmup states, prompt lengths, and "
            "sampling timing. Working-set memory should be treated as "
            "measurement-mode dependent, not promotion-critical. "
            "Actual KV cache bytes (actual_kv_memory_mb) are the stable "
            "compression proof.\n"
        )
        token_sequence_hash = payload["metadata"].get(
            "token_sequence_hash", ""
        )
        if token_sequence_hash:
            fh.write(
                f"**Token sequence hash:** `{token_sequence_hash}`  \n"
            )
        else:
            fh.write(
                "**Token sequence hash:** *empty* — promotion blocked until "
                "teacher-forced rerun produces a non-empty hash.\n"
            )
        # Defensive: never write stale promotion claims when
        # promotion is globally disallowed.
        if not promotion_allowed:
            fh.write(
                "**Current status:** No candidate is promotion eligible. "
                "Official promoted candidate: NONE.\n"
            )

    print(f"  Wrote {json_path}, {csv_path if rows else ''}, {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KV-cache compression shootout"
    )
    parser.add_argument("--quick", action="store_true", help="Fast smoke run")
    parser.add_argument(
        "--canonical", action="store_true",
        help="P0 Fix: Use canonical BS64 configuration (overrides quick mode smoke variant)",
    )
    parser.add_argument(
        "--full-logit-gate", action="store_true",
        help="Run real logit comparison",
    )
    parser.add_argument(
        "--memory-report", action="store_true",
        help="Require all candidates to report memory metrics",
    )
    parser.add_argument(
        "--promotion-report", action="store_true",
        help="Only rank promotion-eligible candidates",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model ID",
    )
    parser.add_argument(
        "--include-legacy", action="store_true",
        help="Include legacy/deprecated candidates (requires --experimental)",
    )
    parser.add_argument(
        "--experimental", action="store_true",
        help="Phase 0: Unlock full experimental registry (non-canonical candidates)",
    )
    parser.add_argument(
        "--require-model", action="store_true",
        help="Require model to load successfully; exit nonzero on failure",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict mode (legacy): equivalent to --strict-execution --strict-promotion",
    )
    parser.add_argument(
        "--strict-execution", action="store_true",
        help="P0 Fix: Fail if any candidate has execution errors (independent of promotion)",
    )
    parser.add_argument(
        "--strict-promotion", action="store_true",
        help="P0 Fix: Fail if promotion policy rejects (independent of execution)",
    )
    parser.add_argument(
        "--require-compressed-execution", action="store_true",
        help="Fix #5: Require packed blocks to be created and read, no fallback",
    )
    parser.add_argument(
        "--governance-only", action="store_true",
        help="Governance-only mode: test schema validation without MLX (for CI smoke testing)",
    )
    parser.add_argument(
        "--bit-width", type=str, default="k8v8",
        choices=["k16v16", "k8v16", "k16v8", "k8v8", "k8v6", "k8v5"],
        help="Bit-width configuration for bit-width isolation ladder (default: k8v8)",
    )
    args = parser.parse_args()

    # Governance-only mode: test schema validation without MLX
    if args.governance_only:
        print("Governance-only mode: testing schema validation without MLX")
        try:
            # Test CandidateResult schema
            from benchmarks.schemas import CandidateResult
            test_result = CandidateResult(
                candidate_name="test",
                model_id="test_model",
                prompt_id="test_prompt",
                context_length=128,
                output_tokens=32,
            )
            # Test JSON serialization
            json_str = test_result.to_json()
            loaded = CandidateResult.from_dict(json.loads(json_str))
            assert loaded.candidate_name == "test"

            # Test token sequence hash
            from rfsn_v11.candidates.logit_capture import compute_token_sequence_hash
            test_hash = compute_token_sequence_hash(
                model_id="test",
                prompt_id="test",
                prompt_text="Hello",
                target_token_ids=[1, 2, 3],
                max_tokens=10,
                temperature=0.0,
            )
            assert len(test_hash) == 64  # SHA256 hex string

            # Test judge logic
            from benchmarks.judge import Judge
            judge = Judge()
            baseline = CandidateResult(
                candidate_name="baseline",
                model_id="test",
                prompt_id="test",
                context_length=128,
                output_tokens=32,
                logit_cosine=1.0,
                top5_overlap=1.0,
                attention_score_cosine=1.0,
                attention_top5_overlap=1.0,
                perplexity_delta=0.0,
                visible_output_drift_score=0.0,
                peak_memory_mb=1000.0,
                kv_cache_memory_mb=64.0,
                compressed_kv_memory_mb=64.0,
                decode_tps=100.0,
                # Add required promotion fields
                commit_hash="test_commit",
                corpus_hash="test_corpus",
                token_sequence_hash="test_hash",
            )
            # Just verify judge runs without error
            verdict = judge.evaluate(baseline, baseline)
            print(f"  Judge evaluation successful: {verdict.label.value}")
            # Don't assert specific verdict since baseline vs baseline may be rejected

            print("✓ Governance-only tests passed")
            print("  - CandidateResult schema validation")
            print("  - JSON serialization round-trip")
            print("  - Token sequence hash computation")
            print("  - Judge logic evaluation")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Governance-only tests failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Determine mode and output directory
    if args.promotion_report:
        mode = "promotion"
        out_dir = ARTIFACTS_ROOT / "promotion"
    elif args.memory_report:
        mode = "memory"
        out_dir = ARTIFACTS_ROOT / "memory"
    elif args.full_logit_gate:
        mode = "full_logit"
        out_dir = ARTIFACTS_ROOT / "full_logit"
    elif args.quick:
        mode = "quick"
        out_dir = ARTIFACTS_ROOT / "quick"
    else:
        mode = "quick"
        out_dir = ARTIFACTS_ROOT / "quick"
        print("No mode specified; defaulting to --quick")

    print(f"KV Shootout — mode={mode}")
    print(f"Outputs: {out_dir}")

    models = (
        [args.model] if args.model
        else (MODELS_QUICK if args.quick else MODELS_FULL)
    )
    prompts = PROMPTS_QUICK if args.quick else PROMPTS_FULL
    max_tokens = MAX_TOKENS_QUICK if args.quick else MAX_TOKENS_FULL

    all_rows: list[dict[str, Any]] = []
    models_tested: list[str] = []
    any_model_loaded = False
    last_tokenizer: Any = None
    last_baseline_result: CandidateResult | None = None
    last_model_id: str = ""

    # Track run validity for strict mode
    models_requested = len(models)
    models_loaded = 0
    candidates_requested = 0
    candidates_completed = 0
    baseline_completed = False
    run_errors = []
    token_sequence_reference = None

    # P0 Fix: Define strict mode flags early for use throughout function
    strict_execution = args.strict or args.strict_execution
    strict_promotion = args.strict or args.strict_promotion

    for model_id in models:
        model, tokenizer = _load_model(model_id)
        if model is None:
            if args.require_model:
                print(f"\nERROR: Required model {model_id} failed to load.")
                sys.exit(1)
            continue
        any_model_loaded = True
        models_loaded += 1
        models_tested.append(model_id)
        last_tokenizer = tokenizer
        last_model_id = model_id

        candidates = _build_candidates(
            quick=args.quick,
            include_legacy=args.include_legacy,
            bit_width_config=args.bit_width,
            canonical=args.canonical,
            experimental=args.experimental,
        )
        candidates_requested += len(candidates)
        if not candidates:
            print("  No candidates available.")
            if strict_execution:
                run_errors.append(f"No candidates available for model {model_id}")
            continue

        per_candidate_results: dict[str, list[CandidateResult]] = {}

        for prompt in prompts:
            print(f"\n  Prompt: {prompt[:60]}...")
            baseline_result: CandidateResult | None = None

            for candidate in candidates:
                print(
                    f"    Running {candidate.name} ...",
                    end=" ",
                    flush=True,
                )
                result = _run_once(
                    candidate, model, tokenizer, prompt, max_tokens,
                    baseline_result=baseline_result, mode=mode,
                    require_compressed_execution=args.require_compressed_execution,
                )
                per_candidate_results.setdefault(
                    candidate.name, []
                ).append(result)

                # Track candidate completion
                if result.gate_status != "ERROR":
                    candidates_completed += 1
                elif strict_execution:
                    run_errors.append(
                        f"Candidate {candidate.name} failed: {result.error}"
                    )

                if candidate.name == "dense_mlx_baseline":
                    baseline_result = result
                    last_baseline_result = result
                    if result.gate_status != "ERROR":
                        baseline_completed = True

                print(
                    f"{result.gate_status}  "
                    f"tps={result.tokens_per_sec or 'N/A'}"
                )

        for name, results in per_candidate_results.items():
            agg = _aggregate(results)
            all_rows.append(agg)

    if not any_model_loaded:
        print("\nNo model loaded — mlx_lm may not be installed.")
        all_rows = [{
            "status": "SKIPPED_NO_MLX_LM",
            "reason": (
                "mlx_lm is not installed; run on Apple Silicon "
                "with pip install -e '.[fusion]'"
            ),
        }]

    # Filter for promotion report
    if mode == "promotion":
        # Promotion eligibility requires full logit gate data.
        # If full_logit artifacts exist, validate them before use.
        full_logit_path = ARTIFACTS_ROOT / "full_logit" / "results.json"
        full_logit_valid = False
        validation_reason = "full_logit artifact not found"
        if full_logit_path.exists():
            try:
                with full_logit_path.open("r", encoding="utf-8") as fh:
                    full_logit_payload = json.load(fh)
                meta = full_logit_payload.get("metadata", {})
                if meta.get("benchmark_methodology") != _ARTIFACT_METHODLOGY:
                    validation_reason = (
                        "full_logit artifact methodology mismatch: "
                        f"{meta.get('benchmark_methodology')}"
                    )
                elif not meta.get("token_sequence_hash"):
                    validation_reason = (
                        "full_logit artifact token_sequence_hash is empty"
                    )
                elif meta.get("artifact_schema_version") != "2.0":
                    validation_reason = (
                        "full_logit artifact schema version mismatch: "
                        f"{meta.get('artifact_schema_version')}"
                    )
                else:
                    stale_phrases = [
                        "stale until regenerated",
                        "Promoted candidate:",
                        "promotion eligible under Alpha 8.3 rules",
                    ]
                    full_md_path = (
                        ARTIFACTS_ROOT / "full_logit" / "results.md"
                    )
                    md_text = (
                        full_md_path.read_text(encoding="utf-8")
                        if full_md_path.exists()
                        else ""
                    )
                    if any(p in md_text for p in stale_phrases):
                        validation_reason = (
                            "full_logit markdown contains stale promotion text"
                        )
                    else:
                        full_logit_valid = True
            except Exception as exc:
                validation_reason = (
                    f"full_logit artifact validation error: {exc}"
                )

        if full_logit_valid:
            try:
                full_logit_results = full_logit_payload.get(
                    "results", full_logit_payload
                )
                # Show candidates whose quality gate passed (PASS or
                # PASS_NO_PROMOTE), not only those with promotion_eligible.
                quality_passed = [
                    r for r in full_logit_results
                    if r.get("gate_status") in (GATE_STATUS_PASS, "PASS_NO_PROMOTE")
                ]
                if quality_passed:
                    print(
                        f"\nUsing full_logit artifacts: "
                        f"{len(quality_passed)} candidate(s) passed quality gate."
                    )
                    all_rows = quality_passed
                else:
                    print("\nNo candidate passed quality gate.")
                    all_rows = [
                        {"note": "No candidate passed quality gate."}
                    ]
            except Exception:
                # Fallback: scan the already-loaded full_logit_payload
                if isinstance(full_logit_payload, dict):
                    fallback_results = full_logit_payload.get("results", [])
                elif isinstance(full_logit_payload, list):
                    fallback_results = full_logit_payload
                else:
                    fallback_results = []
                _pass_statuses = (GATE_STATUS_PASS, "PASS_NO_PROMOTE")
                quality_passed = [
                    r for r in fallback_results
                    if r.get("gate_status") in _pass_statuses
                ]
                if quality_passed:
                    _n = len(quality_passed)
                    print(
                        f"\nUsing full_logit artifacts: "
                        f"{_n} candidate(s) passed quality gate."
                    )
                    all_rows = quality_passed
                else:
                    print("\nNo candidate passed quality gate.")
                    all_rows = [
                        {"note": "No candidate passed quality gate."}
                    ]
        else:
            print(
                f"\nNo candidate is promotion eligible. "
                f"Reason: {validation_reason}"
            )
            all_rows = [
                {
                    "note": (
                        "No candidate is promotion eligible. "
                        f"Reason: {validation_reason}"
                    )
                }
            ]

    # Compute token-sequence hash from the last baseline result if
    # available.  Requires MLX + tokenizer; on CPU-only sandboxes the
    # hash remains empty and promotion is blocked.
    token_sequence_hash = ""
    if mode == "full_logit" and last_baseline_result is not None:
        try:
            baseline_text = last_baseline_result.generated_text
            if baseline_text and last_tokenizer is not None:
                target_ids = last_tokenizer.encode(baseline_text)
                token_sequence_hash = compute_token_sequence_hash(
                    model_id=last_model_id,
                    prompt_id="prompt_0",
                    prompt_text=prompts[0] if prompts else "",
                    target_token_ids=target_ids,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    decode_mode="greedy",
                    methodology=_ARTIFACT_METHODLOGY,
                    tokenizer_id=getattr(
                        last_tokenizer, "name_or_path", None
                    ),
                )
        except Exception:
            pass
    elif mode in ("memory", "quick", "promotion"):
        # Non-full-logit modes should reference the full_logit artifact
        # rather than copying the token hash directly. This preserves
        # provenance by linking to the source artifact.
        full_logit_json = ARTIFACTS_ROOT / "full_logit" / "results.json"
        if full_logit_json.exists():
            try:
                with full_logit_json.open("r", encoding="utf-8") as fh:
                    full_payload = json.load(fh)
                # Reference the source artifact instead of copying hash
                token_sequence_artifact = "full_logit/results.json"
                token_sequence_hash = (
                    full_payload.get("metadata", {})
                    .get("token_sequence_hash", "")
                )
                # Store as reference, not direct copy
                token_sequence_reference = {
                    "token_sequence_hash": token_sequence_hash,
                    "token_sequence_artifact": token_sequence_artifact,
                    "token_sequence_artifact_sha256": _compute_file_sha256(full_logit_json),
                }
            except Exception:
                token_sequence_reference = None
        else:
            token_sequence_reference = None

    # P0 Fix: Strict execution validation (separate from strict promotion)
    if strict_execution:
        run_valid = True
        validation_errors = []

        if models_loaded == 0:
            run_valid = False
            validation_errors.append("No models loaded")

        if models_loaded < models_requested:
            run_valid = False
            validation_errors.append(
                f"Only {models_loaded}/{models_requested} models loaded"
            )

        if candidates_completed == 0:
            run_valid = False
            validation_errors.append("No candidates completed")

        if candidates_completed < candidates_requested:
            run_valid = False
            validation_errors.append(
                f"Only {candidates_completed}/{candidates_requested} candidates completed"
            )

        if not baseline_completed:
            run_valid = False
            validation_errors.append("Baseline did not complete")

        if run_errors:
            run_valid = False
            validation_errors.extend(run_errors)

        # Add run validity metadata
        run_metadata = {
            "models_requested": models_requested,
            "models_loaded": models_loaded,
            "candidates_requested": candidates_requested,
            "candidates_completed": candidates_completed,
            "baseline_completed": baseline_completed,
            "run_valid": run_valid,
            "strict_execution": strict_execution,
            "strict_promotion": strict_promotion,
        }

        if validation_errors:
            run_metadata["validation_errors"] = validation_errors

        # Inject into first row or create a metadata row
        if all_rows and isinstance(all_rows[0], dict):
            if "metadata" not in all_rows[0]:
                all_rows[0]["metadata"] = {}
            all_rows[0]["metadata"].update(run_metadata)

        if not run_valid:
            print("\nSTRICT EXECUTION VALIDATION FAILED:")
            for error in validation_errors:
                print(f"  - {error}")
            sys.exit(1)

    # Use evidence-based promotion policy instead of hardcoded boolean
    # Load release configuration for policy context
    release_config_path = Path("release.toml")
    if release_config_path.exists():
        with release_config_path.open("rb") as f:
            release_config_data = tomllib.load(f)
    else:
        release_config_data = {}

    # Build run bundle for policy evaluation
    run_bundle = {
        "metadata": {
            "token_sequence_hash": token_sequence_hash,
            "token_sequence_provenance": token_sequence_reference,
            "models_tested": models_tested,
            "release_id": release_config_data.get("release_id", "unknown"),
        },
        "results": all_rows,
    }

    # Evaluate promotion eligibility using policy
    promotion_allowed, promotion_blockers = evaluate_promotion_policy(
        run_bundle,
        policy_config={"current_release_id": release_config_data.get("release_id", "alpha-8.4")}
    )

    if not promotion_allowed and promotion_blockers:
        print("\nPromotion blocked by policy:")
        for blocker in promotion_blockers:
            print(f"  - {blocker}")
    if promotion_allowed:
        methodology_status = _METHODOLOGY_STATUS_RERUN_PROMO
    elif not token_sequence_hash:
        methodology_status = _METHODOLOGY_STATUS_RERUN_INCOMPLETE_NO_PROMO
    else:
        methodology_status = _METHODOLOGY_STATUS_RERUN_NO_PROMO

    # Global promotion lock: if promotion is globally disabled, force every
    # row to be non-promotable regardless of individual gate results.
    if not promotion_allowed:
        for row in all_rows:
            if not isinstance(row, dict) or row.get("status", "").startswith("SKIPPED"):
                continue
            if "note" in row:
                continue
            if row.get("gate_status") == GATE_STATUS_PASS:
                row["gate_status"] = "PASS_NO_PROMOTE"
                row["promotion_blocked_reason"] = (
                    "global promotion lock active; "
                    "token provenance and runtime-instrumented traces incomplete"
                )
            row["promotion_eligible"] = False

    _write_artifacts(
        all_rows,
        out_dir,
        mode=mode,
        token_sequence_hash=token_sequence_hash,
        token_sequence_reference=token_sequence_reference if mode != "full_logit" else None,
        promotion_allowed=promotion_allowed,
    )
    _export_winner(
        all_rows, models_tested,
        methodology_status=methodology_status,
        promotion_allowed=promotion_allowed,
        mode=mode,
    )

    # P0 Fix: Final strict execution checks (variables defined at function start)
    if strict_execution:
        # Check for any failed quality gates
        failed_gates = []
        for row in all_rows:
            if isinstance(row, dict) and row.get("gate_status") == GATE_STATUS_FAIL:
                candidate_name = row.get("candidate_name", "unknown")
                failed_gates.append(candidate_name)

        if failed_gates:
            print(f"\nSTRICT EXECUTION: Quality gates failed for {len(failed_gates)} candidate(s):")
            for candidate in failed_gates:
                print(f"  - {candidate}")
            sys.exit(1)

        # Check for any execution errors
        execution_errors = []
        for row in all_rows:
            if isinstance(row, dict) and row.get("error"):
                candidate_name = row.get("candidate_name", "unknown")
                execution_errors.append(candidate_name)

        if execution_errors:
            print(f"\nSTRICT EXECUTION: Execution errors in {len(execution_errors)} candidate(s):")
            for candidate in execution_errors:
                print(f"  - {candidate}")
            sys.exit(1)

    # P0 Fix: Strict promotion is separate from strict execution
    # Quick mode can pass strict-execution but will fail strict-promotion
    if strict_promotion:
        if not promotion_allowed and promotion_blockers:
            print("\nSTRICT PROMOTION: Promotion policy failed:")
            for blocker in promotion_blockers:
                print(f"  - {blocker}")
            sys.exit(2)
        elif not promotion_allowed:
            print("\nSTRICT PROMOTION: Promotion not allowed (no blockers listed)")
            sys.exit(2)

    print("\nDone.")


if __name__ == "__main__":
    main()
