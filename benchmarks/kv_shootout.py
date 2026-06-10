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
If no candidate beats mlx_lm_baseline in quality, the baseline wins.

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
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Suppress mlx-lm legacy sampling-arg deprecation warnings emitted during
# generation. These are known and do not affect results.
warnings.filterwarnings("ignore", message="Specifying sampling arguments", category=UserWarning)
warnings.filterwarnings("ignore", message="Specifying ``repetition_penalty``", category=UserWarning)

# Add repo root to path so rfsn_v11 is importable without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfsn_v11.candidates.base import CandidateResult, KVCompressionCandidate
from rfsn_v11.candidates.quality_gates import (
    LOGIT_COSINE_MIN,
    KL_DIVERGENCE_MAX,
    TOP5_OVERLAP_MIN,
    TOP10_OVERLAP_MIN,
    MAX_LOGIT_DELTA_MAX,
    evaluate_quality_gate,
    compute_promotion_eligibility,
    GATE_STATUS_PASS,
    GATE_STATUS_FAIL,
    GATE_STATUS_PENDING_LOGIT_GATE,
    GATE_STATUS_PENDING_MEMORY_METRICS,
)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

MODELS_FULL = [
    "Qwen/Qwen2.5-1.5B-Instruct",  # primary: head_dim=128, ideal for TQ rotation
]
MODELS_QUICK = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # quick iteration: head_dim=64, fast load
]

PROMPTS_QUICK = [
    "Hello",
    "Write a Python function that adds two numbers.",
]

MAX_TOKENS_FULL = 200
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
        "User: What is the capital of France?\nAssistant: Paris.\nUser: And what language do they speak there?",
    ],
}

# Flat list for non-quick full runs (one prompt per category)
PROMPTS_FULL = [prompts[0] for prompts in PROMPT_SUITE.values()]

# Temperature=0.0 for all candidates to make text comparable across methods.
# Without greedy decoding, stochastic sampling causes false text-heuristic FAILs.
GENERATION_TEMP = 0.0

ARTIFACTS_ROOT = Path("artifacts/bench/shootout")


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------

def _build_candidates(quick: bool = False) -> list[KVCompressionCandidate]:
    """Instantiate all candidates. Skip unavailable ones gracefully."""
    from rfsn_v11.candidates.mlx_lm_baseline import MLXLMBaseline
    from rfsn_v11.candidates.mlx_lm_quantized import MLXLMQuantizedKV
    from rfsn_v11.candidates.rfsn_v10_adapter import RFSNV10Candidate
    from rfsn_v11.candidates.rfsn_v11_adapter import RFSNV11Candidate
    from rfsn_v11.candidates.turboquant_v2_adapter import TurboQuantV2Candidate
    from rfsn_v11.candidates.polar_reference_adapter import PolarReferenceAdapter

    all_candidates: list[KVCompressionCandidate] = [
        MLXLMBaseline(),
        MLXLMQuantizedKV(kv_bits=8),
        RFSNV10Candidate("k8_v5_gs32"),
        RFSNV10Candidate("k8_v5_gs64"),
        RFSNV11Candidate(key_bits=8, value_bits=4, group_size=64, use_wht=True, dim=128),
        TurboQuantV2Candidate(bits=4, group_size=64),
        PolarReferenceAdapter(bits=4, dim=128),
    ]

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
) -> CandidateResult:
    """Run one candidate on one prompt and apply quality gate."""
    _reset_peak_memory()
    result = candidate.run(model, tokenizer, prompt, max_tokens, temp=temp)
    peak_mb = _peak_memory_mb()
    if peak_mb is not None:
        result.working_set_memory_mb = peak_mb

    # Error gate
    if result.error:
        result.gate_status = "ERROR"
        result.promotion_eligible = False
        return result

    # Preserve adapter-specific pending statuses (more specific than generic logic)
    if result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION:
        # Real cache injection is the blocker; do not overwrite with generic logit-pending
        result.promotion_eligible = False
        return result

    # Baseline always passes logit gate by definition
    if candidate.name == "mlx_lm_baseline":
        result.logit_cosine = 1.0
        result.kl_divergence = 0.0
        result.top1_match = 1.0
        result.top5_overlap = 1.0
        result.top10_overlap = 1.0
        result.max_logit_delta = 0.0
        result.first_divergent_token = None
        result.logit_gate_passed = True
        result.memory_gate_passed = True
        result.gate_status = GATE_STATUS_PASS
        result.promotion_eligible = True
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

    # full-logit-gate mode: require real logits
    # For now, most candidates do not capture logits during generation.
    # Mark them as PENDING_LOGIT_GATE honestly.
    if result.logit_cosine is None:
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
            result.notes += "  [logit gate failed: " + "; ".join(gate.failure_reasons) + "]"
        else:
            # Check memory gate for promotion eligibility
            result.memory_gate_passed = (
                result.actual_kv_memory_mb is not None
                and result.working_set_memory_mb is not None
                and result.size_ratio is not None
                and result.compression_factor is not None
            )
            promotion_eligible, gate_status = compute_promotion_eligibility(
                logit_gate_passed=result.logit_gate_passed,
                memory_gate_passed=result.memory_gate_passed,
                actual_kv_memory_mb=result.actual_kv_memory_mb,
                working_set_memory_mb=result.working_set_memory_mb,
                size_ratio=result.size_ratio,
                compression_factor=result.compression_factor,
            )
            result.promotion_eligible = promotion_eligible
            result.gate_status = gate_status

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
        result.notes += "  [text heuristic: exact match — logit gate still pending]"
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
    result.notes += f"  [text heuristic: diverged at token {first_diff} — logit gate pending]"
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

    # Average numeric fields
    numeric = [
        "total_ms",
        "tokens_per_sec",
        "size_ratio",
        "compression_factor",
        "logit_cosine",
        "kl_divergence",
        "top5_overlap",
        "top10_overlap",
        "max_logit_delta",
    ]
    agg: dict[str, Any] = {"name": name, "model_id": model_id}
    for field in numeric:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        agg[field] = float(np.mean(vals)) if vals else None

    # Gate status: if any FAIL, overall FAIL; if all PASS, PASS; otherwise pending
    statuses = [r.gate_status for r in results]
    if any(s == GATE_STATUS_FAIL for s in statuses):
        agg["gate_status"] = GATE_STATUS_FAIL
    elif all(s == GATE_STATUS_PASS for s in statuses):
        agg["gate_status"] = GATE_STATUS_PASS
    else:
        # Use the most common pending status
        pending = [s for s in statuses if s != GATE_STATUS_PASS]
        agg["gate_status"] = pending[0] if pending else GATE_STATUS_PASS

    agg["promotion_eligible"] = all(r.promotion_eligible for r in results)
    agg["count"] = len(results)
    agg["notes"] = " | ".join({r.notes for r in results if r.notes})
    return agg


def _write_artifacts(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Write JSON, CSV, and Markdown artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)

    # CSV
    if rows:
        csv_path = out_dir / "results.csv"
        headers = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    # Markdown
    md_path = out_dir / "results.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# KV Shootout Results\n\n")
        if not rows:
            fh.write("No results.\n")
            return

        # Header
        headers = list(rows[0].keys())
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            cells = []
            for h in headers:
                v = row.get(h, "")
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            fh.write("| " + " | ".join(cells) + " |\n")

    print(f"  Wrote {json_path}, {csv_path if rows else ''}, {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KV-cache compression shootout")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run")
    parser.add_argument("--full-logit-gate", action="store_true", help="Run real logit comparison")
    parser.add_argument("--memory-report", action="store_true", help="Require all candidates to report memory metrics")
    parser.add_argument("--promotion-report", action="store_true", help="Only rank promotion-eligible candidates")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID")
    args = parser.parse_args()

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

    models = [args.model] if args.model else (MODELS_QUICK if args.quick else MODELS_FULL)
    prompts = PROMPTS_QUICK if args.quick else PROMPTS_FULL
    max_tokens = MAX_TOKENS_QUICK if args.quick else MAX_TOKENS_FULL

    all_rows: list[dict[str, Any]] = []

    for model_id in models:
        model, tokenizer = _load_model(model_id)
        if model is None:
            continue

        candidates = _build_candidates(quick=args.quick)
        if not candidates:
            print("  No candidates available.")
            continue

        per_candidate_results: dict[str, list[CandidateResult]] = {}

        for prompt in prompts:
            print(f"\n  Prompt: {prompt[:60]}...")
            baseline_result: CandidateResult | None = None

            for candidate in candidates:
                print(f"    Running {candidate.name} ...", end=" ", flush=True)
                result = _run_once(
                    candidate, model, tokenizer, prompt, max_tokens,
                    baseline_result=baseline_result, mode=mode,
                )
                per_candidate_results.setdefault(candidate.name, []).append(result)

                if candidate.name == "mlx_lm_baseline":
                    baseline_result = result

                print(f"{result.gate_status}  tps={result.tokens_per_sec or 'N/A'}")

        for name, results in per_candidate_results.items():
            agg = _aggregate(results)
            all_rows.append(agg)

    # Filter for promotion report
    if mode == "promotion":
        eligible = [r for r in all_rows if r.get("promotion_eligible")]
        if not eligible:
            print("\nNo candidate is promotion eligible.")
            all_rows = [{"note": "No candidate is promotion eligible."}]
        else:
            all_rows = eligible

    _write_artifacts(all_rows, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
