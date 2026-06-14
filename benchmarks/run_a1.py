#!/usr/bin/env python3
"""Run the full A1 benchmark: dense baseline vs A1_wht_grouped_k8v4_gs64.

Steps 11 and 12: Full A1 CLI runner with judge-based PROMOTE/KEEP_EXPERIMENTAL/REJECT verdict.

Usage
-----
# Smoke mode (no model, synthetic data, validates harness)
    python benchmarks/run_a1.py --smoke

# Real model, quick (0.5B, 2 prompts)
    python benchmarks/run_a1.py --quick

# Full run (0.5B + 1.5B, all prompts)
    python benchmarks/run_a1.py

# Specific model
    python benchmarks/run_a1.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Override output directory
    python benchmarks/run_a1.py --smoke --out-dir /tmp/rfsn_a1

Outputs
-------
    benchmarks/results/a1_latest.json
    benchmarks/reports/a1_latest.md

Exit codes
----------
    0  PROMOTE, KEEP_EXPERIMENTAL, or SMOKE_PASS
    1  REJECT, REGRESSION, or SMOKE_FAIL
    2  Error / missing dependency
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.baseline_mlx import (
    MODELS_FULL,
    MODELS_QUICK,
    PROMPT_SUITE,
    PROMPT_SUITE_QUICK,
    _make_smoke_result,
)
from benchmarks.baseline_mlx import run_single as run_baseline
from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_Grouped
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.judge import Judge, VerdictLabel
from benchmarks.report_generator import ReportGenerator
from benchmarks.schemas import CandidateResult

# ---------------------------------------------------------------------------
# Smoke A1 result
# ---------------------------------------------------------------------------

def _make_smoke_a1_result(
    baseline: CandidateResult,
    rng: np.random.Generator,
) -> CandidateResult:
    """Build a synthetic A1 result that is quality-safe and memory-improved."""
    kv_dense = baseline.kv_cache_memory_mb or 64.0
    compressed = kv_dense * 0.44  # ~56% of FP16 → 44% reduction

    return CandidateResult(
        candidate_name="A1_wht_grouped_k8v4_gs64",
        model_id=baseline.model_id,
        prompt_id=baseline.prompt_id,
        context_length=baseline.context_length,
        output_tokens=baseline.output_tokens,
        preconditioner="wht",
        quantizer="grouped_sym",
        key_bits=8.0,
        value_bits=4.0,
        group_size=64,
        is_benchmark_only=True,
        run_type="smoke",
        promotion_eligible=False,
        requested_backend="unknown",
        executed_backend="unknown",
        logit_cosine=float(rng.uniform(0.997, 0.9999)),
        top1_match_rate=float(rng.uniform(0.96, 0.999)),
        top5_overlap=float(rng.uniform(0.97, 0.999)),
        top10_overlap=float(rng.uniform(0.98, 0.999)),
        perplexity_delta=float(rng.uniform(0.0, 0.01)),
        visible_output_drift_score=float(rng.uniform(0.0, 0.02)),
        attention_score_cosine=float(rng.uniform(0.997, 0.9999)),
        attention_score_mae=float(rng.uniform(0.0001, 0.002)),
        attention_top5_overlap=float(rng.uniform(0.97, 0.999)),
        softmax_kl=float(rng.uniform(0.0001, 0.005)),
        peak_memory_mb=float(baseline.peak_memory_mb * rng.uniform(0.70, 0.90)),
        kv_cache_memory_mb=kv_dense,
        compressed_kv_memory_mb=compressed,
        metadata_memory_mb=0.5,
        effective_bits_per_kv_element=6.0,
        compression_factor=kv_dense / max(compressed, 1e-9),
        prefill_tps=baseline.prefill_tps,
        decode_tps=float(baseline.decode_tps * rng.uniform(0.95, 1.15)),
        first_token_latency_ms=float(baseline.first_token_latency_ms * rng.uniform(1.0, 1.05)),
        total_latency_ms=float(baseline.total_latency_ms * rng.uniform(1.0, 1.1)),
        compression_time_ms=float(rng.uniform(1.0, 5.0)),
        decompression_time_ms=float(rng.uniform(2.0, 8.0)),
        generated_text=f"[smoke] A1 generated text for {baseline.prompt_id!r}.",
        notes="synthetic smoke data — A1 quality-safe",
        source_type="installed_wheel",
        commit_hash="smoke",
        corpus_hash="smoke",
        token_sequence_hash="smoke",
        measured_memory=True,
        proof_counters={
            "requantized_tokens": 0,
            "fallback_attention_calls": 0,
            "dense_shadow_bytes": 0,
            "unknown_layer_events": 0,
        },
    )


# ---------------------------------------------------------------------------
# Quality comparison: fill A1 result with baseline-relative metrics
# ---------------------------------------------------------------------------

def _compare_with_baseline(
    a1_result: CandidateResult,
    baseline: CandidateResult,
) -> None:
    """Fill quality metrics on a1_result by comparing generated logits/text.

    When real logits are available (stored in _logits/_baseline_logits),
    compute real metrics.  Otherwise fall back to text-level drift score.
    """
    # If both have raw logits, compute full quality suite
    if (
        a1_result._logits is not None
        and baseline._logits is not None
    ):
        metrics = BenchmarkCandidate.compute_logit_quality(
            np.array(baseline._logits),
            np.array(a1_result._logits),
        )
        a1_result.logit_cosine = metrics["logit_cosine"]
        a1_result.top1_match_rate = metrics["top1_match_rate"]
        a1_result.top5_overlap = metrics["top5_overlap"]
        a1_result.top10_overlap = metrics["top10_overlap"]
        a1_result.perplexity_delta = metrics["perplexity_delta"]
        a1_result.visible_output_drift_score = metrics["visible_output_drift_score"]
        return

    # Fallback: text-level drift
    if a1_result.generated_text and baseline.generated_text:
        b_words = baseline.generated_text.split()
        c_words = a1_result.generated_text.split()
        n = min(len(b_words), len(c_words), 50)
        if n > 0:
            match = sum(bw == cw for bw, cw in zip(b_words[:n], c_words[:n])) / n
            a1_result.visible_output_drift_score = 1.0 - match
            a1_result.top1_match_rate = match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="RFSN A1 benchmark runner")
    parser.add_argument("--model", default=None)
    parser.add_argument("--quick", action="store_true", help="Small model + short prompts")
    parser.add_argument("--smoke", action="store_true", help="Synthetic mode, no model download")
    parser.add_argument("--output-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    here = Path(__file__).parent
    results_dir = Path(args.out_dir) if args.out_dir else here / "results"
    reports_dir = here / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator(out_dir=results_dir, report_dir=reports_dir)
    judge = Judge(strict=True)
    rng = np.random.default_rng(args.seed)
    prompts = PROMPT_SUITE_QUICK if args.quick else PROMPT_SUITE

    all_baselines: list[CandidateResult] = []
    all_a1: list[CandidateResult] = []
    all_verdicts = []

    # ------------------------------------------------------------------
    # Smoke mode
    # ------------------------------------------------------------------
    if args.smoke:
        print("[smoke] Running synthetic A1 benchmark (no model download)\n")
        model_id = args.model or "smoke/Qwen2.5-0.5B"

        for prompt_id, prompt in prompts.items():
            baseline = _make_smoke_result(model_id, prompt_id, prompt, args.output_tokens, rng)
            a1 = _make_smoke_a1_result(baseline, rng)
            verdict = judge.evaluate(a1, baseline)
            all_baselines.append(baseline)
            all_a1.append(a1)
            all_verdicts.append(verdict)

            print(
                f"  {prompt_id:<28} "
                f"kv_ratio={a1.compressed_kv_memory_mb / max(a1.kv_cache_memory_mb, 1e-9):.2f}  "
                f"cosine={a1.logit_cosine:.4f}  "
                f"top5={a1.top5_overlap:.3f}  "
                f"attn_cos={a1.attention_score_cosine:.4f}  "
                f"→ {verdict.label.value}"
            )

        _print_summary(all_verdicts)
        if not all_baselines:
            print("\nWARNING: No baseline results collected. Skipping report generation.")
            return 1
        generator.write(
            candidates=all_a1,
            baseline=all_baselines[0],
            verdicts=all_verdicts,
            run_tag="a1",
            metadata={"smoke": True, "model_id": model_id},
        )
        print(f"\nResults: {results_dir}/a1_latest.json")
        print(f"Report:  {reports_dir}/a1_latest.md")

        worst = max(all_verdicts, key=lambda v: {
            VerdictLabel.PROMOTE: 0,
            VerdictLabel.KEEP_EXPERIMENTAL: 1,
            VerdictLabel.SMOKE_PASS: 1,
            VerdictLabel.REGRESSION: 2,
            VerdictLabel.SMOKE_FAIL: 3,
            VerdictLabel.REJECT: 3,
        }.get(v.label, 1))
        return 1 if worst.label in (
            VerdictLabel.REJECT, VerdictLabel.REGRESSION, VerdictLabel.SMOKE_FAIL
        ) else 0

    # ------------------------------------------------------------------
    # Real model run
    # ------------------------------------------------------------------
    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx_lm not installed.  Use --smoke or: pip install 'mlx-lm>=0.19'")
        return 2

    a1_candidate = A1_WHT_Grouped()
    if not a1_candidate.is_available():
        print("ERROR: A1 candidate dependencies not available.")
        return 2

    models = [args.model] if args.model else (MODELS_QUICK if args.quick else MODELS_FULL)

    for model_id in models:
        print(f"\nLoading {model_id} ...")
        try:
            model, tokenizer = mlx_lm.load(model_id)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        print(f"  model loaded.  Running {len(prompts)} prompts × 2 (baseline + A1)\n")

        for prompt_id, prompt in prompts.items():
            print(f"  [{prompt_id}]")

            # --- Baseline ---
            print("    baseline ...", end=" ", flush=True)
            t0 = time.perf_counter()
            baseline = run_baseline(
                model, tokenizer, model_id, prompt_id, prompt,
                output_tokens=args.output_tokens, seed=args.seed,
            )
            print(
                f"decode_tps={baseline.decode_tps:.1f}  "
                f"peak={baseline.peak_memory_mb:.0f}MB  "
                f"kv={baseline.kv_cache_memory_mb:.1f}MB  "
                f"[{(time.perf_counter()-t0)*1000:.0f}ms]"
            )

            # --- A1 ---
            print("    A1 ...", end=" ", flush=True)
            t0 = time.perf_counter()
            a1 = a1_candidate.run_on_model(
                model, tokenizer, model_id, prompt_id, prompt,
                output_tokens=args.output_tokens, seed=args.seed,
            )
            if a1.error:
                print(f"ERROR: {a1.error}")
                a1.visible_output_drift_score = 1.0
            else:
                _compare_with_baseline(a1, baseline)
                elapsed = (time.perf_counter() - t0) * 1000.0
                kv_ratio = (a1.compressed_kv_memory_mb or 0) / max(a1.kv_cache_memory_mb or 1, 1e-9)
                print(
                    f"decode_tps={a1.decode_tps:.1f}  "
                    f"kv_ratio={kv_ratio:.2f}  "
                    f"comp_ms={a1.compression_time_ms:.1f}  "
                    f"[{elapsed:.0f}ms]"
                )

            # Judge
            verdict = judge.evaluate(a1, baseline)
            print(f"    verdict: {verdict.label.value}  ({verdict.reason[:80]})")

            all_baselines.append(baseline)
            all_a1.append(a1)
            all_verdicts.append(verdict)

    if not all_a1:
        print("\nNo results produced.")
        return 2

    _print_summary(all_verdicts)

    # Use first baseline as the report reference
    if not all_baselines:
        print("\nWARNING: No baseline results collected. Skipping report generation.")
        return 1
    generator.write(
        candidates=all_a1,
        baseline=all_baselines[0],
        verdicts=all_verdicts,
        run_tag="a1",
        metadata={"models": list({r.model_id for r in all_baselines})},
    )
    print(f"\nResults: {results_dir}/a1_latest.json")
    print(f"Report:  {reports_dir}/a1_latest.md")

    worst = max(all_verdicts, key=lambda v: {
        VerdictLabel.PROMOTE: 0,
        VerdictLabel.KEEP_EXPERIMENTAL: 1,
        VerdictLabel.SMOKE_PASS: 1,
        VerdictLabel.REGRESSION: 2,
        VerdictLabel.SMOKE_FAIL: 3,
        VerdictLabel.REJECT: 3,
    }.get(v.label, 1))
    return 1 if worst.label in (
        VerdictLabel.REJECT, VerdictLabel.REGRESSION, VerdictLabel.SMOKE_FAIL
    ) else 0


def _print_summary(verdicts: list) -> None:
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.label.value] = counts.get(v.label.value, 0) + 1

    print("\n" + "=" * 60)
    print("A1 VERDICT SUMMARY")
    print("=" * 60)
    for label, count in sorted(counts.items()):
        print(f"  {label:<22} {count}")

    promoted = [v.candidate_name for v in verdicts if v.label == VerdictLabel.PROMOTE]
    if promoted:
        print(f"\n  PROMOTED: {', '.join(set(promoted))}")
    rejected = [v.candidate_name for v in verdicts if v.label == VerdictLabel.REJECT]
    if rejected:
        print(f"  REJECTED: {', '.join(set(rejected))}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
