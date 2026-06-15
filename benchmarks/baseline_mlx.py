#!/usr/bin/env python3
"""Dense MLX baseline runner — the ground truth for every compression candidate.

Usage
-----
# Real model run (downloads model on first use)
    python benchmarks/baseline_mlx.py

# Quick: small model only, short prompts
    python benchmarks/baseline_mlx.py --quick

# Specific model
    python benchmarks/baseline_mlx.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit

# Smoke mode: no model download, synthetic data, validates harness
    python benchmarks/baseline_mlx.py --smoke

# Output to custom directory
    python benchmarks/baseline_mlx.py --out-dir /tmp/rfsn_results

Outputs
-------
    benchmarks/results/baseline_mlx_latest.json
    benchmarks/reports/baseline_mlx_latest.md

Metric definitions
------------------
prefill_tps          tokens/sec during prompt encoding (from mlx_lm stream_generate)
decode_tps           tokens/sec during autoregressive decode
first_token_latency_ms  wall-clock ms until the first generated token is produced
total_latency_ms     wall-clock ms from generate() call to last token
peak_memory_mb       peak device memory reported by mlx_lm (converted from GB)
kv_cache_memory_mb   estimated FP16 KV size: 2 * layers * heads * head_dim * seq_len * 2 bytes
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfsn_v11.candidates.base import CandidateResult
from rfsn_v11.candidates.json_utils import dumps_json_strict  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed prompt set
# ---------------------------------------------------------------------------

PROMPT_SUITE: dict[str, str] = {
    "short_chat_512": (
        "You are a helpful assistant.\n\nUser: What is the capital of France? "
        "Please give a one-sentence answer.\nAssistant:"
    ),
    "coding_512": (
        "Write a Python function that takes a list of integers and returns the "
        "sum of all even numbers in the list. Include type hints and a docstring."
    ),
    "retrieval_2048": (
        "The following is a long passage about machine learning history. "
        "After reading it, answer: who invented the perceptron?\n\n"
        + ("Machine learning has a rich history dating back to the 1940s. " * 60)
        + "\n\nQuestion: Who invented the perceptron?\nAnswer:"
    ),
    "summarization_2048": (
        "Please summarize the following article in exactly two sentences:\n\n"
        + (
            "Artificial intelligence research has undergone several major paradigm shifts. "
            "The field began with symbolic AI in the 1950s, moved to expert systems in the "
            "1970s and 80s, then neural networks gained prominence in the 90s, followed by "
            "deep learning's rise in the 2010s. Each transition was driven by new theoretical "
            "insights, increased computational power, and larger datasets. "
        ) * 40
        + "\n\nSummary:"
    ),
    "needle_8192": (
        "The following document contains a hidden fact. Find it.\n\n"
        + ("This document discusses various topics including history, science, and art. " * 200)
        + "\n\n[HIDDEN FACT: The secret code is RFSN-42]\n\n"
        + ("More text follows about unrelated subjects. " * 100)
        + "\n\nQuestion: What is the secret code mentioned in the document?\nAnswer:"
    ),
    "multi_turn_4096": (
        "This is a multi-turn conversation.\n\n"
        "User: Tell me about the history of computing.\n"
        "Assistant: Computing history spans from mechanical calculators to modern CPUs.\n"
        "User: Who invented the transistor?\n"
        "Assistant: The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs in 1947.\n"
        "User: What impact did it have?\n"
        "Assistant: It enabled miniaturization and gave birth to modern electronics.\n"
        "User: Now explain how a CPU works in simple terms.\n"
        "Assistant:"
    ),
}

PROMPT_SUITE_QUICK: dict[str, str] = {
    "short_chat_512": PROMPT_SUITE["short_chat_512"],
    "coding_512": PROMPT_SUITE["coding_512"],
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS_QUICK = ["mlx-community/Qwen2.5-0.5B-Instruct-4bit"]
MODELS_FULL = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
]

# ---------------------------------------------------------------------------
# Smoke data — synthetic, no model required
# ---------------------------------------------------------------------------

def _make_smoke_result(
    model_id: str,
    prompt_id: str,
    prompt: str,
    output_tokens: int,
    rng: np.random.Generator,
) -> CandidateResult:
    """Build a synthetic CandidateResult that exercises the full schema."""
    context_length = len(prompt.split())
    prefill_tps = rng.uniform(800, 1200)
    decode_tps = rng.uniform(40, 80)
    total_latency_ms = (output_tokens / decode_tps) * 1000 + (context_length / prefill_tps) * 1000
    first_token_latency_ms = (context_length / prefill_tps) * 1000 + rng.uniform(5, 20)
    peak_memory_mb = rng.uniform(1500, 3000)

    # Synthetic KV memory estimate
    layers, heads, head_dim = 24, 8, 128
    kv_bytes = 2 * layers * heads * head_dim * context_length * 2  # FP16
    kv_cache_memory_mb = kv_bytes / (1024 ** 2)

    return CandidateResult(
        candidate_name="dense_mlx_baseline",
        model_id=model_id,
        prompt_id=prompt_id,
        context_length=context_length,
        output_tokens=output_tokens,
        preconditioner="none",
        quantizer="none",
        logit_cosine=1.0,
        top1_match_rate=1.0,
        top5_overlap=1.0,
        top10_overlap=1.0,
        perplexity_delta=0.0,
        visible_output_drift_score=0.0,
        attention_score_cosine=1.0,
        attention_score_mae=0.0,
        attention_top5_overlap=1.0,
        softmax_kl=0.0,
        peak_memory_mb=peak_memory_mb,
        kv_cache_memory_mb=kv_cache_memory_mb,
        compressed_kv_memory_mb=kv_cache_memory_mb,  # dense = no compression
        metadata_memory_mb=0.0,
        effective_bits_per_kv_element=16.0,
        compression_factor=1.0,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        first_token_latency_ms=first_token_latency_ms,
        total_latency_ms=total_latency_ms,
        compression_time_ms=0.0,
        decompression_time_ms=0.0,
        attention_time_ms=rng.uniform(0.5, 2.0),
        generated_text=f"[smoke] Generated {output_tokens} tokens for prompt {prompt_id!r}.",
        notes="synthetic smoke data — no real model used",
        source_type="synthetic",
        run_type="smoke",
        promotion_eligible=False,
        commit_hash="smoke",
        corpus_hash="smoke",
        token_sequence_hash="smoke",
        measured_memory=False,
        requested_backend="unknown",
        executed_backend="unknown",
    )


# ---------------------------------------------------------------------------
# Real model run helpers
# ---------------------------------------------------------------------------

def _estimate_kv_memory_mb(model: Any, context_length: int) -> float:
    """Estimate FP16 KV cache memory for a given context length."""
    try:
        args = getattr(model, "args", None)
        if args is None:
            return 0.0
        num_layers = getattr(args, "num_hidden_layers", None) or getattr(args, "n_layers", 0)
        num_kv_heads = (
            getattr(args, "num_key_value_heads", None)
            or getattr(args, "num_attention_heads", None)
            or 0
        )
        hidden_size = getattr(args, "hidden_size", None) or getattr(args, "dim", 0)
        num_heads = getattr(args, "num_attention_heads", None) or getattr(args, "n_heads", 1)
        head_dim = hidden_size // max(num_heads, 1)
        # K + V, FP16 = 2 bytes
        kv_bytes = 2 * num_layers * num_kv_heads * head_dim * context_length * 2
        return kv_bytes / (1024 ** 2)
    except Exception:
        return 0.0


def run_single(
    model: Any,
    tokenizer: Any,
    model_id: str,
    prompt_id: str,
    prompt: str,
    output_tokens: int = 100,
    seed: int = 42,
) -> CandidateResult:
    """Run a single dense MLX baseline generation and return a full CandidateResult."""
    try:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        # Tokenise to get real context length
        input_ids = tokenizer.encode(prompt)
        context_length = len(input_ids)

        first_token_time: float | None = None
        t_start = time.perf_counter()
        last_response = None
        generated_text_parts: list[str] = []
        all_logprobs: list[np.ndarray] = []

        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=output_tokens,
            sampler=sampler,
        ):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            last_response = response
            if response.text:
                generated_text_parts.append(response.text)
            # Collect per-token logprobs for quality metrics
            if hasattr(response, "logprobs") and response.logprobs is not None:
                try:
                    lp = np.array(response.logprobs).flatten()
                    all_logprobs.append(lp)
                except Exception:
                    pass

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        first_token_latency_ms = (
            (first_token_time - t_start) * 1000.0 if first_token_time is not None else None
        )

        generated_text = "".join(generated_text_parts)

        if last_response is None:
            raise RuntimeError("stream_generate yielded no responses")

        prefill_tps = float(getattr(last_response, "prompt_tps", 0.0))
        decode_tps = float(getattr(last_response, "generation_tps", 0.0))
        gen_tokens = int(getattr(last_response, "generation_tokens", 0))
        # peak_memory is in GB from mlx_lm
        peak_memory_gb = float(getattr(last_response, "peak_memory", 0.0))
        peak_memory_mb = peak_memory_gb * 1024.0

        kv_cache_memory_mb = _estimate_kv_memory_mb(model, context_length)

        return CandidateResult(
            name="dense_mlx_baseline",
            model_id=model_id,
            prompt_id=prompt_id,
            prompt=prompt,
            # Dense baseline has perfect quality by definition
            logit_cosine=1.0,
            kl_divergence=0.0,
            top1_match=1.0,
            top5_overlap=1.0,
            top10_overlap=1.0,
            max_logit_delta=0.0,
            first_divergent_token=None,
            # Memory
            actual_kv_memory_mb=kv_cache_memory_mb,
            working_set_memory_mb=peak_memory_mb,
            measurement_kind="ESTIMATED",
            # Compression
            size_ratio=1.0,
            compression_factor=1.0,
            # Timing
            prefill_ms=None,
            decode_ms=None,
            total_ms=total_latency_ms,
            # Throughput
            tokens_per_sec=decode_tps,
            # Packed fields (baseline has no compression)
            packed_blocks_created=0,
            packed_blocks_read=0,
            packed_attention_calls=0,
            dense_fallback_calls=0,
            full_history_materialization_calls=0,
            packed_bytes_written=0,
            packed_bytes_read=0,
            # Gate status
            gate_status="PASS_NO_PROMOTE",
            promotion_eligible=False,
            candidate_status="CONTROL",
            logit_gate_passed=True,
            memory_gate_passed=True,
            generated_text=generated_text,
            generated_tokens=gen_tokens,
            notes="FP16 dense baseline — no compression applied",
        )

    except Exception as exc:
        return CandidateResult(
            name="dense_mlx_baseline",
            model_id=model_id,
            prompt_id=prompt_id,
            error=str(exc),
            gate_status="ERROR",
            promotion_eligible=False,
            # Packed fields (baseline has no compression)
            packed_blocks_created=0,
            packed_blocks_read=0,
            packed_attention_calls=0,
            dense_fallback_calls=0,
            full_history_materialization_calls=0,
            packed_bytes_written=0,
            packed_bytes_read=0,
        )


# ---------------------------------------------------------------------------
# Stability check: two identical runs should produce near-identical metrics
# ---------------------------------------------------------------------------

def _check_determinism(
    result_a: CandidateResult,
    result_b: CandidateResult,
) -> tuple[bool, str]:
    """Return (stable, reason).  Checks text identity and TPS within 15 %."""
    if result_a.error or result_b.error:
        return False, f"run error: {result_a.error or result_b.error}"
    if result_a.generated_text.strip() != result_b.generated_text.strip():
        return False, "generated texts differ (temp=0 should be deterministic)"
    tps_a = result_a.decode_tps or 0
    tps_b = result_b.decode_tps or 0
    if tps_a > 0 and tps_b > 0:
        rel_diff = abs(tps_a - tps_b) / max(tps_a, tps_b)
        if rel_diff > 0.15:
            return False, f"decode_tps unstable: {tps_a:.1f} vs {tps_b:.1f} ({rel_diff*100:.1f}% diff)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _build_markdown_report(
    results: list[CandidateResult],
    model_id: str,
    timestamp: str,
    smoke: bool,
) -> str:
    lines = [
        "# RFSN Dense MLX Baseline Report",
        "",
        f"**Model:** `{model_id}`  ",
        f"**Generated:** {timestamp}  ",
        f"**Mode:** {'smoke (synthetic)' if smoke else 'real model'}  ",
        "",
        "## Per-Prompt Results",
        "",
        "| prompt_id | context_len | output_tokens | prefill_tps | decode_tps | TTFT_ms | total_ms | peak_mb | kv_mb |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def _f(v: float | None, fmt: str = ".1f") -> str:
            return format(v, fmt) if v is not None else "—"
        lines.append(
            f"| {r.prompt_id} "
            f"| {r.context_length} "
            f"| {r.output_tokens} "
            f"| {_f(r.prefill_tps)} "
            f"| {_f(r.decode_tps)} "
            f"| {_f(r.first_token_latency_ms)} "
            f"| {_f(r.total_latency_ms)} "
            f"| {_f(r.peak_memory_mb)} "
            f"| {_f(r.kv_cache_memory_mb)} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- All runs at temperature=0, seed=42.",
        "- Dense baseline has logit_cosine=1.0, top5_overlap=1.0 by definition.",
        "- `kv_mb` is an estimate based on model architecture; actual device allocation may differ.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RFSN dense MLX baseline runner")
    parser.add_argument("--model", default=None, help="HuggingFace model ID or local path")
    parser.add_argument("--quick", action="store_true", help="Small model + short prompts only")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: synthetic data, no model download")
    parser.add_argument("--output-tokens", type=int, default=100, help="Max tokens to generate per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None, help="Override output directory (default: benchmarks/results/)")
    parser.add_argument("--check-determinism", action="store_true", help="Run each prompt twice to check stability")
    args = parser.parse_args()

    # Output paths
    here = Path(__file__).parent
    results_dir = Path(args.out_dir) if args.out_dir else here / "results"
    reports_dir = here / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- Smoke mode ---
    if args.smoke:
        print("[smoke] Running synthetic baseline (no model download)")
        rng = np.random.default_rng(args.seed)
        prompts = PROMPT_SUITE_QUICK if args.quick else PROMPT_SUITE
        model_id = args.model or "smoke/Qwen2.5-0.5B"
        all_results: list[CandidateResult] = []
        for prompt_id, prompt in prompts.items():
            r = _make_smoke_result(model_id, prompt_id, prompt, args.output_tokens, rng)
            all_results.append(r)
            print(f"  {prompt_id}: decode_tps={r.decode_tps:.1f}  peak_mb={r.peak_memory_mb:.0f}")
        _save_results(all_results, model_id, timestamp, results_dir, reports_dir, smoke=True)
        print(f"\n[smoke] Results saved to {results_dir}/baseline_mlx_latest.json")
        return

    # --- Real model run ---
    models = [args.model] if args.model else (MODELS_QUICK if args.quick else MODELS_FULL)
    prompts = PROMPT_SUITE_QUICK if args.quick else PROMPT_SUITE

    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx_lm is not installed.  Run: pip install 'mlx-lm>=0.19'")
        print("       or use --smoke for synthetic mode.")
        sys.exit(1)

    for model_id in models:
        print(f"\nLoading {model_id} ...")
        try:
            model, tokenizer = mlx_lm.load(model_id)
        except Exception as exc:
            print(f"  ERROR loading {model_id}: {exc}")
            continue

        all_results = []
        for prompt_id, prompt in prompts.items():
            print(f"  Running: {prompt_id} ...", end=" ", flush=True)
            r = run_single(
                model, tokenizer, model_id, prompt_id, prompt,
                output_tokens=args.output_tokens, seed=args.seed,
            )
            if r.error:
                print(f"ERROR: {r.error}")
            else:
                print(
                    f"decode_tps={r.decode_tps:.1f}  "
                    f"TTFT={r.first_token_latency_ms:.0f}ms  "
                    f"peak={r.peak_memory_mb:.0f}MB"
                )

                if args.check_determinism:
                    print("    checking determinism ...", end=" ", flush=True)
                    r2 = run_single(
                        model, tokenizer, model_id, prompt_id, prompt,
                        output_tokens=args.output_tokens, seed=args.seed,
                    )
                    stable, reason = _check_determinism(r, r2)
                    print(f"{'STABLE' if stable else 'UNSTABLE'}: {reason}")

            all_results.append(r)

        _save_results(all_results, model_id, timestamp, results_dir, reports_dir, smoke=False)
        print(f"\nResults saved to {results_dir}/baseline_mlx_latest.json")


def _save_results(
    results: list[CandidateResult],
    model_id: str,
    timestamp: str,
    results_dir: Path,
    reports_dir: Path,
    smoke: bool,
) -> None:
    payload = {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "model_id": model_id,
        "smoke": smoke,
        "results": [r.to_dict() for r in results],
    }

    # Timestamped + latest
    ts_path = results_dir / f"baseline_mlx_{timestamp}.json"
    latest_json = results_dir / "baseline_mlx_latest.json"
    for p in (ts_path, latest_json):
        p.write_text(dumps_json_strict(payload, indent=2, default=str))

    # Markdown report
    md = _build_markdown_report(results, model_id, timestamp, smoke)
    ts_md = reports_dir / f"baseline_mlx_{timestamp}.md"
    latest_md = reports_dir / "baseline_mlx_latest.md"
    for p in (ts_md, latest_md):
        p.write_text(md)


if __name__ == "__main__":
    main()
