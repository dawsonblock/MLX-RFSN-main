#!/usr/bin/env python3
"""Performance benchmarking script for direct packed reference.

This script benchmarks the performance of direct packed reference attention
compared to dense baseline, measuring:
- Tokens per second (TPS)
- Latency per token
- Memory usage
- Speedup factor

This should be run after the reference implementation is verified correct.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def benchmark_performance(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 10,
    max_tokens: int = 200,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> dict[str, Any]:
    """Benchmark performance of direct packed reference.

    Args:
        model_id: Model to test with.
        prompt: Test prompt.
        max_tokens: Maximum tokens to generate.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.
        warmup_runs: Number of warmup runs before benchmarking.
        benchmark_runs: Number of benchmark runs.

    Returns:
        Dictionary containing performance benchmark results.
    """
    try:
        import mlx_lm
        import mlx.core as mx
    except ImportError as e:
        return {
            "error": f"MLX not available: {e}",
            "status": "SKIPPED_NO_MLX",
        }

    print(f"Loading model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)

    results = {
        "model_id": model_id,
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
        },
        "baseline_performance": {},
        "quantized_performance": {},
        "comparison": {},
    }

    # Benchmark baseline (dense)
    print("Benchmarking baseline (dense)...")
    baseline_results = _benchmark_baseline(
        model, tokenizer, prompt, max_tokens, warmup_runs, benchmark_runs
    )
    results["baseline_performance"] = baseline_results

    # Benchmark quantized (direct packed)
    print("Benchmarking quantized (direct packed)...")
    quantized_results = _benchmark_quantized(
        model, tokenizer, prompt, max_tokens,
        key_bits, value_bits, group_size, warmup_runs, benchmark_runs
    )
    results["quantized_performance"] = quantized_results

    # Compare performance
    results["comparison"] = _compare_performance(baseline_results, quantized_results)

    return results


def _benchmark_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    """Benchmark baseline dense attention."""
    import mlx.core as mx

    prompt_ids = tokenizer.encode(prompt)

    # Warmup
    for _ in range(warmup_runs):
        y = mx.array(prompt_ids)
        for _ in range(max_tokens):
            _ = model(y[None])
            token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
            y = mx.array([token_id])

    # Benchmark
    total_tokens = 0
    total_time = 0.0
    latencies = []

    for run in range(benchmark_runs):
        y = mx.array(prompt_ids)
        run_tokens = 0
        run_start = time.perf_counter()

        for _ in range(max_tokens):
            token_start = time.perf_counter()
            _ = model(y[None])
            token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
            y = mx.array([token_id])
            token_end = time.perf_counter()
            latencies.append((token_end - token_start) * 1000)  # ms
            run_tokens += 1

        run_end = time.perf_counter()
        total_tokens += run_tokens
        total_time += (run_end - run_start)

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "mean_tps": total_tokens / total_time,
        "mean_latency_ms": (total_time / total_tokens) * 1000,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
    }


def _benchmark_quantized(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    """Benchmark quantized direct packed attention."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx

    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name="performance_benchmark",
        num_layers=len(model.layers),
        key_codec=key_codec,
        value_codec=value_codec,
    )

    cache_list = [
        RfsnQuantizedKVCache(
            layer_cache=session.get_layer_cache(i),
            session=session,
        )
        for i in range(len(model.layers))
    ]

    cfg = RFSNConfig(
        quantization=QuantizationConfig(
            default_bits=key_bits,
            group_size=group_size,
        ),
    )
    generator = RFSNGenerator(
        model,
        tokenizer,
        cfg,
        enable_quantized_kv=True,
    )

    prompt_ids = tokenizer.encode(prompt)

    # Warmup
    for _ in range(warmup_runs):
        y = mx.array(prompt_ids)
        for _ in range(max_tokens):
            _ = model(y[None], cache=cache_list)
            token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
            y = mx.array([token_id])

    # Benchmark
    total_tokens = 0
    total_time = 0.0
    latencies = []

    for run in range(benchmark_runs):
        y = mx.array(prompt_ids)
        run_tokens = 0
        run_start = time.perf_counter()

        for _ in range(max_tokens):
            token_start = time.perf_counter()
            _ = model(y[None], cache=cache_list)
            token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
            y = mx.array([token_id])
            token_end = time.perf_counter()
            latencies.append((token_end - token_start) * 1000)  # ms
            run_tokens += 1

        run_end = time.perf_counter()
        total_tokens += run_tokens
        total_time += (run_end - run_start)

    # Get runtime counters
    counters = session.counters()

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "mean_tps": total_tokens / total_time,
        "mean_latency_ms": (total_time / total_tokens) * 1000,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "runtime_counters": counters,
    }


def _compare_performance(
    baseline: dict[str, Any],
    quantized: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline and quantized performance."""
    speedup = quantized["mean_tps"] / baseline["mean_tps"]
    latency_improvement = baseline["mean_latency_ms"] / quantized["mean_latency_ms"]

    return {
        "speedup_factor": speedup,
        "latency_improvement_factor": latency_improvement,
        "tps_difference": quantized["mean_tps"] - baseline["mean_tps"],
        "latency_difference_ms": baseline["mean_latency_ms"] - quantized["mean_latency_ms"],
    }


def _percentile(values: list[float], p: int) -> float:
    """Calculate percentile."""
    import statistics
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    if f == c:
        return sorted_values[f]
    d = k - f
    return sorted_values[f] * (1 - d) + sorted_values[c] * d


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark direct packed reference performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model to test with",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog. " * 10,
        help="Test prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=8,
        help="Key quantization bits",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=8,
        help="Value quantization bits",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/performance_benchmark.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Direct Packed Reference Performance Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")
    print(f"Warmup runs: {args.warmup_runs}, Benchmark runs: {args.benchmark_runs}")

    results = benchmark_performance(
        args.model, args.prompt, args.max_tokens,
        args.key_bits, args.value_bits, args.group_size,
        args.warmup_runs, args.benchmark_runs
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Benchmark skipped: MLX not available")
        return 1

    baseline = results.get("baseline_performance", {})
    quantized = results.get("quantized_performance", {})
    comparison = results.get("comparison", {})

    print(f"\nBaseline (dense):")
    print(f"  TPS: {baseline['mean_tps']:.2f}")
    print(f"  Latency: {baseline['mean_latency_ms']:.2f} ms")

    print(f"\nQuantized (direct packed):")
    print(f"  TPS: {quantized['mean_tps']:.2f}")
    print(f"  Latency: {quantized['mean_latency_ms']:.2f} ms")

    print(f"\nComparison:")
    print(f"  Speedup: {comparison['speedup_factor']:.2f}x")
    print(f"  Latency improvement: {comparison['latency_improvement_factor']:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
