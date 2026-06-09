#!/usr/bin/env python3
"""Benchmark improved sparse attention strategies.

Compares different block selection strategies to identify which
achieves the target 0.90+ sparse quality threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.attention_improved import QualityAwareSparseAttention


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two tensors."""
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def benchmark_strategy(
    strategy: str,
    shape: tuple[int, int, int, int],
    top_k_ratio: float,
    iterations: int = 5,
) -> dict[str, Any]:
    """Benchmark a specific block selection strategy."""
    mx.random.seed(42)

    queries = mx.random.normal(shape)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)

    # Dense baseline
    dense_out = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=1.0 / math.sqrt(shape[3])
    )

    # Sparse with strategy
    attention = QualityAwareSparseAttention(
        block_size=64,
        selection_strategy=strategy,
        enable_quality_monitoring=False,
    )

    latencies = []
    sparse_outputs = []
    for _ in range(iterations):
        import time

        t0 = time.perf_counter()
        sparse_out, num_blocks, mode, metadata = attention.execute(
            queries=queries,
            keys=keys,
            values=values,
            top_k_ratio=top_k_ratio,
            layer_id="test",
        )
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        sparse_outputs.append(sparse_out)

    # Compute quality metrics
    quality_metrics = []
    for sparse_out in sparse_outputs:
        cos_sim = cosine_similarity(dense_out, sparse_out)
        max_abs_diff = float(mx.max(mx.abs(dense_out - sparse_out)).item())
        quality_metrics.append({"cosine": cos_sim, "max_abs_diff": max_abs_diff})

    avg_cosine = sum(m["cosine"] for m in quality_metrics) / len(quality_metrics)
    avg_max_abs = sum(m["max_abs_diff"] for m in quality_metrics) / len(
        quality_metrics
    )

    return {
        "strategy": strategy,
        "shape": list(shape),
        "top_k_ratio": top_k_ratio,
        "iterations": iterations,
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "num_active_blocks": num_blocks,
        "cosine_vs_dense": avg_cosine,
        "max_abs_diff_vs_dense": avg_max_abs,
        "meets_threshold": avg_cosine >= 0.90,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark improved sparse attention strategies"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main12/sparse_improvements_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = ["hybrid", "attention_max", "variance", "gradient"]
    shapes = [(1, 8, 1024, 64), (1, 8, 2048, 64), (1, 32, 4096, 128)]
    top_k_ratios = [0.5, 0.75, 0.875]

    results = []
    for strategy in strategies:
        for shape in shapes:
            for top_k_ratio in top_k_ratios:
                print(
                    f"Benchmarking: {strategy}, shape={shape}, "
                    f"top_k={top_k_ratio}"
                )
                result = benchmark_strategy(
                    strategy, shape, top_k_ratio, args.iterations
                )
                results.append(result)
                print(
                    f"  Cosine: {result['cosine_vs_dense']:.4f}, "
                    f"Meets threshold: {result['meets_threshold']}"
                )

    # Summary statistics
    strategy_summary = {}
    for strategy in strategies:
        strategy_results = [r for r in results if r["strategy"] == strategy]
        avg_cosine = sum(r["cosine_vs_dense"] for r in strategy_results) / len(
            strategy_results
        )
        meets_threshold_count = sum(1 for r in strategy_results if r["meets_threshold"])
        strategy_summary[strategy] = {
            "avg_cosine": avg_cosine,
            "meets_threshold_rate": meets_threshold_count / len(strategy_results),
            "total_runs": len(strategy_results),
        }

    payload = {
        "results": results,
        "summary": strategy_summary,
        "best_strategy": max(
            strategy_summary.items(),
            key=lambda x: x[1]["avg_cosine"],
        )[0],
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path}")
    print(f"\nBest strategy: {payload['best_strategy']}")
    for strategy, stats in strategy_summary.items():
        print(
            f"  {strategy}: avg_cosine={stats['avg_cosine']:.4f}, "
            f"threshold_rate={stats['meets_threshold_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
