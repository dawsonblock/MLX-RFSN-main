#!/usr/bin/env python3
"""Benchmark and profile RFSN v10 operations.

Runs comprehensive profiling to identify bottlenecks and optimization opportunities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from rfsn_v10.kernels import (
    apply_hash_signs_metal,
    packed_dequant_metal,
    packed_dequant_wht_sign_metal,
    wht64_metal,
)
from rfsn_v10.profiler import RFSNProfiler, profile_kernel_execution
from rfsn_v10.bitpack import BitPackedQuantizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RFSN v10 operations")
    parser.add_argument(
        "--out",
        default="artifacts/proof/main12/profiling_report.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mx.random.seed(42)

    # Test data
    n_values = 16384
    bits = 8
    group_size = 64
    seed = 42

    values = mx.random.normal((n_values,))
    quantizer = BitPackedQuantizer(bits=bits, group_size=group_size)
    packed, scales = quantizer.quantize(values)

    profiler = RFSNProfiler()
    report = {"operations": {}, "bottlenecks": []}

    # Profile individual kernels
    print("Profiling packed_dequant_metal...")
    dequant_stats = profile_kernel_execution(
        packed_dequant_metal,
        packed,
        scales,
        n_values,
        bits,
        group_size,
        mx.float32,
        iterations=args.iterations,
    )
    report["operations"]["packed_dequant"] = dequant_stats
    print(f"  Avg: {dequant_stats['avg_latency_ms']:.3f}ms")

    print("Profiling wht64_metal...")
    deq = packed_dequant_metal(packed, scales, n_values, bits, group_size, mx.float32)
    wht_stats = profile_kernel_execution(
        wht64_metal,
        deq,
        iterations=args.iterations,
    )
    report["operations"]["wht64"] = wht_stats
    print(f"  Avg: {wht_stats['avg_latency_ms']:.3f}ms")

    print("Profiling apply_hash_signs_metal...")
    wht = wht64_metal(deq)
    sign_stats = profile_kernel_execution(
        apply_hash_signs_metal,
        wht,
        seed,
        iterations=args.iterations,
    )
    report["operations"]["hash_sign"] = sign_stats
    print(f"  Avg: {sign_stats['avg_latency_ms']:.3f}ms")

    print("Profiling fused packed_dequant_wht_sign_metal...")
    fused_stats = profile_kernel_execution(
        packed_dequant_wht_sign_metal,
        packed,
        scales,
        n_values,
        bits,
        group_size,
        seed,
        mx.float32,
        iterations=args.iterations,
    )
    report["operations"]["fused"] = fused_stats
    print(f"  Avg: {fused_stats['avg_latency_ms']:.3f}ms")

    # Calculate sequential vs fused comparison
    sequential_total = (
        dequant_stats["avg_latency_ms"]
        + wht_stats["avg_latency_ms"]
        + sign_stats["avg_latency_ms"]
    )
    fused_total = fused_stats["avg_latency_ms"]
    speedup = sequential_total / fused_total

    report["comparison"] = {
        "sequential_total_ms": sequential_total,
        "fused_total_ms": fused_total,
        "speedup": speedup,
        "improvement_pct": ((sequential_total - fused_total) / sequential_total) * 100,
    }

    # Identify bottlenecks
    ops = [
        ("packed_dequant", dequant_stats["avg_latency_ms"]),
        ("wht64", wht_stats["avg_latency_ms"]),
        ("hash_sign", sign_stats["avg_latency_ms"]),
    ]
    ops.sort(key=lambda x: x[1], reverse=True)

    report["bottlenecks"] = [
        {"operation": name, "latency_ms": latency, "pct_of_total": latency / sequential_total * 100}
        for name, latency in ops
    ]

    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nProfiling report written to: {out_path}")
    print(f"\nSequential total: {sequential_total:.3f}ms")
    print(f"Fused total: {fused_total:.3f}ms")
    print(f"Speedup: {speedup:.2f}x ({report['comparison']['improvement_pct']:.1f}% improvement)")


if __name__ == "__main__":
    main()
