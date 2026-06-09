#!/usr/bin/env python3
"""Benchmark optimization configs: bit-width and group_size variations."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager

SHAPES = [
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
    (1, 32, 8192, 128),
]


CONFIGS = [
    {"name": "baseline_8_3_gs64",
     "k_bits": 8, "v_bits": 3, "group_size": 64},
    {"name": "2bit_2_2_gs64",
     "k_bits": 2, "v_bits": 2, "group_size": 64},
    {"name": "3bit_3_3_gs64",
     "k_bits": 3, "v_bits": 3, "group_size": 64},
    {"name": "4bit_4_4_gs64",
     "k_bits": 4, "v_bits": 4, "group_size": 64},
    {"name": "baseline_8_3_gs128",
     "k_bits": 8, "v_bits": 3, "group_size": 128},
    {"name": "baseline_8_3_gs256",
     "k_bits": 8, "v_bits": 3, "group_size": 256},
    {"name": "2bit_2_2_gs128",
     "k_bits": 2, "v_bits": 2, "group_size": 128},
    {"name": "2bit_2_2_gs256",
     "k_bits": 2, "v_bits": 2, "group_size": 256},
]


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def benchmark_config(shape, cfg, iterations=10):
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            cache_dir=td,
            k_bits=cfg["k_bits"],
            v_bits=cfg["v_bits"],
            group_size=cfg["group_size"],
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            prefer_fused_kernel=True,
        )

        x = mx.random.normal(shape).astype(mx.float16)
        seq_len = shape[2]
        mgr.store("key", x, x, seq_len)

        # Reference for quality measurement (no quantization)
        ref = x.astype(mx.float32)

        # Warmup
        out = mgr.retrieve("key")
        mx.eval(*out)

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = mgr.retrieve("key")
            mx.eval(*out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        k_rec, v_rec = out
        k_cos = cosine_similarity(k_rec, ref)
        v_cos = cosine_similarity(v_rec, ref)

        # Partial reconstruction benchmarks
        cache = mgr.active_caches["key"]
        num_blocks = cache.num_blocks
        if num_blocks > 0:
            half_blocks = list(range(0, num_blocks // 2))
            half_times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                out_partial = mgr.retrieve_blocks(
                    "key", half_blocks, block_size=cfg.get("block_size", 64),
                )
                mx.eval(*out_partial)
                t1 = time.perf_counter()
                half_times.append((t1 - t0) * 1000)

            sparse_blocks = list(range(0, num_blocks, 2))
            sparse_times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                out_sparse = mgr.retrieve_blocks(
                    "key", sparse_blocks, block_size=cfg.get("block_size", 64),
                )
                mx.eval(*out_sparse)
                t1 = time.perf_counter()
                sparse_times.append((t1 - t0) * 1000)

            half_p50 = statistics.median(half_times)
            sparse_p50 = statistics.median(sparse_times)
            full_p50 = statistics.median(times)
            half_speedup = full_p50 / half_p50 if half_p50 > 0 else 0.0
            sparse_speedup = full_p50 / sparse_p50 if sparse_p50 > 0 else 0.0
        else:
            half_p50 = 0.0
            sparse_p50 = 0.0
            half_speedup = 0.0
            sparse_speedup = 0.0

        return {
            "shape": shape,
            "config": cfg["name"],
            "k_bits": cfg["k_bits"],
            "v_bits": cfg["v_bits"],
            "group_size": cfg["group_size"],
            "mean_ms": statistics.mean(times),
            "p50_ms": statistics.median(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)],
            "k_cosine_vs_ref": k_cos,
            "v_cosine_vs_ref": v_cos,
            "min_cosine": min(k_cos, v_cos),
            "half_blocks_ms": half_p50,
            "sparse_blocks_ms": sparse_p50,
            "half_speedup": half_speedup,
            "sparse_speedup": sparse_speedup,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output", type=str,
        default="artifacts/proof/main24/optimization_benchmark.json",
    )
    args = parser.parse_args()

    mx.random.seed(42)
    all_results = []

    print("=== Optimization Sweep ===")
    hdr = f"{'Config':<25} {'Shape':<25} {'Mean(ms)':<10}"
    hdr += f" {'P50(ms)':<10} {'MinCos':<10} {'HalfSpd':<8} {'SparseSpd':<9}"
    print(hdr)
    print("-" * 95)

    for shape in SHAPES:
        for cfg in CONFIGS:
            try:
                result = benchmark_config(
                    shape, cfg, iterations=args.iterations,
                )
                all_results.append(result)
                print(
                    f"{result['config']:<25} "
                    f"{str(shape):<25} "
                    f"{result['mean_ms']:<10.2f} "
                    f"{result['p50_ms']:<10.2f} "
                    f"{result['min_cosine']:<10.6f} "
                    f"{result['half_speedup']:<8.2f} "
                    f"{result['sparse_speedup']:<9.2f}"
                )
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"{cfg['name']:<25} {str(shape):<25} FAILED: {e}")
                all_results.append({
                    "shape": shape,
                    "config": cfg["name"],
                    "error": str(e),
                })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": all_results, "configs_tested": len(CONFIGS)},
            f, indent=2,
        )

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
