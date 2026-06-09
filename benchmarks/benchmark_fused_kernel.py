#!/usr/bin/env python3
"""Benchmark fused kernel performance vs sequential reference.

Compares the fused packed-dequant-WHT-sign Metal kernel against
the sequential multi-kernel path and the sequential reference route.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.kernels import packed_dequant_wht_sign_metal
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def _cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def _rel_mae(a: mx.array, b: mx.array) -> float:
    denom = mx.maximum(mx.mean(mx.abs(a)), mx.array(1e-8))
    return (mx.mean(mx.abs(a - b)) / denom).item()


def _setup_quantized_data(
    shape: tuple[int, ...],
    bits: int,
    group_size: int,
    seed: int,
) -> tuple[mx.array, mx.array, int, tuple[int, ...]]:
    """Quantize random data using the manager's path."""
    manager = RFSNTurboQuantKVManager(
        k_bits=bits,
        v_bits=bits,
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=False,
        group_size=group_size,
    )
    x = mx.random.normal(shape)
    x_pre = manager._apply_signs_on_the_fly(x, seed)
    x_wht = manager._apply_wht_pretransform(x_pre)
    q, scales = manager._quantize(x_wht.reshape(-1), bits)
    packed, n_v = BitPackedQuantizer.pack(q, bits)
    return packed, scales, n_v, shape


def _reference_reconstruction(
    manager: RFSNTurboQuantKVManager,
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    shape: tuple[int, ...],
    bits: int,
    seed: int,
) -> mx.array:
    """Sequential reference reconstruction via manager methods."""
    codes = BitPackedQuantizer.unpack(packed, n_values, bits)
    deq = manager._dequantize_unsigned(codes, scales, bits).reshape(shape)
    ref = manager._apply_wht_pretransform(deq)
    ref = manager._apply_signs_on_the_fly(ref, seed)
    return ref


def benchmark_fused_kernel(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    shape: tuple[int, ...],
    bits: int,
    group_size: int,
    seed: int,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark the fused single-kernel path."""
    latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        out = packed_dequant_wht_sign_metal(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=group_size,
            seed=seed,
            out_dtype=mx.float32,
        )
        mx.eval(out)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    ordered = sorted(latencies)
    mean = sum(ordered) / len(ordered)
    p50 = ordered[len(ordered) // 2]
    p95_index = max(
        0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    )
    p95 = ordered[p95_index]

    return {
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused kernel performance"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main24/fused_kernel_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        {"shape": (1, 8, 2048, 64), "bits": 8, "group_size": 64},
        {"shape": (1, 8, 2048, 64), "bits": 3, "group_size": 64},
        {"shape": (1, 8, 1024, 64), "bits": 8, "group_size": 64},
        {"shape": (1, 8, 1024, 64), "bits": 3, "group_size": 64},
        {"shape": (1, 4, 128, 64), "bits": 8, "group_size": 64},
        {"shape": (1, 4, 128, 64), "bits": 3, "group_size": 64},
    ]

    results: list[dict[str, Any]] = []
    seed = 42
    mx.random.seed(42)

    for config in configs:
        shape = config["shape"]
        bits = config["bits"]
        group_size = config["group_size"]

        print(f"Benchmarking: shape={shape}, bits={bits}")

        packed, scales, n_values, _ = _setup_quantized_data(
            shape, bits, group_size, seed
        )

        manager = RFSNTurboQuantKVManager(
            k_bits=bits,
            v_bits=bits,
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=False,
            group_size=group_size,
        )

        # Fused kernel benchmark
        fused_stats = benchmark_fused_kernel(
            packed, scales, n_values, shape,
            bits, group_size, seed, args.iterations,
        )
        print(f"  Fused mean: {fused_stats['latency_ms_mean']:.3f}ms")

        # Reference reconstruction for comparison
        ref = _reference_reconstruction(
            manager, packed, scales, n_values, shape, bits, seed
        )

        # Fused result
        fused = packed_dequant_wht_sign_metal(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=group_size,
            seed=seed,
            out_dtype=mx.float32,
        ).reshape(shape)
        mx.eval(fused, ref)

        cosine = _cosine_similarity(ref, fused)
        max_abs_diff = float(mx.max(mx.abs(ref - fused)).item())
        rel_mae = _rel_mae(ref, fused)

        print(f"  Cosine vs reference: {cosine:.6f}")
        print(f"  Max abs diff: {max_abs_diff:.6e}")
        print(f"  Rel MAE: {rel_mae:.6e}")

        fallback_used = False
        status = (
            "valid"
            if not fallback_used and cosine >= 0.999 and max_abs_diff <= 1e-3
            else "invalid"
        )

        row = {
            "route": "metal_fused_dequant_wht_sign",
            "shape": list(shape),
            "bits": bits,
            "latency_ms_mean": fused_stats["latency_ms_mean"],
            "latency_ms_p50": fused_stats["latency_ms_p50"],
            "latency_ms_p95": fused_stats["latency_ms_p95"],
            "cosine_vs_reference": cosine,
            "max_abs_diff_vs_reference": max_abs_diff,
            "rel_mae_vs_reference": rel_mae,
            "fallback_used": fallback_used,
            "status": status,
        }
        results.append(row)

    all_valid = all(r["status"] == "valid" for r in results)
    avg_cosine = sum(r["cosine_vs_reference"] for r in results) / len(results)
    avg_max_diff = (
        sum(r["max_abs_diff_vs_reference"] for r in results) / len(results)
    )

    payload = {
        "results": results,
        "summary": {
            "all_valid": all_valid,
            "avg_cosine_vs_reference": avg_cosine,
            "avg_max_abs_diff_vs_reference": avg_max_diff,
        },
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path}")
    print(f"All valid: {all_valid}")
    print(f"Average cosine: {avg_cosine:.6f}")


if __name__ == "__main__":
    main()
