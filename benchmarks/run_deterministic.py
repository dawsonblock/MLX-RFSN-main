#!/usr/bin/env python3
"""Deterministic benchmark runner — Week 6.

Runs core benchmarks with fixed seeds, warm-up iterations, and median-of-N
sampling.  Results are saved as JSON in ``benchmarks/results/``.

Usage:
    python benchmarks/run_deterministic.py
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx

from rfsn_v10.attention import AdaptiveBlockSparseAttention
from rfsn_v10.kernels import backend

RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
WARMUP = 3
ITERATIONS = 10


def _timed(fn, *args, **kwargs):
    """Run fn once and return (result, elapsed_ms)."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    mx.eval(out)
    elapsed = (time.perf_counter() - start) * 1000
    return out, elapsed


def bench_attention(
    b: int, h: int, t_q: int, t_k: int, d: int, top_k: float, block_size: int
) -> dict:
    """Benchmark sparse vs dense attention."""
    q = mx.random.normal((b, h, t_q, d))
    k = mx.random.normal((b, h, t_k, d))
    v = mx.random.normal((b, h, t_k, d))
    mx.eval(q, k, v)

    for _ in range(WARMUP):
        out, _, _ = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=top_k, block_size=block_size
        )
        mx.eval(out)

    sparse_ms = []
    dense_ms = []
    for _ in range(ITERATIONS):
        _, ms = _timed(
            AdaptiveBlockSparseAttention.execute,
            q, k, v, top_k_ratio=top_k, block_size=block_size,
        )
        sparse_ms.append(ms)
        _, ms = _timed(
            AdaptiveBlockSparseAttention.execute,
            q, k, v, top_k_ratio=1.0, block_size=block_size,
        )
        dense_ms.append(ms)

    return {
        "sparse_ms": median(sparse_ms),
        "dense_ms": median(dense_ms),
        "speedup": median(dense_ms) / max(median(sparse_ms), 1e-6),
    }


def bench_quantized_decode(
    n_h: int, n_keys: int, d_head: int, bits: int, group_size: int
) -> dict:
    """Benchmark quantized attention decode."""
    n_values = n_h * n_keys * d_head
    codes_per_word = 32 // bits
    n_words = (n_values + codes_per_word - 1) // codes_per_word
    packed_k = mx.zeros((n_words,), dtype=mx.uint32)
    packed_v = mx.zeros((n_words,), dtype=mx.uint32)
    n_scales = (n_values + group_size - 1) // group_size
    scales_k = mx.ones((n_scales,), dtype=mx.float32)
    scales_v = mx.ones((n_scales,), dtype=mx.float32)
    queries = mx.ones((n_h, d_head), dtype=mx.float32)
    mx.eval(packed_k, packed_v, scales_k, scales_v, queries)

    for _ in range(WARMUP):
        out = backend.quantized_attention_decode(
            queries, packed_k, packed_v, scales_k, scales_v,
            n_keys=n_keys, bits=bits, group_size=group_size,
        )
        mx.eval(out)

    times = []
    for _ in range(ITERATIONS):
        _, ms = _timed(
            backend.quantized_attention_decode,
            queries, packed_k, packed_v, scales_k, scales_v,
            n_keys=n_keys, bits=bits, group_size=group_size,
        )
        times.append(ms)

    return {"decode_ms": median(times)}


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mx.random.seed(42)

    results: dict[str, dict] = {}
    results["metadata"] = {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "backend": backend.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Attention sweep
    for T_k in [256, 512, 1024]:
        label = f"attention_Tk={T_k}"
        results[label] = bench_attention(
            b=1, h=8, t_q=1, t_k=T_k, d=64,
            top_k=0.5, block_size=64,
        )

    # Quantized decode sweep
    for n_keys in [1024, 4096, 16384]:
        label = f"quant_decode_nkeys={n_keys}"
        results[label] = bench_quantized_decode(
            n_h=8, n_keys=n_keys, d_head=64, bits=4, group_size=64,
        )

    out_path = RESULTS_DIR / "latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path}")
    for k, v in results.items():
        if k == "metadata":
            continue
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
