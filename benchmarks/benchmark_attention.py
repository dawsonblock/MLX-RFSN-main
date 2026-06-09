#!/usr/bin/env python3
"""
RFSN v10 - Attention Benchmarks.
Measures dense vs sparse decode latency, top_k_ratio sweep, accuracy drift.
"""
from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone

try:
    import mlx.core as mx
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "MLX is required for attention benchmarks. "
        "Run this on Apple Silicon with: pip install mlx"
    ) from exc

from rfsn_v10.attention import AdaptiveBlockSparseAttention


def get_metadata() -> dict:
    return {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def benchmark_attention(shape_q, shape_kv, top_k_ratio, block_size=64, iterations=10):
    mx.random.seed(42)
    B, H, T_q, D = shape_q
    B_k, H_k, T_k, D_k = shape_kv

    q = mx.random.normal(shape_q)
    k = mx.random.normal(shape_kv)
    v = mx.random.normal(shape_kv)
    mx.eval(q, k, v)

    # Dense baseline
    scale = 1.0 / math.sqrt(D)
    dense_times = []
    for _ in range(iterations):
        t0 = time.monotonic()
        out_dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(out_dense)
        t1 = time.monotonic()
        dense_times.append(t1 - t0)

    # Sparse
    sparse_times = []
    active_blocks_list = []
    cosines = []
    execution_modes = []
    for _ in range(iterations):
        t0 = time.monotonic()
        out_sparse, n_active, execution_mode = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=top_k_ratio, block_size=block_size,
        )
        mx.eval(out_sparse)
        t1 = time.monotonic()
        sparse_times.append(t1 - t0)
        active_blocks_list.append(n_active)
        execution_modes.append(execution_mode)

        dense_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(dense_out)
        cosines.append(cosine_similarity(out_sparse, dense_out))

    return {
        "top_k_ratio": top_k_ratio,
        "block_size": block_size,
        "shape_q": str(shape_q),
        "shape_kv": str(shape_kv),
        "dense_latency_ms": (sum(dense_times) / len(dense_times)) * 1000.0,
        "sparse_latency_ms": (sum(sparse_times) / len(sparse_times)) * 1000.0,
        "speedup": (sum(dense_times) / len(dense_times)) / (sum(sparse_times) / len(sparse_times)) if sum(sparse_times) > 0 else 0,
        "avg_active_blocks": sum(active_blocks_list) / len(active_blocks_list),
        "avg_cosine_vs_dense": sum(cosines) / len(cosines),
        "execution_mode": execution_modes[-1] if execution_modes else "unknown",
    }


def main():
    print("=" * 60)
    print("RFSN v10 Attention Benchmarks")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    shape_q = (1, 8, 1, 64)
    shape_kv = (1, 8, 2048, 64)
    top_k_ratios = [1.0, 0.75, 0.50, 0.25, 0.125]

    results = {"metadata": meta, "runs": []}

    for ratio in top_k_ratios:
        r = benchmark_attention(shape_q, shape_kv, ratio)
        print(f"  top_k={ratio}: sparse={r['sparse_latency_ms']:.3f}ms "
              f"dense={r['dense_latency_ms']:.3f}ms "
              f"speedup={r['speedup']:.2f}x "
              f"cosine={r['avg_cosine_vs_dense']:.4f} "
              f"blocks={r['avg_active_blocks']:.0f}")
        results["runs"].append(r)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
