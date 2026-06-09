#!/usr/bin/env python3
"""
RFSN v10 - KV Cache Benchmarks.
Measures store latency, retrieve latency, compression ratio, quality metrics.
"""
from __future__ import annotations

import json
import platform
from statistics import median
import time
from datetime import datetime, timezone

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


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


def rel_mae(a: mx.array, b: mx.array) -> float:
    denom = mx.maximum(mx.mean(mx.abs(a)), mx.array(1e-8))
    return (mx.mean(mx.abs(a - b)) / denom).item()


def max_abs_error(a: mx.array, b: mx.array) -> float:
    return mx.max(mx.abs(a - b)).item()


def benchmark_kv(shape, k_bits, v_bits, use_incoherent, iterations=5):
    mx.random.seed(42)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, use_incoherent=use_incoherent,
            max_memory_gb=1.0, max_pinned_memory_gb=0.5, cache_dir=td,
        )
        k = mx.random.normal(shape)
        v = mx.random.normal(shape)
        mx.eval(k, v)

        # Store benchmark
        store_times = []
        for _ in range(iterations):
            t0 = time.monotonic()
            mgr.store("bench", k, v, shape[2])
            mx.eval(mgr.active_caches["bench"].k_packed)
            t1 = time.monotonic()
            store_times.append(t1 - t0)

        # Retrieve benchmark
        retrieve_times = []
        key_cosines = []
        key_rel_maes = []
        key_max_abs = []
        value_cosines = []
        value_rel_maes = []
        value_max_abs = []
        for _ in range(iterations):
            t0 = time.monotonic()
            result = mgr.retrieve("bench", out_dtype=mx.float32)
            k_rec, v_rec = result
            mx.eval(k_rec, v_rec)
            t1 = time.monotonic()
            retrieve_times.append(t1 - t0)
            key_cosines.append(cosine_similarity(k, k_rec))
            key_rel_maes.append(rel_mae(k, k_rec))
            key_max_abs.append(max_abs_error(k, k_rec))
            value_cosines.append(cosine_similarity(v, v_rec))
            value_rel_maes.append(rel_mae(v, v_rec))
            value_max_abs.append(max_abs_error(v, v_rec))

        cache = mgr.active_caches["bench"]
        original_bytes = k.size * 4 + v.size * 4
        packed_bytes = (cache.k_packed.size + cache.v_packed.size) * 4 + \
                       (cache.k_scales.size + cache.v_scales.size) * 4

        return {
            "shape": str(shape),
            "k_bits": k_bits,
            "v_bits": v_bits,
            "use_incoherent": use_incoherent,
            "store_latency_ms": float(median(store_times) * 1000.0),
            "retrieve_latency_ms": float(median(retrieve_times) * 1000.0),
            "original_bytes": original_bytes,
            "packed_bytes": packed_bytes,
            "compression_ratio": packed_bytes / original_bytes,
            "key_cosine_sim": sum(key_cosines) / len(key_cosines),
            "key_rel_mae": sum(key_rel_maes) / len(key_rel_maes),
            "key_max_abs_error": sum(key_max_abs) / len(key_max_abs),
            "value_cosine_sim": sum(value_cosines) / len(value_cosines),
            "value_rel_mae": sum(value_rel_maes) / len(value_rel_maes),
            "value_max_abs_error": sum(value_max_abs) / len(value_max_abs),
        }


def main():
    print("=" * 60)
    print("RFSN v10 KV Cache Benchmarks")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    shapes = [(1, 8, 1024, 64), (1, 8, 2048, 64), (1, 32, 4096, 128)]
    configs = [
        (8, 3, True),
        (8, 3, False),
        (8, 8, False),
    ]

    results = {"metadata": meta, "runs": []}

    for shape in shapes:
        for k_bits, v_bits, use_inc in configs:
            r = benchmark_kv(shape, k_bits, v_bits, use_inc)
            line = (
                f"  {shape} k={k_bits}b v={v_bits}b incoherent={use_inc}: "
                f"store={r['store_latency_ms']:.2f}ms "
                f"retrieve={r['retrieve_latency_ms']:.2f}ms "
                f"ratio={r['compression_ratio']:.3f} "
                f"k_cos={r['key_cosine_sim']:.4f} "
                f"v_cos={r['value_cosine_sim']:.4f}"
            )
            print(line)
            results["runs"].append(r)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
