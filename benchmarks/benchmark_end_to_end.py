#!/usr/bin/env python3
"""
RFSN v10 - End-to-End Runtime Benchmarks.
Measures full decode pipeline: KV store + sparse attention + audit.
"""
from __future__ import annotations

import json
import platform
from statistics import median
import tempfile
from datetime import datetime, timezone

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10 import RFSNRuntime


def get_metadata() -> dict:
    return {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_e2e(
    shape_q,
    shape_kv,
    k_bits,
    v_bits,
    use_incoherent,
    top_k_ratio,
    reserved_sink_blocks=1,
    reserved_recent_blocks=2,
    enable_sparse_decode=True,
    use_compressed_on_miss=False,
    iterations=5,
):
    mx.random.seed(42)
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, use_incoherent=use_incoherent,
            max_memory_gb=1.0, max_pinned_memory_gb=0.5, cache_dir=td,
        )
        runtime = RFSNRuntime(
            kv_manager=mgr, model_id="bench_model", block_size=64,
            audit_mode=True, top_k_ratio=top_k_ratio,
            enable_sparse_decode=enable_sparse_decode,
            reserved_sink_blocks=reserved_sink_blocks,
            reserved_recent_blocks=reserved_recent_blocks,
            use_compressed_on_miss=use_compressed_on_miss,
        )

        q = mx.random.normal(shape_q)
        k = mx.random.normal(shape_kv)
        v = mx.random.normal(shape_kv)
        mx.eval(q, k, v)

        # Measure cache-miss latency across distinct batch_ids to reduce noise.
        miss_timings = []
        miss_info = None
        for i in range(iterations):
            _, miss_info = runtime.execute_decode_step(
                skill_pattern="bench",
                layer_id="l0",
                batch_id=f"b_miss_{i}",
                queries=q,
                keys=k,
                values=v,
                top_k_ratio=top_k_ratio,
                reserved_sink_blocks=reserved_sink_blocks,
                reserved_recent_blocks=reserved_recent_blocks,
            )
            miss_timings.append(miss_info["total_latency_ms"])

        hit_timings = []
        hit_infos = []

        # Seed a dedicated cache key for hit-path timing.
        runtime.execute_decode_step(
            skill_pattern="bench",
            layer_id="l0",
            batch_id="b_hit",
            queries=q,
            keys=k,
            values=v,
            top_k_ratio=top_k_ratio,
            reserved_sink_blocks=reserved_sink_blocks,
            reserved_recent_blocks=reserved_recent_blocks,
        )

        for _ in range(iterations):
            _, info = runtime.execute_decode_step(
                skill_pattern="bench", layer_id="l0", batch_id="b_hit",
                queries=q, keys=k, values=v, top_k_ratio=top_k_ratio,
                reserved_sink_blocks=reserved_sink_blocks,
                reserved_recent_blocks=reserved_recent_blocks,
            )
            hit_timings.append(info["total_latency_ms"])
            hit_infos.append(info)

        telemetry = runtime.get_telemetry()
        latest = telemetry[-1] if telemetry else None

        return {
            "shape_q": str(shape_q),
            "shape_kv": str(shape_kv),
            "k_bits": k_bits,
            "v_bits": v_bits,
            "top_k_ratio": top_k_ratio,
            "reserved_sink_blocks": reserved_sink_blocks,
            "reserved_recent_blocks": reserved_recent_blocks,
            "use_compressed_on_miss": use_compressed_on_miss,
            "sparse_allowed_by_gate": bool(hit_infos[-1]["sparse_allowed_by_gate"]) if hit_infos else None,
            "cache_miss_total_latency_ms": float(median(miss_timings)),
            "cache_hit_total_latency_ms": float(median(hit_timings)),
            "cache_hit_execution_mode": hit_infos[-1]["execution_mode"] if hit_infos else None,
            "active_blocks": hit_infos[-1]["num_active_blocks"] if hit_infos else None,
            "kv_cache_hit": latest.kv_cache_hit if latest else None,
            "audit_cosine": latest.audit_cosine if latest else None,
            "quant_audit_cosine": latest.quant_audit_cosine if latest else None,
            "sparse_audit_cosine": latest.sparse_audit_cosine if latest else None,
            "sparse_audit_rel_mae": latest.sparse_audit_rel_mae if latest else None,
            "effective_sparsity": latest.effective_sparsity if latest else None,
        }


def main():
    print("=" * 60)
    print("RFSN v10 End-to-End Benchmarks")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    shape_q_decode = (1, 8, 1, 64)
    shape_q_dense = (1, 8, 8, 64)
    shape_kv = (1, 8, 2048, 64)
    configs = [
        ("cache_miss_full_precision_path", shape_q_decode, 8, 3, True, 0.50, False),
        ("cache_miss_use_compressed_on_miss_path", shape_q_decode, 8, 3, True, 0.50, True),
        ("cache_hit_compressed_path", shape_q_decode, 8, 3, True, 0.50, True),
        ("sparse_decode_path", shape_q_decode, 8, 3, True, 0.60, True),
        ("dense_decode_path", shape_q_dense, 8, 3, True, 0.25, True),
    ]

    results = {"metadata": meta, "runs": []}

    def fmt_metric(value):
        return "n/a" if value is None else f"{value:.4f}"

    for scenario, shape_q, k_bits, v_bits, use_inc, ratio, use_compressed_on_miss in configs:
        r = benchmark_e2e(
            shape_q,
            shape_kv,
            k_bits,
            v_bits,
            use_inc,
            ratio,
            use_compressed_on_miss=use_compressed_on_miss,
        )
        print(
            f"  {scenario}: "
            f"miss={r['cache_miss_total_latency_ms']:.2f}ms "
            f"hit={r['cache_hit_total_latency_ms']:.2f}ms "
            f"mode={r['cache_hit_execution_mode']} "
            f"quant_cos={fmt_metric(r['quant_audit_cosine'])} "
            f"sparse_cos={fmt_metric(r['sparse_audit_cosine'])}"
        )
        r["scenario"] = scenario
        results["runs"].append(r)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
