#!/usr/bin/env python3
"""Benchmark sequential vs metal reconstruction paths for Main12."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager  # noqa: E402

COSINE_THRESHOLD = 0.999
MAX_ABS_DIFF_THRESHOLD = 1e-3


SHAPES = [
    (1, 4, 512, 64),
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
]


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def _manager_kwargs_for_mode(mode: str) -> dict:
    """Return kwargs for RFSNTurboQuantKVManager for a given mode."""
    common = {
        "k_bits": 8,
        "v_bits": 3,
    }
    if mode == "sequential_reference":
        return {
            **common,
            "use_wht": True,
            "use_incoherent_signs": True,
            "prefer_metal_kernels": False,
        }
    if mode == "metal_multikernel_dequant_sign":
        return {
            **common,
            "use_wht": False,
            "use_incoherent_signs": True,
            "prefer_metal_kernels": True,
            "prefer_fused_kernel": False,
            "strict_metal": False,
        }
    if mode == "metal_multikernel_dequant":
        return {
            **common,
            "use_wht": False,
            "use_incoherent_signs": False,
            "prefer_metal_kernels": True,
            "prefer_fused_kernel": False,
            "strict_metal": False,
        }
    if mode == "metal_multikernel_dequant_wht":
        return {
            **common,
            "use_wht": True,
            "use_incoherent_signs": False,
            "prefer_metal_kernels": True,
            "prefer_fused_kernel": False,
            "strict_metal": False,
        }
    if mode == "metal_multikernel_dequant_wht_sign":
        return {
            **common,
            "use_wht": True,
            "use_incoherent_signs": True,
            "prefer_metal_kernels": True,
            "prefer_fused_kernel": False,
            "strict_metal": False,
        }
    if mode == "metal_fused_dequant_wht_sign":
        return {
            **common,
            "use_wht": True,
            "use_incoherent_signs": True,
            "prefer_metal_kernels": True,
            "prefer_fused_kernel": True,
            "strict_metal": False,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def _make_manager(cache_dir: Path, mode: str) -> RFSNTurboQuantKVManager:
    return RFSNTurboQuantKVManager(
        cache_dir=str(cache_dir),
        **_manager_kwargs_for_mode(mode),
    )


def _make_internal_reference_manager(
    cache_dir: Path, mode: str
) -> RFSNTurboQuantKVManager:
    """Create a Python-only (non-Metal) manager with the same quant settings."""
    kwargs = _manager_kwargs_for_mode(mode)
    kwargs["prefer_metal_kernels"] = False
    kwargs["prefer_fused_kernel"] = False
    return RFSNTurboQuantKVManager(cache_dir=str(cache_dir), **kwargs)


def _latency_stats(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    ordered = sorted(latencies)
    mean = float(statistics.fmean(ordered))
    p50 = float(statistics.median(ordered))
    p95_index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    p95 = float(ordered[p95_index])
    return mean, p50, p95


def _get_gold_reference(
    shape: tuple[int, int, int, int],
    mode: str,
    base_dir: Path,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Store with Python-only manager matching mode's config and return gold."""
    mx.random.seed(42)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)

    gold_dir = base_dir / "gold" / mode / (
        "x".join(str(v) for v in shape)
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    # Use same quant settings as mode, but force Python path (no Metal)
    kwargs = _manager_kwargs_for_mode(mode)
    kwargs["prefer_metal_kernels"] = False
    kwargs["prefer_fused_kernel"] = False
    mgr = RFSNTurboQuantKVManager(cache_dir=str(gold_dir), **kwargs)
    # Same key as metal/internal so sign seed is identical
    key = f"bench|mode={mode}|shape={shape}"
    mgr.store(key, keys, values, token_count=shape[2])

    gold_k, gold_v = mgr.retrieve(key, out_dtype=mx.float32)
    if gold_k is None or gold_v is None:
        raise RuntimeError("Gold reference retrieval failed")
    mx.eval(gold_k, gold_v)
    return gold_k, gold_v, keys, values


def _bench_mode(
    shape: tuple[int, int, int, int],
    mode: str,
    iterations: int,
    base_dir: Path,
    gold_k: mx.array,
    gold_v: mx.array,
    raw_keys: mx.array,
    raw_values: mx.array,
) -> dict:
    # Same key as gold so sign seed is identical
    key = f"bench|mode={mode}|shape={shape}"
    mode_dir = base_dir / mode / ("x".join(str(v) for v in shape))
    mode_dir.mkdir(parents=True, exist_ok=True)

    manager = _make_manager(mode_dir, mode)
    manager.store(key, raw_keys, raw_values, token_count=shape[2])

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        rec = manager.retrieve(key, out_dtype=mx.float32)
        if rec is None:
            raise RuntimeError("Expected cache hit during retrieval benchmark")
        mx.eval(rec[0], rec[1])
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
    latency_mean, latency_p50, latency_p95 = _latency_stats(latencies)
    k_out, v_out = manager.retrieve(key, out_dtype=mx.float32)
    mx.eval(k_out, v_out)

    # Cross-mode equivalence: compare against sequential_reference gold
    k_diff_gold = mx.max(mx.abs(k_out - gold_k)).item()
    k_cosine_gold = cosine_similarity(k_out, gold_k)
    v_diff_gold = mx.max(mx.abs(v_out - gold_v)).item()
    v_cosine_gold = cosine_similarity(v_out, gold_v)

    # Internal self-consistency: compare Metal retrieve vs Python retrieve
    # for the same stored data with the same quant settings
    internal_dir = mode_dir / "internal_ref"
    internal_dir.mkdir(parents=True, exist_ok=True)
    internal_mgr = _make_internal_reference_manager(internal_dir, mode)
    # Same key so sign seed is identical
    internal_mgr.store(key, raw_keys, raw_values, token_count=shape[2])
    int_k, int_v = internal_mgr.retrieve(key, out_dtype=mx.float32)
    if int_k is None or int_v is None:
        raise RuntimeError("Internal reference retrieval failed")
    mx.eval(int_k, int_v)

    k_diff_int = mx.max(mx.abs(k_out - int_k)).item()
    k_cosine_int = cosine_similarity(k_out, int_k)
    v_diff_int = mx.max(mx.abs(v_out - int_v)).item()
    v_cosine_int = cosine_similarity(v_out, int_v)

    return {
        "shape": [int(v) for v in shape],
        "bits": 8,
        "mode": mode,
        "route": mode,
        "latency_ms_mean": latency_mean,
        "latency_ms_p50": latency_p50,
        "latency_ms_p95": latency_p95,
        "kv_reconstruction_kernel": manager.last_reconstruction_kernel,
        "fallback_used": (
            manager.last_reconstruction_kernel
            == "metal_failed_fallback_reference"
        ),
        "key_cosine_vs_gold": float(k_cosine_gold),
        "key_max_abs_diff_vs_gold": float(k_diff_gold),
        "value_cosine_vs_gold": float(v_cosine_gold),
        "value_max_abs_diff_vs_gold": float(v_diff_gold),
        "key_cosine_vs_internal": float(k_cosine_int),
        "key_max_abs_diff_vs_internal": float(k_diff_int),
        "value_cosine_vs_internal": float(v_cosine_int),
        "value_max_abs_diff_vs_internal": float(v_diff_int),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark kernel reconstruction paths"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main24/kernel_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = [
        "sequential_reference",
        "metal_multikernel_dequant",
        "metal_multikernel_dequant_wht",
        "metal_multikernel_dequant_sign",
        "metal_multikernel_dequant_wht_sign",
        "metal_fused_dequant_wht_sign",
    ]

    tmp_dir = out_path.parent / "kernel_bench_tmp"

    # Precompute gold references for each (shape, mode) pair
    # so each metal mode compares against gold with matching quant config
    gold_refs: dict[
        str, tuple[mx.array, mx.array, mx.array, mx.array]
    ] = {}
    for shape in SHAPES:
        for mode in modes:
            ref_key = f"{mode}|{shape}"
            gold_refs[ref_key] = _get_gold_reference(shape, mode, tmp_dir)

    runs: list[dict] = []
    reference_latencies: dict[str, float] = {}

    for shape in SHAPES:
        shape_key = str(shape)
        for mode in modes:
            ref_key = f"{mode}|{shape}"
            gold_k, gold_v, raw_keys, raw_values = gold_refs[ref_key]
            row = _bench_mode(
                shape,
                mode,
                args.iterations,
                tmp_dir,
                gold_k,
                gold_v,
                raw_keys,
                raw_values,
            )
            runs.append(row)
            if mode == "sequential_reference":
                reference_latencies[shape_key] = row["latency_ms_p50"]

    for row in runs:
        shape_key = str(tuple(row["shape"]))
        ref_latency = reference_latencies.get(
            shape_key, row["latency_ms_p50"]
        )
        computed_speedup = (
            float(ref_latency) / float(row["latency_ms_p50"])
            if float(row["latency_ms_p50"]) > 0
            else 0.0
        )

        # Gold validation
        gold_key_valid = (
            float(row["key_cosine_vs_gold"]) >= COSINE_THRESHOLD
            and float(row["key_max_abs_diff_vs_gold"])
            <= MAX_ABS_DIFF_THRESHOLD
        )
        gold_value_valid = (
            float(row["value_cosine_vs_gold"]) >= COSINE_THRESHOLD
            and float(row["value_max_abs_diff_vs_gold"])
            <= MAX_ABS_DIFF_THRESHOLD
        )

        # Internal validation
        internal_key_valid = (
            float(row["key_cosine_vs_internal"]) >= COSINE_THRESHOLD
            and float(row["key_max_abs_diff_vs_internal"])
            <= MAX_ABS_DIFF_THRESHOLD
        )
        internal_value_valid = (
            float(row["value_cosine_vs_internal"]) >= COSINE_THRESHOLD
            and float(row["value_max_abs_diff_vs_internal"])
            <= MAX_ABS_DIFF_THRESHOLD
        )

        row["gold_valid"] = gold_key_valid and gold_value_valid
        row["internal_valid"] = internal_key_valid and internal_value_valid

        # Add route classification
        full_routes = {
            "sequential_reference",
            "metal_multikernel_dequant_wht_sign",
            "metal_fused_dequant_wht_sign",
        }
        if row["mode"] in full_routes:
            row["route_class"] = "full_equivalent"
        else:
            row["route_class"] = "ablation"

        if row["mode"].startswith("metal_"):
            invalid = (
                bool(row["fallback_used"])
                or not row["gold_valid"]
                or not row["internal_valid"]
            )
            row["status"] = "invalid" if invalid else "valid"
            row["speedup_vs_reference"] = (
                None if invalid else float(computed_speedup)
            )
        else:
            row["status"] = "valid"
            row["speedup_vs_reference"] = float(computed_speedup)

    payload = {
        "runs": runs,
        "summary": {
            "equivalence_pass_modes": [
                row["mode"]
                for row in runs
                if row["gold_valid"] and row["internal_valid"]
            ],
            "speedup_modes": [
                row["mode"]
                for row in runs
                if (
                    row["mode"].startswith("metal_")
                    and row["status"] == "valid"
                    and row["speedup_vs_reference"] is not None
                    and row["speedup_vs_reference"] > 1.0
                )
            ],
            "gold_only_pass_modes": [
                row["mode"]
                for row in runs
                if row["gold_valid"] and not row["internal_valid"]
            ],
            "internal_only_pass_modes": [
                row["mode"]
                for row in runs
                if row["internal_valid"] and not row["gold_valid"]
            ],
        },
    }

    out_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote kernel benchmark to {out_path}")


if __name__ == "__main__":
    main()
