#!/usr/bin/env python3
"""RFSN v10 — unified benchmark runner.

Runs the full deterministic benchmark suite and writes results to
``benchmarks/results/run_all_<timestamp>.json``.

Usage:
    python benchmarks/run_all.py              # run all suites
    python benchmarks/run_all.py --fast       # skip slow suites (attention only)
    python benchmarks/run_all.py --check      # compare against baseline and exit non-zero on regression
    python benchmarks/run_all.py --out PATH   # override output path

Exit codes:
    0  — all suites completed (or --check passed)
    1  — at least one suite failed or regression detected
    2  — missing required dependency (e.g. MLX not available on this platform)

NOTE: Some suites require MLX (Apple Silicon only).  On Linux / NumPy-only
CI the runner will skip MLX suites and report their status as 'skipped'.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None
    _HAS_MLX = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_PATH = Path(__file__).parent / "production_baseline.json"

WARMUP = 3
ITERATIONS = 10

# Regression tolerance: alert if a metric is > X% worse than baseline
REGRESSION_TOLERANCE = 0.10  # 10 %


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _timed_mlx(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms) using MLX eval for sync."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    mx.eval(out)
    return out, (time.perf_counter() - start) * 1000


def _timed_cpu(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - start) * 1000


def _ok(label: str, data: dict) -> dict:
    return {"suite": label, "status": "ok", "data": data}


def _skipped(label: str, reason: str) -> dict:
    return {"suite": label, "status": "skipped", "reason": reason}


def _failed(label: str, exc: Exception) -> dict:
    return {
        "suite": label,
        "status": "failed",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


# ---------------------------------------------------------------------------
# Suite: attention (MLX)
# ---------------------------------------------------------------------------

def suite_attention_mlx() -> dict:
    if not _HAS_MLX:
        return _skipped("attention_mlx", "MLX not available")

    from rfsn_v10.attention import AdaptiveBlockSparseAttention

    try:
        mx.random.seed(42)
        results = {}
        for t_k in [256, 512, 1024]:
            q = mx.random.normal((1, 8, 1, 64))
            k = mx.random.normal((1, 8, t_k, 64))
            v = mx.random.normal((1, 8, t_k, 64))
            mx.eval(q, k, v)

            for _ in range(WARMUP):
                out, _, _ = AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.5)
                mx.eval(out)

            sparse_ms, dense_ms = [], []
            for _ in range(ITERATIONS):
                _, ms = _timed_mlx(
                    lambda: AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.5)[0]
                )
                sparse_ms.append(ms)
                _, ms = _timed_mlx(
                    lambda: AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=1.0)[0]
                )
                dense_ms.append(ms)

            results[f"t_k={t_k}"] = {
                "sparse_ms": median(sparse_ms),
                "dense_ms": median(dense_ms),
                "speedup": median(dense_ms) / max(median(sparse_ms), 1e-6),
            }

        return _ok("attention_mlx", results)
    except Exception as exc:
        return _failed("attention_mlx", exc)


# ---------------------------------------------------------------------------
# Suite: quantized decode (MLX)
# ---------------------------------------------------------------------------

def suite_quant_decode_mlx() -> dict:
    if not _HAS_MLX:
        return _skipped("quant_decode_mlx", "MLX not available")

    from rfsn_v10.kernels import backend

    try:
        results = {}
        for n_keys in [1024, 4096, 16384]:
            n_h, d_head, bits, gs = 8, 64, 4, 64
            n_values = n_h * n_keys * d_head
            cpw = 32 // bits
            n_words = (n_values + cpw - 1) // cpw
            n_scales = (n_values + gs - 1) // gs
            packed_k = mx.zeros((n_words,), dtype=mx.uint32)
            packed_v = mx.zeros((n_words,), dtype=mx.uint32)
            scales_k = mx.ones((n_scales,), dtype=mx.float32)
            scales_v = mx.ones((n_scales,), dtype=mx.float32)
            queries = mx.ones((n_h, d_head), dtype=mx.float32)
            mx.eval(packed_k, packed_v, scales_k, scales_v, queries)

            for _ in range(WARMUP):
                out = backend.quantized_attention_decode(
                    queries, packed_k, packed_v, scales_k, scales_v,
                    n_keys=n_keys, bits=bits, group_size=gs,
                )
                mx.eval(out)

            times = []
            for _ in range(ITERATIONS):
                _, ms = _timed_mlx(
                    backend.quantized_attention_decode,
                    queries, packed_k, packed_v, scales_k, scales_v,
                    n_keys=n_keys, bits=bits, group_size=gs,
                )
                times.append(ms)

            results[f"n_keys={n_keys}"] = {"decode_ms": median(times)}

        return _ok("quant_decode_mlx", results)
    except Exception as exc:
        return _failed("quant_decode_mlx", exc)


# ---------------------------------------------------------------------------
# Suite: bitpack (NumPy, platform-agnostic)
# ---------------------------------------------------------------------------

def suite_bitpack_numpy() -> dict:
    if not _HAS_NUMPY:
        return _skipped("bitpack_numpy", "NumPy not available")

    try:
        # Import the numpy-specific backend directly to avoid MLX dispatch
        from rfsn_v10.kernels._numpy_backend import NumpyBackend
        nb = NumpyBackend()

        results = {}
        for n in [4096, 65536, 262144]:
            data = np.random.default_rng(0).integers(0, 15, n, dtype=np.uint32)

            for _ in range(WARMUP):
                packed, count = nb.pack_bits(data, bits=4)
                _ = nb.unpack_bits(packed, count, bits=4)

            times = []
            for _ in range(ITERATIONS):
                _, ms = _timed_cpu(nb.pack_bits, data, bits=4)
                times.append(ms)

            results[f"n={n}"] = {"pack_ms": median(times)}

        return _ok("bitpack_numpy", results)
    except Exception as exc:
        return _failed("bitpack_numpy", exc)


# ---------------------------------------------------------------------------
# Suite: KV roundtrip cosine (MLX)
# ---------------------------------------------------------------------------

def suite_kv_roundtrip_mlx() -> dict:
    if not _HAS_MLX:
        return _skipped("kv_roundtrip_mlx", "MLX not available")

    try:
        import tempfile
        from rfsn_v10.kv_manager import RFSNTurboQuantKVManager

        rng = np.random.default_rng(99) if _HAS_NUMPY else None
        results = {}
        for seq_len, gs in [(128, 32), (512, 32), (512, 64)]:
            if rng is not None:
                k_np = rng.standard_normal((1, 4, seq_len, 64)).astype(np.float32) * 0.1
                v_np = rng.standard_normal((1, 4, seq_len, 64)).astype(np.float32) * 0.1
            else:
                k_np = [[[[0.1] * 64] * seq_len] * 4]
                v_np = [[[[0.1] * 64] * seq_len] * 4]

            k = mx.array(k_np)
            v = mx.array(v_np)

            with tempfile.TemporaryDirectory(prefix="rfsn_bench_") as tmpdir:
                mgr = RFSNTurboQuantKVManager(
                    k_bits=8, v_bits=5, group_size=gs,
                    use_wht=True, use_incoherent_signs=True,
                    prefer_metal_kernels=True, strict_metal=False,
                    max_memory_gb=1.0, cache_dir=tmpdir,
                )
                t0 = time.perf_counter()
                mgr.store("k", k, v, token_count=seq_len)
                rec = mgr.retrieve("k", out_dtype=mx.float32)
                elapsed_ms = (time.perf_counter() - t0) * 1000

            rk, rv = rec
            k_np2 = np.array(rk.tolist(), dtype=np.float64).ravel()
            k_np0 = np.array(k.tolist(), dtype=np.float64).ravel()
            denom = np.linalg.norm(k_np0) * np.linalg.norm(k_np2)
            cosine = float(np.dot(k_np0, k_np2) / denom) if denom > 1e-12 else 1.0

            label = f"seq={seq_len}_gs={gs}"
            results[label] = {
                "cosine_K": round(cosine, 6),
                "roundtrip_ms": round(elapsed_ms, 3),
            }

        return _ok("kv_roundtrip_mlx", results)
    except Exception as exc:
        return _failed("kv_roundtrip_mlx", exc)


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def check_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float = REGRESSION_TOLERANCE,
) -> list[str]:
    """Return list of regression strings (empty = no regression)."""
    regressions = []
    for suite_name, suite_result in current.items():
        if suite_name in ("metadata", "timestamp"):
            continue
        if not isinstance(suite_result, dict) or suite_result.get("status") != "ok":
            continue
        base_suite = baseline.get(suite_name)
        if not base_suite or base_suite.get("status") != "ok":
            continue
        cur_data = suite_result.get("data", {})
        base_data = base_suite.get("data", {})
        for scenario, cur_metrics in cur_data.items():
            base_metrics = base_data.get(scenario, {})
            for metric, cur_val in cur_metrics.items():
                if not isinstance(cur_val, (int, float)):
                    continue
                base_val = base_metrics.get(metric)
                if base_val is None or not isinstance(base_val, (int, float)):
                    continue
                # For latency metrics (lower is better): flag if cur > base * (1+tol)
                if "ms" in metric:
                    if cur_val > base_val * (1 + tolerance):
                        regressions.append(
                            f"{suite_name}/{scenario}/{metric}: "
                            f"{cur_val:.3f} > baseline {base_val:.3f} (+{tolerance*100:.0f}%)"
                        )
                # For quality metrics (higher is better): flag if cur < base * (1-tol)
                elif metric in ("speedup", "cosine_K"):
                    if cur_val < base_val * (1 - tolerance):
                        regressions.append(
                            f"{suite_name}/{scenario}/{metric}: "
                            f"{cur_val:.4f} < baseline {base_val:.4f} (-{tolerance*100:.0f}%)"
                        )
    return regressions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFSN v10 benchmark runner")
    parser.add_argument("--fast", action="store_true", help="Skip slow suites")
    parser.add_argument(
        "--check", action="store_true",
        help="Compare against production_baseline.json and exit non-zero on regression"
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    suites = [
        suite_bitpack_numpy,
        suite_attention_mlx,
    ]
    if not args.fast:
        suites += [
            suite_quant_decode_mlx,
            suite_kv_roundtrip_mlx,
        ]

    all_results: dict[str, Any] = {
        "metadata": {
            "hardware": platform.machine(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "mlx_available": _HAS_MLX,
            "timestamp": timestamp,
            "fast_mode": args.fast,
        }
    }

    any_failed = False
    for suite_fn in suites:
        label = suite_fn.__name__.replace("suite_", "")
        print(f"  running {label} ...", end=" ", flush=True)
        result = suite_fn()
        all_results[label] = result
        status = result.get("status", "unknown")
        print(status)
        if status == "failed":
            any_failed = True
            print(f"    ERROR: {result.get('error', '')}")

    out_path = args.out or RESULTS_DIR / f"run_all_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Also update latest.json for check_regression.py compatibility
    latest = RESULTS_DIR / "latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    if args.check:
        if not BASELINE_PATH.exists():
            print(f"WARNING: No baseline at {BASELINE_PATH}; skipping regression check.")
        else:
            with open(BASELINE_PATH, encoding="utf-8") as f:
                baseline = json.load(f)
            regressions = check_regression(all_results, baseline)
            if regressions:
                print(f"\nREGRESSION DETECTED ({len(regressions)} item(s)):")
                for r in regressions:
                    print(f"  - {r}")
                return 1
            else:
                print("\nNo regressions detected.")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
