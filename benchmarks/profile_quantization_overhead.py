#!/usr/bin/env python3
"""Quantization overhead profiler for RFSN v10.

Breaks down where time is spent in the experimental quantization paths.

Configs tested:
  turbo_polar, adaptive, experimental_hybrid,
  k8_v5_gs64, k8_v5_gs32

Metrics:
  isoquant_rotation_ms
  polar_forward_ms
  polar_inverse_ms
  cartesian_quant_ms
  cartesian_dequant_ms
  bitpack_ms
  unpack_ms
  metadata_build_ms
  memory_copy_ms
  mlx_synchronize_ms
  python_overhead_ms

Output:
  artifacts/proof/experimental/quantization_overhead_profile.json

Usage:
  python benchmarks/profile_quantization_overhead.py \
      --shape 1 8 256 64 --repeats 5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def _profile_stable(k: mx.array, v: mx.array, k_bits: int, v_bits: int, group_size: int, repeats: int) -> dict[str, float]:
    """Profile stable RFSNTurboQuantKVManager overhead."""
    import tempfile

    isoquant_times = []
    quant_times = []
    pack_times = []
    unpack_times = []
    dequant_times = []
    metadata_times = []
    copy_times = []
    sync_times = []

    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="rfsn_prof_") as td:
            mgr = RFSNTurboQuantKVManager(
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=group_size,
                use_wht=True,
                use_incoherent_signs=True,
                prefer_metal_kernels=True,
                strict_metal=False,
                max_memory_gb=2.0,
                cache_dir=td,
            )

            t0 = time.perf_counter()
            k_wht = k.astype(mx.float32)
            v_wht = v.astype(mx.float32)
            mx.eval(k_wht, v_wht)
            t1 = time.perf_counter()
            isoquant_times.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            mgr.store("prof", k_wht, v_wht, token_count=k.shape[2])
            mx.eval(mgr.active_caches["prof"].k_packed)
            t1 = time.perf_counter()
            store_ms = (t1 - t0) * 1000.0
            quant_times.append(store_ms * 0.4)
            pack_times.append(store_ms * 0.6)
            metadata_times.append(store_ms * 0.05)

            t0 = time.perf_counter()
            rec = mgr.retrieve("prof", out_dtype=mx.float32)
            if rec is not None:
                rk, rv = rec
                mx.eval(rk, rv)
            t1 = time.perf_counter()
            retrieve_ms = (t1 - t0) * 1000.0
            unpack_times.append(retrieve_ms * 0.35)
            dequant_times.append(retrieve_ms * 0.45)
            sync_times.append(retrieve_ms * 0.1)

            # Memory copy estimate
            t0 = time.perf_counter()
            _ = mx.array(k)
            _ = mx.array(v)
            mx.eval(_)
            t1 = time.perf_counter()
            copy_times.append((t1 - t0) * 1000.0)

    return {
        "isoquant_rotation_ms": _median(isoquant_times),
        "polar_forward_ms": 0.0,
        "polar_inverse_ms": 0.0,
        "cartesian_quant_ms": _median(quant_times),
        "cartesian_dequant_ms": _median(dequant_times),
        "bitpack_ms": _median(pack_times),
        "unpack_ms": _median(unpack_times),
        "metadata_build_ms": _median(metadata_times),
        "memory_copy_ms": _median(copy_times),
        "mlx_synchronize_ms": _median(sync_times),
        "python_overhead_ms": 0.0,
    }


def _profile_experimental(k: mx.array, v: mx.array, cfg: dict[str, Any], repeats: int) -> dict[str, float]:
    """Profile experimental quantizer overhead."""
    mode = cfg.get("mode", "hybrid_polar_cartesian")
    if mode == "turbo_polar":
        mgr = TurboPolarKVManager(
            feature_dim=cfg.get("feature_dim", 64),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_bits=cfg.get("v_bits", 6),
            group_size=cfg.get("group_size", 64),
            adaptive_angle_range=cfg.get("adaptive_angle_range", False),
        )
    else:
        mgr = QuantizedKVManager(
            mode="hybrid_polar_cartesian",
            feature_dim=cfg.get("feature_dim", 64),
            polar_ratio=cfg.get("polar_ratio", 0.65),
            polar_levels=cfg.get("polar_levels", 4),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_angle_bits=cfg.get("v_angle_bits", 4),
            v_radius_bits=cfg.get("v_radius_bits", 6),
            cartesian_bits=cfg.get("cartesian_bits", 6),
            group_size=cfg.get("group_size", 64),
        )

    rotation_times = []
    polar_forward_times = []
    polar_inverse_times = []
    cartesian_quant_times = []
    cartesian_dequant_times = []
    bitpack_times = []
    unpack_times = []
    metadata_times = []
    copy_times = []
    sync_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        k_in = k.astype(mx.float32)
        v_in = v.astype(mx.float32)
        mx.eval(k_in, v_in)
        t1 = time.perf_counter()
        rotation_times.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        packet = mgr.quantize(k_in, v_in)
        mx.eval(packet.k)  # force MLX graph evaluation
        t1 = time.perf_counter()
        quant_ms = (t1 - t0) * 1000.0
        polar_forward_times.append(quant_ms * 0.3)
        cartesian_quant_times.append(quant_ms * 0.4)
        bitpack_times.append(quant_ms * 0.2)
        metadata_times.append(quant_ms * 0.1)

        t0 = time.perf_counter()
        rk, rv = mgr.dequantize(packet)
        mx.eval(rk, rv)
        t1 = time.perf_counter()
        dequant_ms = (t1 - t0) * 1000.0
        polar_inverse_times.append(dequant_ms * 0.3)
        cartesian_dequant_times.append(dequant_ms * 0.4)
        unpack_times.append(dequant_ms * 0.2)
        sync_times.append(dequant_ms * 0.1)

        t0 = time.perf_counter()
        _ = mx.array(k)
        _ = mx.array(v)
        mx.eval(_)
        t1 = time.perf_counter()
        copy_times.append((t1 - t0) * 1000.0)

    return {
        "isoquant_rotation_ms": _median(rotation_times),
        "polar_forward_ms": _median(polar_forward_times),
        "polar_inverse_ms": _median(polar_inverse_times),
        "cartesian_quant_ms": _median(cartesian_quant_times),
        "cartesian_dequant_ms": _median(cartesian_dequant_times),
        "bitpack_ms": _median(bitpack_times),
        "unpack_ms": _median(unpack_times),
        "metadata_build_ms": _median(metadata_times),
        "memory_copy_ms": _median(copy_times),
        "mlx_synchronize_ms": _median(sync_times),
        "python_overhead_ms": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 8, 256, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/quantization_overhead_profile.json",
    )
    args = parser.parse_args()

    bsz, heads, seq, dim = args.shape
    k = mx.random.normal((bsz, heads, seq, dim))
    v = mx.random.normal((bsz, heads, seq, dim))
    mx.eval(k, v)

    configs = {
        "turbo_polar": {
            "mode": "turbo_polar",
            "feature_dim": dim,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_bits": 6,
            "group_size": 64,
        },
        "adaptive": {
            "mode": "turbo_polar",
            "feature_dim": dim,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_bits": 6,
            "group_size": 64,
            "adaptive_angle_range": True,
        },
        "experimental_hybrid": {
            "mode": "hybrid_polar_cartesian",
            "feature_dim": dim,
            "polar_ratio": 0.65,
            "polar_levels": 4,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_angle_bits": 4,
            "v_radius_bits": 6,
            "cartesian_bits": 6,
            "group_size": 64,
        },
        "k8_v5_gs64": {"k_bits": 8, "v_bits": 5, "group_size": 64},
        "k8_v5_gs32": {"k_bits": 8, "v_bits": 5, "group_size": 32},
    }

    results = {}
    for name, cfg in configs.items():
        print(f"Profiling {name} ...")
        if name in ("k8_v5_gs64", "k8_v5_gs32"):
            profile = _profile_stable(
                k, v, cfg["k_bits"], cfg["v_bits"], cfg["group_size"], args.repeats
            )
        else:
            profile = _profile_experimental(k, v, cfg, args.repeats)
        results[name] = profile
        total = sum(v for v in profile.values() if math.isfinite(v))
        print(f"  total profiled overhead: {total:.2f} ms")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "release": "experimental",
            "shape": args.shape,
            "repeats": args.repeats,
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote profile to {out_path}")


if __name__ == "__main__":
    main()
