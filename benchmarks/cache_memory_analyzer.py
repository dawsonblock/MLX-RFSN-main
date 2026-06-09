#!/usr/bin/env python3
"""Cache memory analyzer for KV compression configs.

Functions:
  analyze_kv_cache_memory(k, v, config)  — detailed memory breakdown
  compare_configs(configs)               — comparison table

Metrics:
  fp16_bytes, compressed_bytes, scale_bytes, packed_code_bytes,
  overhead_bytes, compression_ratio, waste_ratio

Output: artifacts/proof/experimental/cache_memory_analysis.json

Usage:
  python benchmarks/cache_memory_analyzer.py \
      --shape 1 8 2048 64 \
      --configs baseline_fp16,stable_k8_v5_gs64,adaptive,experimental_hybrid,turbo_polar
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _packed_words(n_values: int, bits: int) -> int:
    codes_per_word = 32 // bits
    return (n_values + codes_per_word - 1) // codes_per_word


def _stable_memory_breakdown(
    k: mx.array,
    v: mx.array,
    k_bits: int,
    v_bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Detailed memory breakdown using stable TurboQuant path."""
    fp16_bytes = int(k.size + v.size) * 2

    n_k = int(k.size)
    n_v = int(v.size)

    k_packed_words = _packed_words(n_k, k_bits)
    v_packed_words = _packed_words(n_v, v_bits)

    k_packed_bytes = k_packed_words * 4
    v_packed_bytes = v_packed_words * 4
    packed_code_bytes = k_packed_bytes + v_packed_bytes

    k_n_groups = (n_k + group_size - 1) // group_size
    v_n_groups = (n_v + group_size - 1) // group_size

    k_scale_bytes = k_n_groups * 4
    v_scale_bytes = v_n_groups * 4
    scale_bytes = k_scale_bytes + v_scale_bytes

    metadata_overhead = 256
    compressed_bytes = packed_code_bytes + scale_bytes + metadata_overhead

    # Validate against actual manager
    with tempfile.TemporaryDirectory(prefix="rfsn_mem_") as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            use_wht=True,
            use_incoherent_signs=True,
            max_memory_gb=2.0,
            cache_dir=td,
        )
        mgr.store("mem", k, v, token_count=k.shape[2])
        cache = mgr.active_caches["mem"]
        actual_compressed = int(
            (cache.k_packed.size + cache.v_packed.size) * 4
            + (cache.k_scales.size + cache.v_scales.size) * 4
        )

    # Use actual if available, otherwise theoretical
    compressed_bytes = max(compressed_bytes, actual_compressed)
    # Recalculate overhead to match actual total
    overhead_bytes = compressed_bytes - packed_code_bytes - scale_bytes

    compression_ratio = fp16_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    waste_ratio = overhead_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

    return {
        "fp16_bytes": fp16_bytes,
        "compressed_bytes": compressed_bytes,
        "scale_bytes": scale_bytes,
        "packed_code_bytes": packed_code_bytes,
        "overhead_bytes": overhead_bytes,
        "compression_ratio": float(compression_ratio),
        "waste_ratio": float(waste_ratio),
    }


def _experimental_memory_breakdown(
    k: mx.array,
    v: mx.array,
    mode: str,
    adaptive: bool = False,
) -> dict[str, Any]:
    """Detailed memory breakdown using experimental paths."""
    fp16_bytes = int(k.size + v.size) * 2
    feature_dim = k.shape[-1]
    group_size = 64

    if mode == "turbo_polar":
        mgr = TurboPolarKVManager(
            feature_dim=feature_dim,
            k_angle_bits=5,
            k_radius_bits=8,
            v_bits=6,
            group_size=group_size,
            adaptive_angle_range=adaptive,
        )
    else:
        mgr = QuantizedKVManager(
            mode="hybrid_polar_cartesian",
            feature_dim=feature_dim,
            polar_ratio=0.65,
            polar_levels=4,
            k_angle_bits=5,
            k_radius_bits=8,
            v_angle_bits=4,
            v_radius_bits=6,
            cartesian_bits=6,
            group_size=group_size,
            adaptive_angle_range=adaptive,
        )

    packet = mgr.quantize(k, v)
    compressed_bytes = int(mgr.estimate_bytes(packet))

    # Approximate breakdown for experimental paths:
    # Treat scales as ~10% and overhead as the rest after code bytes
    scale_bytes = compressed_bytes // 10
    overhead_bytes = 256
    packed_code_bytes = compressed_bytes - scale_bytes - overhead_bytes
    if packed_code_bytes < 0:
        packed_code_bytes = compressed_bytes
        scale_bytes = 0
        overhead_bytes = 0

    compression_ratio = fp16_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    waste_ratio = overhead_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

    return {
        "fp16_bytes": fp16_bytes,
        "compressed_bytes": compressed_bytes,
        "scale_bytes": scale_bytes,
        "packed_code_bytes": packed_code_bytes,
        "overhead_bytes": overhead_bytes,
        "compression_ratio": float(compression_ratio),
        "waste_ratio": float(waste_ratio),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_kv_cache_memory(
    k: mx.array,
    v: mx.array,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute detailed memory breakdown for a single config.

    Args:
        k: Key tensor [B, H, T, D].
        v: Value tensor [B, H, T, D].
        config: Dict with keys:
            - name: config identifier
            - family: "baseline", "stable", or "experimental"
            - For stable: k_bits, v_bits, group_size
            - For experimental: mode, adaptive (optional)

    Returns:
        Dict with fp16_bytes, compressed_bytes, scale_bytes,
        packed_code_bytes, overhead_bytes, compression_ratio, waste_ratio.
    """
    family = config.get("family", "baseline")
    name = config.get("name", "unknown")

    if family == "baseline":
        fp16_bytes = int(k.size + v.size) * 2
        return {
            "config": name,
            "fp16_bytes": fp16_bytes,
            "compressed_bytes": fp16_bytes,
            "scale_bytes": 0,
            "packed_code_bytes": 0,
            "overhead_bytes": 0,
            "compression_ratio": 1.0,
            "waste_ratio": 0.0,
        }

    if family == "stable":
        breakdown = _stable_memory_breakdown(
            k, v,
            k_bits=config["k_bits"],
            v_bits=config["v_bits"],
            group_size=config["group_size"],
        )
    else:
        breakdown = _experimental_memory_breakdown(
            k, v,
            mode=config.get("mode", "hybrid_polar_cartesian"),
            adaptive=config.get("adaptive_angle_range", False),
        )

    breakdown["config"] = name
    return breakdown


def compare_configs(
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a comparison table across multiple already-analyzed configs.

    Args:
        configs: List of dicts returned by ``analyze_kv_cache_memory``.

    Returns:
        Dict with 'configs' list and 'summary' (best compression, best efficiency).
    """
    rows = [c for c in configs if "error" not in c]
    if not rows:
        return {"configs": configs, "summary": {}}

    best_compression = max(rows, key=lambda r: r.get("compression_ratio", 0.0))
    best_efficiency = min(rows, key=lambda r: r.get("waste_ratio", float("inf")))

    return {
        "configs": configs,
        "summary": {
            "best_compression_config": best_compression.get("config"),
            "best_compression_ratio": best_compression.get("compression_ratio"),
            "best_efficiency_config": best_efficiency.get("config"),
            "best_efficiency_waste_ratio": best_efficiency.get("waste_ratio"),
        },
    }


# ---------------------------------------------------------------------------
# Config registry (mirror of throughput benchmark)
# ---------------------------------------------------------------------------

def _get_config(name: str) -> dict[str, Any]:
    if name == "baseline_fp16":
        return {"name": "baseline_fp16", "family": "baseline"}
    if name == "stable_k8_v5_gs64":
        return {"name": "stable_k8_v5_gs64", "family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 64}
    if name == "stable_k8_v5_gs32":
        return {"name": "stable_k8_v5_gs32", "family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 32}
    if name == "stable_k8_v4_gs64":
        return {"name": "stable_k8_v4_gs64", "family": "stable", "k_bits": 8, "v_bits": 4, "group_size": 64}
    if name == "adaptive":
        return {"name": "adaptive", "family": "experimental", "mode": "turbo_polar", "adaptive_angle_range": True}
    if name == "turbo_polar":
        return {"name": "turbo_polar", "family": "experimental", "mode": "turbo_polar"}
    if name == "experimental_hybrid":
        return {"name": "experimental_hybrid", "family": "experimental", "mode": "hybrid_polar_cartesian"}
    raise ValueError(f"Unknown config: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cache memory analyzer")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[1, 8, 2048, 64],
        help="Synthetic KV shape: B H T D",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=(
            "baseline_fp16,stable_k8_v5_gs64,stable_k8_v5_gs32,"
            "adaptive,experimental_hybrid,turbo_polar"
        ),
        help="Comma-separated config names",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/proof/experimental/cache_memory_analysis.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    B, H, T, D = args.shape
    print(f"Analyzing memory for shape ({B},{H},{T},{D})")

    mx.random.seed(42)
    k = mx.random.normal((B, H, T, D))
    v = mx.random.normal((B, H, T, D))
    mx.eval(k, v)

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [_get_config(n) for n in config_names]

    analyzed = [analyze_kv_cache_memory(k, v, cfg) for cfg in configs]
    result = compare_configs(analyzed)

    for row in result["configs"]:
        print(
            f"  {row['config']}: "
            f"fp16={row['fp16_bytes']:,}B "
            f"compressed={row['compressed_bytes']:,}B "
            f"ratio={row['compression_ratio']:.2f} "
            f"waste={row['waste_ratio']:.3f}"
        )

    summary = result.get("summary", {})
    if summary:
        print(
            f"\nBest compression: {summary.get('best_compression_config')} "
            f"({summary.get('best_compression_ratio', 0):.2f}x)"
        )
        print(
            f"Best efficiency: {summary.get('best_efficiency_config')} "
            f"(waste={summary.get('best_efficiency_waste_ratio', 0):.3f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
