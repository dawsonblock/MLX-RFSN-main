#!/usr/bin/env python3
"""Per-layer sensitivity analysis for KV compression variants.

For each layer, tests:
  k8_v5_gs64, k8_v4_gs64, k8_v5_gs32, adaptive, experimental_hybrid

Metrics per layer:
  layer_id, cosine_drop, top5_drop, KL_delta, NLL_delta,
  attention_score_mae, attention_softmax_kl,
  recommended_k_bits, recommended_v_bits, recommended_group_size

Outputs:
  artifacts/proof/experimental/per_layer_sensitivity.json
  artifacts/proof/experimental/layer_policy.json

Usage:
  python benchmarks/per_layer_sensitivity.py \
      --layers 24 --shape 1 8 512 64 --repeats 3 \
      --out artifacts/proof/experimental/per_layer_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


# ---------------------------------------------------------------------------
# Attention helpers (synthetic, no model)
# ---------------------------------------------------------------------------

def _attention_scores(q: mx.array, k: mx.array) -> mx.array:
    """Raw attention scores [B,H,Tq,Tk]."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(float(d))
    return (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale


def _attention_out(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Standard attention output [B,H,Tq,D]."""
    scores = _attention_scores(q, k)
    probs = mx.softmax(scores, axis=-1)
    return probs @ v.astype(mx.float32)


def _cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return float((dot / mx.maximum(norm, mx.array(1e-8))).item())


def _kl_div_attention(p: mx.array, q: mx.array, eps: float = 1e-10) -> float:
    """KL(p || q) over last axis (attention distributions)."""
    p = p + eps
    q = q + eps
    kl = mx.sum(p * mx.log(p / q), axis=-1)
    return float(mx.mean(kl).item())


def _topk_overlap_weights(a: mx.array, b: mx.array, k: int = 5) -> float:
    """Top-k index overlap ratio between two 1D arrays."""
    a_idx = set(mx.argpartition(-a, kth=k - 1, axis=-1)[..., :k].tolist())
    b_idx = set(mx.argpartition(-b, kth=k - 1, axis=-1)[..., :k].tolist())
    if not a_idx:
        return 0.0
    return len(a_idx & b_idx) / len(a_idx)


# ---------------------------------------------------------------------------
# Compression helpers per variant
# ---------------------------------------------------------------------------

def _compress_stable(k: mx.array, v: mx.array, k_bits: int, v_bits: int, group_size: int):
    with tempfile.TemporaryDirectory(prefix="rfsn_sens_") as td:
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
        mgr.store("sens", k, v, token_count=k.shape[2])
        rec = mgr.retrieve("sens", out_dtype=mx.float32)
        if rec is None:
            raise RuntimeError("Cache miss")
        return rec[0], rec[1], mgr


def _compress_experimental(k: mx.array, v: mx.array, mode: str, adaptive: bool = False):
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
    rk, rv = mgr.dequantize(packet)
    return rk, rv, mgr


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _evaluate_variant(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rk: mx.array,
    rv: mx.array,
) -> dict[str, float]:
    baseline_out = _attention_out(q, k, v)
    compressed_out = _attention_out(q, rk, rv)

    cos = _cosine_similarity(baseline_out, compressed_out)
    cosine_drop = max(0.0, 1.0 - cos)

    base_scores = _attention_scores(q, k)
    comp_scores = _attention_scores(q, rk)
    attention_score_mae = float(mx.mean(mx.abs(base_scores - comp_scores)).item())

    base_probs = mx.softmax(base_scores, axis=-1)
    comp_probs = mx.softmax(comp_scores, axis=-1)
    attention_softmax_kl = _kl_div_attention(base_probs, comp_probs)

    # Top-5 overlap averaged over heads and query positions
    B, H, Tq, Tk = base_scores.shape
    top5_overlaps = []
    for b in range(B):
        for h in range(H):
            for t in range(Tq):
                ov = _topk_overlap_weights(base_scores[b, h, t], comp_scores[b, h, t], k=5)
                top5_overlaps.append(ov)
    top5_overlap = sum(top5_overlaps) / len(top5_overlaps) if top5_overlaps else 0.0
    top5_drop = max(0.0, 1.0 - top5_overlap)

    # KL_delta: average KL over query positions and heads
    kl_vals = []
    for b in range(B):
        for h in range(H):
            for t in range(Tq):
                kl_vals.append(_kl_div_attention(base_probs[b, h, t], comp_probs[b, h, t]))
    kl_delta = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0

    # NLL_delta: treat baseline probs as target; cross-entropy delta
    nll_base = -mx.sum(base_probs * mx.log(base_probs + 1e-10), axis=-1)
    nll_comp = -mx.sum(base_probs * mx.log(comp_probs + 1e-10), axis=-1)
    nll_delta = float(mx.mean(nll_comp - nll_base).item())

    return {
        "cosine_drop": cosine_drop,
        "top5_drop": top5_drop,
        "KL_delta": kl_delta,
        "NLL_delta": nll_delta,
        "attention_score_mae": attention_score_mae,
        "attention_softmax_kl": attention_softmax_kl,
    }


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _variant_specs() -> list[dict[str, Any]]:
    return [
        {"name": "k8_v5_gs64", "k_bits": 8, "v_bits": 5, "group_size": 64, "family": "stable"},
        {"name": "k8_v4_gs64", "k_bits": 8, "v_bits": 4, "group_size": 64, "family": "stable"},
        {"name": "k8_v5_gs32", "k_bits": 8, "v_bits": 5, "group_size": 32, "family": "stable"},
        {"name": "adaptive", "mode": "turbo_polar", "adaptive": True, "family": "experimental"},
        {"name": "experimental_hybrid", "mode": "hybrid_polar_cartesian", "adaptive": False, "family": "experimental"},
    ]


# ---------------------------------------------------------------------------
# Layer benchmarking
# ---------------------------------------------------------------------------

def benchmark_layer(
    layer_id: int,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    repeats: int = 3,
) -> dict[str, Any]:
    variants = _variant_specs()
    variant_results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_recommendation: dict[str, Any] = {}

    for spec in variants:
        metrics_runs: list[dict[str, float]] = []
        for _ in range(repeats):
            if spec["family"] == "stable":
                rk, rv, _mgr = _compress_stable(
                    k, v, spec["k_bits"], spec["v_bits"], spec["group_size"]
                )
            else:
                rk, rv, _mgr = _compress_experimental(
                    k, v, spec.get("mode", "turbo_polar"), spec.get("adaptive", False)
                )
            mx.eval(rk, rv)
            metrics_runs.append(_evaluate_variant(q, k, v, rk, rv))

        # Average metrics across repeats
        avg_metrics: dict[str, float] = {}
        keys = list(metrics_runs[0].keys())
        for key in keys:
            vals = [m[key] for m in metrics_runs]
            avg_metrics[key] = sum(vals) / len(vals)

        # Simple composite score: lower is better
        score = (
            avg_metrics["cosine_drop"] * 2.0
            + avg_metrics["top5_drop"]
            + avg_metrics["KL_delta"] * 10.0
            + avg_metrics["attention_softmax_kl"] * 10.0
        )

        variant_results.append({
            "variant": spec["name"],
            **avg_metrics,
            "composite_score": score,
        })

        if score < best_score:
            best_score = score
            if spec["family"] == "stable":
                best_recommendation = {
                    "recommended_k_bits": spec["k_bits"],
                    "recommended_v_bits": spec["v_bits"],
                    "recommended_group_size": spec["group_size"],
                    "recommended_variant": spec["name"],
                }
            else:
                best_recommendation = {
                    "recommended_k_bits": 6,
                    "recommended_v_bits": 6,
                    "recommended_group_size": 64,
                    "recommended_variant": spec["name"],
                }

    return {
        "layer_id": layer_id,
        "variants": variant_results,
        **best_recommendation,
    }


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

def generate_layer_policy(sensitivity_results: list[dict[str, Any]]) -> dict[str, Any]:
    max_layer = max(
        (row.get("layer_id", 0) for row in sensitivity_results), default=0
    )
    conservative_end = max(3, max_layer // 8)
    aggressive_start = max(16, max_layer - max_layer // 4)

    policy: dict[str, Any] = {
        "description": (
            f"layers 0-{conservative_end}: conservative precision; "
            f"layers {conservative_end + 1}-{aggressive_start - 1}: standard precision; "
            f"layers {aggressive_start}+: aggressive precision if safe"
        ),
        "default_fallback": {
            "k_bits": 8,
            "v_bits": 5,
            "group_size": 64,
            "variant": "k8_v5_gs64",
        },
        "layers": {},
    }

    for row in sensitivity_results:
        layer_id = row.get("layer_id")
        if layer_id is None:
            continue

        rec = {
            "k_bits": row.get("recommended_k_bits", 8),
            "v_bits": row.get("recommended_v_bits", 5),
            "group_size": row.get("recommended_group_size", 64),
            "variant": row.get("recommended_variant", "k8_v5_gs64"),
        }

        # Apply policy biases
        if layer_id <= conservative_end:
            rec["bias"] = "conservative"
            rec["k_bits"] = max(rec["k_bits"], 8)
            rec["v_bits"] = max(rec["v_bits"], 5)
            rec["group_size"] = min(rec.get("group_size", 64), 64)
        elif layer_id < aggressive_start:
            rec["bias"] = "standard"
        else:
            rec["bias"] = "aggressive"
            # Only allow aggressive if quality is good enough
            variants = row.get("variants", [])
            best_variant = next(
                (v for v in variants if v.get("variant") == rec["variant"]),
                None,
            )
            if best_variant is not None:
                if (
                    best_variant.get("cosine_drop", 1.0) < 0.005
                    and best_variant.get("KL_delta", 1.0) < 0.001
                ):
                    rec["k_bits"] = max(4, rec["k_bits"] - 2)
                    rec["v_bits"] = max(3, rec["v_bits"] - 2)
                else:
                    rec["note"] = "aggressive blocked by quality"
            else:
                rec["note"] = "variant not found; using standard"

        policy["layers"][str(layer_id)] = rec

    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-layer KV compression sensitivity")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[1, 8, 512, 64],
        help="Synthetic KV shape: B H T D",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per variant")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/proof/experimental/per_layer_sensitivity.json"),
        help="Sensitivity JSON output",
    )
    parser.add_argument(
        "--policy-out",
        type=Path,
        default=Path("artifacts/proof/experimental/layer_policy.json"),
        help="Policy JSON output",
    )
    args = parser.parse_args()

    B, H, T, D = args.shape
    print(f"Per-layer sensitivity: {args.layers} layers, shape ({B},{H},{T},{D})")

    mx.random.seed(42)

    sensitivity_results: list[dict[str, Any]] = []
    for layer_id in range(args.layers):
        # Layer-specific synthetic tensors (seeded for reproducibility but varied per layer)
        layer_seed = 42 + layer_id * 7
        mx.random.seed(layer_seed)
        q = mx.random.normal((B, H, 1, D))
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        mx.eval(q, k, v)

        print(f"  Layer {layer_id} ...", end=" ", flush=True)
        row = benchmark_layer(layer_id, q, k, v, repeats=args.repeats)
        sensitivity_results.append(row)
        print(
            f"best={row.get('recommended_variant', '?')} "
            f"k={row.get('recommended_k_bits', '?')}v={row.get('recommended_v_bits', '?')}"
        )

    policy = generate_layer_policy(sensitivity_results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"layers": sensitivity_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote sensitivity to {args.out}")

    args.policy_out.parent.mkdir(parents=True, exist_ok=True)
    args.policy_out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Wrote policy to {args.policy_out}")


if __name__ == "__main__":
    main()
