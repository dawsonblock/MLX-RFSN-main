#!/usr/bin/env python3
"""
QJL Attention-Score Validation.

Benchmarks whether QJL score correction improves attention-score fidelity
compared to using the quantized (base) key approximation alone.

Logic:
  fp16_scores  = queries @ keys / sqrt(D)
  base_scores  = queries @ k_base / sqrt(D)
  qjl_scores   = corrected_attention_scores(queries, k_base, sketch)

Metrics:
  base_score_mae, qjl_score_mae
  base_score_rmse, qjl_score_rmse
  base_topk_overlap, qjl_topk_overlap
  base_softmax_kl, qjl_softmax_kl

Pass condition:
  qjl_score_mae < base_score_mae
  AND qjl_softmax_kl < base_softmax_kl
  AND qjl_topk_overlap >= base_topk_overlap
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from rfsn_v10.quantization.hybrid_polar_cartesian import (
    HybridPolarCartesianQuantizer,
)
from rfsn_v10.quantization.qjl_score_correction import QJLScoreCorrector
from rfsn_v10.quantization.polar_quant import PolarQuantizer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _topk_overlap(a: mx.array, b: mx.array, k: int = 5) -> float:
    """Fraction of top-k indices that overlap between two score tensors."""
    a_idx = set(np.argsort(-np.array(a.flatten()))[:k].tolist())
    b_idx = set(np.argsort(-np.array(b.flatten()))[:k].tolist())
    if not a_idx:
        return 0.0
    return float(len(a_idx & b_idx) / len(a_idx))


def _softmax_kl(p: mx.array, q: mx.array) -> float:
    """KL divergence between softmax of two score tensors."""
    p_f = mx.softmax(p.flatten().astype(mx.float32), axis=0)
    q_f = mx.softmax(q.flatten().astype(mx.float32), axis=0)
    eps = 1e-10
    kl = mx.sum(p_f * mx.log((p_f + eps) / (q_f + eps)))
    return float(kl.item())


def _mae(a: mx.array, b: mx.array) -> float:
    diff = a.astype(mx.float32) - b.astype(mx.float32)
    return float(mx.mean(mx.abs(diff)).item())


def _rmse(a: mx.array, b: mx.array) -> float:
    diff = a.astype(mx.float32) - b.astype(mx.float32)
    return float(mx.sqrt(mx.mean(diff * diff)).item())


def _make_base_keys(keys: mx.array, mode: str, feature_dim: int) -> mx.array:
    """Create a quantized/dequantized base key approximation."""
    if mode == "polar":
        q = PolarQuantizer(
            levels=4, angle_bits=5, radius_bits=8, radius_group_size=64
        )
    elif mode == "hybrid":
        q = HybridPolarCartesianQuantizer(
            feature_dim=feature_dim,
            polar_ratio=0.65,
            polar_levels=4,
            polar_angle_bits=5,
            polar_radius_bits=8,
            cartesian_bits=5,
            group_size=64,
        )
    else:
        # Simple uniform quantization as base
        qmax = 15
        scale = mx.max(mx.abs(keys), axis=-1, keepdims=True) / qmax
        codes = mx.round(keys / scale).astype(mx.int32)
        codes = mx.clip(codes, -qmax, qmax)
        return codes.astype(mx.float32) * scale
    packed = q.quantize(keys)
    return q.dequantize(packed)


def _run_qjl_validation(
    *,
    feature_dim: int,
    proj_dim: int,
    num_queries: int,
    num_keys: int,
    batch_size: int,
    num_heads: int,
    base_mode: str,
    out_path: Path,
) -> dict[str, Any]:
    mx.random.seed(42)

    queries = mx.random.normal(
        (batch_size, num_heads, num_queries, feature_dim)
    ).astype(mx.float16)
    keys = mx.random.normal(
        (batch_size, num_heads, num_keys, feature_dim)
    ).astype(mx.float16)

    k_base = _make_base_keys(keys, base_mode, feature_dim)

    # fp16 reference scores
    scale = math.sqrt(float(feature_dim))
    k_t = keys.astype(mx.float32).transpose(0, 1, 3, 2)
    fp16_scores = (queries.astype(mx.float32) @ k_t) / scale

    # base (quantized-only) scores
    kb_t = k_base.astype(mx.float32).transpose(0, 1, 3, 2)
    base_scores = (queries.astype(mx.float32) @ kb_t) / scale

    # QJL-corrected scores
    qjl = QJLScoreCorrector(feature_dim=feature_dim, proj_dim=proj_dim)
    residual = keys.astype(mx.float32) - k_base.astype(mx.float32)
    sketch = qjl.sketch_residual(residual)
    qjl_scores = qjl.corrected_attention_scores(
        queries, k_base, sketch
    )

    # Metrics
    base_mae = _mae(fp16_scores, base_scores)
    qjl_mae = _mae(fp16_scores, qjl_scores)
    base_rmse = _rmse(fp16_scores, base_scores)
    qjl_rmse = _rmse(fp16_scores, qjl_scores)
    base_topk = _topk_overlap(fp16_scores, base_scores, k=5)
    qjl_topk = _topk_overlap(fp16_scores, qjl_scores, k=5)
    base_kl = _softmax_kl(fp16_scores, base_scores)
    qjl_kl = _softmax_kl(fp16_scores, qjl_scores)

    passes = (
        qjl_mae < base_mae
        and qjl_kl < base_kl
        and qjl_topk >= base_topk
    )

    result = {
        "feature_dim": feature_dim,
        "proj_dim": proj_dim,
        "num_queries": num_queries,
        "num_keys": num_keys,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "base_mode": base_mode,
        "base_score_mae": base_mae,
        "qjl_score_mae": qjl_mae,
        "base_score_rmse": base_rmse,
        "qjl_score_rmse": qjl_rmse,
        "base_topk_overlap": base_topk,
        "qjl_topk_overlap": qjl_topk,
        "base_softmax_kl": base_kl,
        "qjl_softmax_kl": qjl_kl,
        "qjl_improves_mae": qjl_mae < base_mae,
        "qjl_improves_kl": qjl_kl < base_kl,
        "qjl_preserves_topk": qjl_topk >= base_topk,
        "passes_all": passes,
        "notes": [
            "Pass condition: qjl_mae < base_mae AND "
            "qjl_kl < base_kl AND qjl_topk >= base_topk",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote QJL validation to {out_path}")
    print(f"  base_mae={base_mae:.6f} qjl_mae={qjl_mae:.6f}")
    print(f"  base_kl={base_kl:.6f} qjl_kl={qjl_kl:.6f}")
    print(f"  base_topk={base_topk:.4f} qjl_topk={qjl_topk:.4f}")
    print(f"  passes_all={passes}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate QJL attention-score correction"
    )
    parser.add_argument(
        "--feature-dim", type=int, default=64, help="Feature dimension"
    )
    parser.add_argument(
        "--proj-dim", type=int, default=64, help="QJL projection dimension"
    )
    parser.add_argument(
        "--num-queries", type=int, default=32, help="Number of query tokens"
    )
    parser.add_argument(
        "--num-keys", type=int, default=128, help="Number of key tokens"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size"
    )
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--base-mode",
        default="hybrid",
        choices=["polar", "hybrid", "uniform"],
        help="Base quantization mode for keys",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/qjl_attention_score.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    _run_qjl_validation(
        feature_dim=args.feature_dim,
        proj_dim=args.proj_dim,
        num_queries=args.num_queries,
        num_keys=args.num_keys,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        base_mode=args.base_mode,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
