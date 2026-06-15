#!/usr/bin/env python3
"""Production validation using MLX-LM direct-packed attention.

Replaces the PyTorch-based ``validate_production_model.py`` with a native
MLX-LM path that exercises:
* PackedV4AttentionKernel
* RfsnDirectPackedKVCache
* The K8/V8 direct-packed dispatch path

This script is invoked by the production-validation workflow when
RFSN_ENABLE_TRUE_PACKED=1 is set.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.utils import generate_step

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
    RfsnDirectPackedKVCache,
    collect_backend_stats,
    unwrap_model_attention,
    wrap_model_attention,
)
from rfsn_v10.kernels.metal.packed_v4_attention import HAS_TRUE_PACKED_KERNEL


def _cosine(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1)
    b_f = b.reshape(-1)
    dot = mx.sum(a_f * b_f).item()
    na = mx.sum(a_f * a_f).item() ** 0.5
    nb = mx.sum(b_f * b_f).item() ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _topk_overlap(a: mx.array, b: mx.array, k: int) -> float:
    ai = set(mx.argpartition(a, -k)[-k:].tolist())
    bi = set(mx.argpartition(b, -k)[-k:].tolist())
    if not ai:
        return 0.0
    return len(ai & bi) / len(ai)


def _perplexity(logits: mx.array, target: int) -> float:
    log_probs = mx.log(mx.softmax(logits.astype(mx.float32)))
    return float(mx.exp(-log_probs[target]).item())


def _dense_baseline(
    model: Any, tokenizer: Any, prompt_ids: mx.array, max_tokens: int
) -> dict[str, Any]:
    """Run dense MLX-LM baseline and return logits + tokens."""
    tokens = []
    logits_list = []
    t0 = time.perf_counter()
    for token, logit in generate_step(
        prompt_ids, model, max_tokens=max_tokens, temp=0.0
    ):
        tokens.append(int(token))
        logits_list.append(logit)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "tokens": tokens,
        "logits": logits_list,
        "total_ms": total_ms,
        "per_token_ms": total_ms / max_tokens if max_tokens > 0 else 0.0,
    }


def _packed_path(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    max_tokens: int,
    staging_capacity: int = 64,
) -> dict[str, Any]:
    """Run direct-packed path and return logits + tokens + stats."""
    k_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    caches = [
        RfsnDirectPackedKVCache(
            layer_id=i,
            key_codec=k_codec,
            value_codec=v_codec,
            staging_capacity=staging_capacity,
            dense_residual_window=0,
            strict=True,
        )
        for i in range(len(model.layers))
    ]

    wrap_model_attention(model, caches, strict=True)
    try:
        tokens = []
        logits_list = []
        t0 = time.perf_counter()
        for token, logit in generate_step(
            prompt_ids, model, max_tokens=max_tokens, temp=0.0, prompt_cache=caches
        ):
            tokens.append(int(token))
            logits_list.append(logit)
        total_ms = (time.perf_counter() - t0) * 1000.0
        stats = collect_backend_stats(model)
    finally:
        unwrap_model_attention(model)

    layer0 = caches[0].layer_cache
    return {
        "tokens": tokens,
        "logits": logits_list,
        "total_ms": total_ms,
        "per_token_ms": total_ms / max_tokens if max_tokens > 0 else 0.0,
        "stats": stats,
        "requantized_tokens": layer0.requantized_token_count,
        "sealed_blocks": len(list(layer0.iter_key_blocks())),
        "total_tokens": layer0.total_token_count(),
    }


def _compare_runs(
    dense: dict[str, Any], packed: dict[str, Any]
) -> dict[str, Any]:
    """Compute differential metrics between dense and packed runs."""
    d_tokens = dense["tokens"]
    p_tokens = packed["tokens"]

    token_match_rate = (
        sum(1 for a, b in zip(d_tokens, p_tokens) if a == b) / len(d_tokens)
        if d_tokens else 0.0
    )

    logit_cosines = []
    logit_max_diffs = []
    top1_matches = []
    top5_overlaps = []
    perplexity_deltas = []

    for d_logit, p_logit in zip(dense["logits"], packed["logits"]):
        d_logit = d_logit.reshape(-1)
        p_logit = p_logit.reshape(-1)
        logit_cosines.append(_cosine(d_logit, p_logit))
        logit_max_diffs.append(
            float(mx.max(mx.abs(d_logit - p_logit)).item())
        )
        top1_matches.append(
            int(mx.argmax(d_logit).item()) == int(mx.argmax(p_logit).item())
        )
        top5_overlaps.append(_topk_overlap(d_logit, p_logit, k=5))

    return {
        "token_match_rate": round(token_match_rate, 4),
        "logit_cosine_mean": round(float(mx.array(logit_cosines).mean().item()), 4),
        "logit_cosine_min": round(float(mx.array(logit_cosines).min().item()), 4),
        "logit_max_abs_diff_mean": round(
            float(mx.array(logit_max_diffs).mean().item()), 4
        ),
        "top1_match_rate": round(
            sum(top1_matches) / len(top1_matches) if top1_matches else 0.0, 4
        ),
        "top5_overlap_mean": round(
            float(mx.array(top5_overlaps).mean().item()), 4
        ),
        "packed_vs_dense_latency_ratio": round(
            packed["total_ms"] / dense["total_ms"] if dense["total_ms"] > 0 else 0.0,
            3,
        ),
        "dense_per_token_ms": round(dense["per_token_ms"], 3),
        "packed_per_token_ms": round(packed["per_token_ms"], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate direct-packed attention against dense baseline"
    )
    parser.add_argument(
        "--model-id",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="MLX-LM model identifier",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize the process of photosynthesis in three sentences.",
        help="Generation prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="Tokens to generate"
    )
    parser.add_argument(
        "--staging-capacity", type=int, default=64, help="Block seal threshold"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/proof/packed_direct_validation.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--require-packed-kernel",
        action="store_true",
        default=True,
        help="Fail if HAS_TRUE_PACKED_KERNEL is False",
    )
    args = parser.parse_args()

    if args.require_packed_kernel and not HAS_TRUE_PACKED_KERNEL:
        print(
            "FATAL: HAS_TRUE_PACKED_KERNEL is False. "
            "Set RFSN_ENABLE_TRUE_PACKED=1 and ensure Metal is available.",
            file=os.sys.stderr,
        )
        return 1

    print(f"Loading model: {args.model_id}")
    model, tokenizer = load(args.model_id)
    prompt_ids = mx.array(tokenizer.encode(args.prompt))

    print("Running dense baseline...")
    dense_result = _dense_baseline(model, tokenizer, prompt_ids, args.max_tokens)

    print("Running packed path...")
    packed_result = _packed_path(
        model, tokenizer, prompt_ids, args.max_tokens, args.staging_capacity
    )

    print("Computing differential metrics...")
    comparison = _compare_runs(dense_result, packed_result)

    # Backend audit
    all_backends = [s["executed_backend"] for s in packed_result.get("stats", [])]
    all_contracts = [s.get("execution_contract") for s in packed_result.get("stats", [])]

    # P0: require actual packed dispatch
    packed_dispatched = any("packed_metal" in b for b in all_backends)
    all_contracts_present = all(c is not None for c in all_contracts)
    min_blocks = min((c["num_key_blocks"] for c in all_contracts if c), default=0)

    result = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "staging_capacity": args.staging_capacity,
        "has_true_packed_kernel": HAS_TRUE_PACKED_KERNEL,
        "token_match_rate": comparison["token_match_rate"],
        "logit_cosine_mean": comparison["logit_cosine_mean"],
        "logit_cosine_min": comparison["logit_cosine_min"],
        "logit_max_abs_diff_mean": comparison["logit_max_abs_diff_mean"],
        "top1_match_rate": comparison["top1_match_rate"],
        "top5_overlap_mean": comparison["top5_overlap_mean"],
        "dense_per_token_ms": comparison["dense_per_token_ms"],
        "packed_per_token_ms": comparison["packed_per_token_ms"],
        "packed_vs_dense_latency_ratio": comparison["packed_vs_dense_latency_ratio"],
        "requantized_tokens": packed_result["requantized_tokens"],
        "sealed_blocks": packed_result["sealed_blocks"],
        "total_tokens": packed_result["total_tokens"],
        "packed_dispatched": packed_dispatched,
        "all_contracts_present": all_contracts_present,
        "min_key_blocks": min_blocks,
        "layer_count": len(model.layers),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Results written to {args.out}")

    # Hard-fail checks
    ok = True
    if not packed_dispatched:
        print("ERROR: No layer dispatched the packed kernel.", file=os.sys.stderr)
        ok = False
    if not all_contracts_present:
        print("ERROR: Some layers missing execution contract.", file=os.sys.stderr)
        ok = False
    if min_blocks == 0:
        print("ERROR: Zero key blocks recorded.", file=os.sys.stderr)
        ok = False
    if comparison["token_match_rate"] < 0.95:
        print(
            f"ERROR: Token match rate {comparison['token_match_rate']} < 0.95",
            file=os.sys.stderr,
        )
        ok = False
    if comparison["logit_cosine_min"] < 0.99:
        print(
            f"WARNING: Logit cosine min {comparison['logit_cosine_min']} < 0.99",
            file=os.sys.stderr,
        )

    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
