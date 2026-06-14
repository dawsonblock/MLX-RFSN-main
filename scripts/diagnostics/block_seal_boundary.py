#!/usr/bin/env python3
"""Block-seal boundary test script (tokens 62-66).

This script tests specifically around the block-seal boundary (tokens 62-66)
to detect if quality collapse occurs when blocks transition from staging to sealed.

A sharp quality collapse at 64 or 65 would indicate lifecycle, ordering,
or offset failure rather than ordinary quantization error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_block_seal_boundary(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 10,  # Long prompt to trigger block sealing
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
    staging_capacity: int = 64,
) -> dict[str, Any]:
    """Test quality around block-seal boundary.

    Args:
        model_id: Model to test with.
        prompt: Test prompt (long enough to trigger block sealing).
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.
        staging_capacity: Staging capacity.

    Returns:
        Dictionary containing boundary test results.
    """
    try:
        import mlx_lm
        import numpy as np
        import mlx.core as mx
    except ImportError as e:
        return {
            "error": f"MLX not available: {e}",
            "status": "SKIPPED_NO_MLX",
        }

    print(f"Loading model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)

    # Generate enough tokens to cross block-seal boundary
    # With staging_capacity=64, first block seal should occur around token 64
    target_tokens = 70  # Test tokens 60-70
    boundary_range = range(60, 71)  # Focus on 60-70

    results = {
        "model_id": model_id,
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
            "staging_capacity": staging_capacity,
        },
        "boundary_range": list(boundary_range),
        "token_metrics": [],
        "boundary_analysis": {},
    }

    # Get baseline metrics
    print("Computing baseline metrics...")
    baseline_metrics = _get_token_metrics_baseline(
        model, tokenizer, prompt, target_tokens
    )

    # Get quantized metrics
    print("Computing quantized metrics...")
    quantized_metrics = _get_token_metrics_quantized(
        model, tokenizer, prompt, target_tokens,
        key_bits, value_bits, group_size, staging_capacity
    )

    # Compare metrics around boundary
    print("Analyzing boundary region...")
    for token_idx in boundary_range:
        if token_idx < len(baseline_metrics) and token_idx < len(quantized_metrics):
            comparison = {
                "token": token_idx,
                "baseline_logit_max": float(baseline_metrics[token_idx]["logit_max"]),
                "quantized_logit_max": float(quantized_metrics[token_idx]["logit_max"]),
                "logit_max_diff": float(abs(
                    baseline_metrics[token_idx]["logit_max"] -
                    quantized_metrics[token_idx]["logit_max"]
                )),
                "baseline_top_token": int(baseline_metrics[token_idx]["top_token"]),
                "quantized_top_token": int(quantized_metrics[token_idx]["top_token"]),
                "top_token_match": (
                    baseline_metrics[token_idx]["top_token"] ==
                    quantized_metrics[token_idx]["top_token"]
                ),
                "cosine_similarity": float(_compute_cosine_similarity(
                    baseline_metrics[token_idx]["logits"],
                    quantized_metrics[token_idx]["logits"]
                )),
            }
            results["token_metrics"].append(comparison)

    # Analyze boundary behavior
    results["boundary_analysis"] = _analyze_boundary_behavior(results["token_metrics"])

    return results


def _get_token_metrics_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Get token-level metrics from baseline model."""
    import mlx.core as mx
    import numpy as np

    prompt_ids = tokenizer.encode(prompt)
    metrics = []

    # Prefill
    y = mx.array(prompt_ids)
    logits = model(y[None])
    logits = logits[:, -1, :]
    metrics.append({
        "logits": np.array(logits.astype(mx.float32).squeeze(0)),
        "logit_max": float(mx.max(logits)),
        "top_token": int(mx.argmax(logits, axis=-1)[0]),
    })

    # Decode
    for _ in range(max_tokens):
        logits = model(y[None])
        logits = logits[:, -1, :]
        metrics.append({
            "logits": np.array(logits.astype(mx.float32).squeeze(0)),
            "logit_max": float(mx.max(logits)),
            "top_token": int(mx.argmax(logits, axis=-1)[0]),
        })
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    return metrics


def _get_token_metrics_quantized(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
    staging_capacity: int,
) -> list[dict[str, Any]]:
    """Get token-level metrics from quantized model."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx
    import numpy as np

    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name="block_seal_test",
        num_layers=len(model.layers),
        key_codec=key_codec,
        value_codec=value_codec,
    )

    cache_list = [
        RfsnQuantizedKVCache(
            layer_cache=session.get_layer_cache(i),
            session=session,
        )
        for i in range(len(model.layers))
    ]

    cfg = RFSNConfig(
        quantization=QuantizationConfig(
            default_bits=key_bits,
            group_size=group_size,
        ),
    )
    generator = RFSNGenerator(
        model,
        tokenizer,
        cfg,
        enable_quantized_kv=True,
        staging_capacity=staging_capacity,
    )

    prompt_ids = tokenizer.encode(prompt)
    metrics = []

    # Prefill
    y = mx.array(prompt_ids)
    logits = model(y[None], cache=cache_list)
    logits = logits[:, -1, :]
    metrics.append({
        "logits": np.array(logits.astype(mx.float32).squeeze(0)),
        "logit_max": float(mx.max(logits)),
        "top_token": int(mx.argmax(logits, axis=-1)[0]),
    })

    # Decode
    for _ in range(max_tokens):
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]
        metrics.append({
            "logits": np.array(logits.astype(mx.float32).squeeze(0)),
            "logit_max": float(mx.max(logits)),
            "top_token": int(mx.argmax(logits, axis=-1)[0]),
        })
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    return metrics


def _compute_cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        return float(np.dot(a, b) / (norm_a * norm_b))
    return 0.0


def _analyze_boundary_behavior(token_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze behavior around block-seal boundary."""
    analysis = {
        "sharp_quality_drop": False,
        "drop_token": None,
        "top_token_mismatch_rate": 0.0,
        "mean_cosine_before": 0.0,
        "mean_cosine_after": 0.0,
        "cosine_drop": 0.0,
    }

    # Find where top token mismatches begin
    mismatches = [m for m in token_metrics if not m["top_token_match"]]
    if mismatches:
        analysis["drop_token"] = mismatches[0]["token"]
        analysis["top_token_mismatch_rate"] = len(mismatches) / len(token_metrics)

    # Check for sharp cosine drop at token 64
    before_64 = [m for m in token_metrics if m["token"] < 64]
    after_64 = [m for m in token_metrics if m["token"] >= 64]

    if before_64 and after_64:
        analysis["mean_cosine_before"] = sum(m["cosine_similarity"] for m in before_64) / len(before_64)
        analysis["mean_cosine_after"] = sum(m["cosine_similarity"] for m in after_64) / len(after_64)
        analysis["cosine_drop"] = analysis["mean_cosine_before"] - analysis["mean_cosine_after"]

        # Flag sharp drop (>0.01 cosine drop)
        if analysis["cosine_drop"] > 0.01:
            analysis["sharp_quality_drop"] = True

    return analysis


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test block-seal boundary behavior"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model to test with",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog. " * 10,
        help="Test prompt",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=8,
        help="Key quantization bits",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=8,
        help="Value quantization bits",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--staging-capacity",
        type=int,
        default=64,
        help="Staging capacity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/block_seal_boundary.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Block-Seal Boundary Test ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")
    print(f"Staging capacity: {args.staging_capacity}")

    results = test_block_seal_boundary(
        args.model, args.prompt,
        args.key_bits, args.value_bits, args.group_size, args.staging_capacity
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Test skipped: MLX not available")
        return 1

    analysis = results.get("boundary_analysis", {})
    print(f"\nBoundary analysis:")
    print(f"  Sharp quality drop: {analysis['sharp_quality_drop']}")
    print(f"  Drop token: {analysis['drop_token']}")
    print(f"  Top token mismatch rate: {analysis['top_token_mismatch_rate']:.2%}")
    print(f"  Cosine before token 64: {analysis['mean_cosine_before']:.4f}")
    print(f"  Cosine after token 64: {analysis['mean_cosine_after']:.4f}")
    print(f"  Cosine drop: {analysis['cosine_drop']:.4f}")

    if analysis["sharp_quality_drop"]:
        print("\n⚠️  WARNING: Sharp quality drop detected at block-seal boundary")
        print("This indicates a potential lifecycle, ordering, or offset failure.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
