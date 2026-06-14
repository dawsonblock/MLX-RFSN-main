#!/usr/bin/env python3
"""Divergent layer/token identification script.

This script identifies the exact layer and token where quantization
divergence exceeds acceptable thresholds, providing detailed analysis
to help pinpoint the root cause of quality degradation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def identify_divergent_layer_token(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "Hello, world!",
    max_tokens: int = 100,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
    max_diff_threshold: float = 0.1,
    cosine_threshold: float = 0.99,
) -> dict[str, Any]:
    """Identify the first layer and token where divergence occurs.

    Args:
        model_id: Model to test with.
        prompt: Test prompt.
        max_tokens: Maximum tokens to generate.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.
        max_diff_threshold: Maximum acceptable max difference.
        cosine_threshold: Minimum acceptable cosine similarity.

    Returns:
        Dictionary containing divergence analysis.
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

    results = {
        "model_id": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
            "max_diff_threshold": max_diff_threshold,
            "cosine_threshold": cosine_threshold,
        },
        "divergence_points": [],
        "first_divergence": None,
        "divergence_summary": {},
    }

    # Get baseline and quantized outputs
    print("Computing baseline outputs...")
    baseline_outputs = _get_detailed_outputs(model, tokenizer, prompt, max_tokens, use_cache=False)

    print("Computing quantized outputs...")
    quantized_outputs = _get_detailed_outputs(
        model, tokenizer, prompt, max_tokens,
        use_cache=True, key_bits=key_bits, value_bits=value_bits, group_size=group_size
    )

    # Analyze divergence point by point
    print("Analyzing divergence...")
    for token_idx in range(min(len(baseline_outputs), len(quantized_outputs))):
        baseline = baseline_outputs[token_idx]
        quantized = quantized_outputs[token_idx]

        # Compare logits
        diff = _compare_outputs(baseline, quantized)

        if diff["max_difference"] > max_diff_threshold or diff["cosine_similarity"] < cosine_threshold:
            divergence_point = {
                "token": token_idx,
                "max_difference": diff["max_difference"],
                "cosine_similarity": diff["cosine_similarity"],
                "exceeds_max_diff": diff["max_difference"] > max_diff_threshold,
                "below_cosine": diff["cosine_similarity"] < cosine_threshold,
                "baseline_top_token": baseline["top_token"],
                "quantized_top_token": quantized["top_token"],
                "top_token_mismatch": baseline["top_token"] != quantized["top_token"],
            }
            results["divergence_points"].append(divergence_point)

            # Track first divergence
            if results["first_divergence"] is None:
                results["first_divergence"] = divergence_point

    # Generate summary
    results["divergence_summary"] = _generate_divergence_summary(results["divergence_points"])

    return results


def _get_detailed_outputs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    use_cache: bool = False,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> list[dict[str, Any]]:
    """Get detailed model outputs with or without quantization."""
    import mlx.core as mx
    import numpy as np

    if use_cache:
        from rfsn_v10.cache.cartesian_codec import CartesianCodec
        from rfsn_v10.cache.session import GenerationCacheSession
        from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
        from rfsn_v10.config import QuantizationConfig, RFSNConfig
        from rfsn_v10.runtime.generation import RFSNGenerator

        key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
        value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

        session = GenerationCacheSession(
            name="divergence_test",
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
        )
    else:
        cache_list = None

    prompt_ids = tokenizer.encode(prompt)
    outputs = []

    # Prefill
    y = mx.array(prompt_ids)
    logits = model(y[None], cache=cache_list)
    logits = logits[:, -1, :]
    outputs.append({
        "stage": "prefill",
        "logits": np.array(logits.astype(mx.float32).squeeze(0)),
        "top_token": int(mx.argmax(logits, axis=-1)[0]),
    })

    # Decode
    for token_idx in range(max_tokens):
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]
        outputs.append({
            "stage": f"decode_{token_idx}",
            "logits": np.array(logits.astype(mx.float32).squeeze(0)),
            "top_token": int(mx.argmax(logits, axis=-1)[0]),
        })
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    return outputs


def _compare_outputs(baseline: dict[str, Any], quantized: dict[str, Any]) -> dict[str, Any]:
    """Compare baseline and quantized outputs."""
    import numpy as np

    base_logits = baseline["logits"]
    quant_logits = quantized["logits"]

    # Max difference
    max_diff = float(np.max(np.abs(base_logits - quant_logits)))

    # Cosine similarity
    norm_base = np.linalg.norm(base_logits)
    norm_quant = np.linalg.norm(quant_logits)
    if norm_base > 0 and norm_quant > 0:
        cosine = float(np.dot(base_logits, quant_logits) / (norm_base * norm_quant))
    else:
        cosine = 0.0

    return {
        "max_difference": max_diff,
        "cosine_similarity": cosine,
    }


def _generate_divergence_summary(divergence_points: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary of divergence patterns."""
    summary = {
        "total_divergences": len(divergence_points),
        "first_divergence_token": None,
        "divergence_rate": 0.0,
        "max_diff_observed": 0.0,
        "min_cosine_observed": 1.0,
        "top_token_mismatches": 0,
    }

    if divergence_points:
        summary["first_divergence_token"] = divergence_points[0]["token"]
        summary["max_diff_observed"] = max(p["max_difference"] for p in divergence_points)
        summary["min_cosine_observed"] = min(p["cosine_similarity"] for p in divergence_points)
        summary["top_token_mismatches"] = sum(1 for p in divergence_points if p["top_token_mismatch"])

    return summary


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify divergent layer and token"
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
        default="Hello, world!",
        help="Test prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
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
        "--max-diff-threshold",
        type=float,
        default=0.1,
        help="Maximum acceptable max difference",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.99,
        help="Minimum acceptable cosine similarity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/divergence_analysis.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Divergent Layer/Token Identification ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")
    print(f"Thresholds: max_diff={args.max_diff_threshold}, cosine={args.cosine_threshold}")

    results = identify_divergent_layer_token(
        args.model, args.prompt, args.max_tokens,
        args.key_bits, args.value_bits, args.group_size,
        args.max_diff_threshold, args.cosine_threshold
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Analysis skipped: MLX not available")
        return 1

    summary = results.get("divergence_summary", {})
    print(f"\nDivergence summary:")
    print(f"  Total divergences: {summary['total_divergences']}")
    print(f"  First divergence token: {summary['first_divergence_token']}")
    print(f"  Max diff observed: {summary['max_diff_observed']:.6f}")
    print(f"  Min cosine observed: {summary['min_cosine_observed']:.6f}")
    print(f"  Top token mismatches: {summary['top_token_mismatches']}")

    first_divergence = results.get("first_divergence")
    if first_divergence:
        print(f"\nFirst divergence details:")
        print(f"  Token: {first_divergence['token']}")
        print(f"  Max difference: {first_divergence['max_difference']:.6f}")
        print(f"  Cosine similarity: {first_divergence['cosine_similarity']:.6f}")
        print(f"  Top token mismatch: {first_divergence['top_token_mismatch']}")
    else:
        print("\nNo divergence detected within thresholds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
