#!/usr/bin/env python3
"""K/V bit-width isolation ladder diagnostic script.

This script tests progressively lower bit-widths to identify where
quality degradation begins. It runs the diagnostic ladder:

1. K16/V16 (near FP16 quality)
2. K8/V16
3. K16/V8
4. K8/V8
5. K8/V6
6. K8/V5 (current problematic configuration)
7. WHT disabled
8. Sign transform disabled
9. Group size 32 vs 64
10. Various staging capacities
11. Various dense residual windows

At each layer and token, we compare against native MLX to find the
first point of divergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_bit_width_ladder(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "Hello, world!",
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Run the bit-width isolation ladder diagnostic.

    Args:
        model_id: Model to test with.
        prompt: Test prompt.
        max_tokens: Maximum tokens to generate.

    Returns:
        Dictionary containing ladder results and divergence analysis.
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

    # Define ladder configurations
    ladder_configs = [
        {"name": "k16_v16_gs64", "key_bits": 16, "value_bits": 16, "group_size": 64},
        {"name": "k8_v16_gs64", "key_bits": 8, "value_bits": 16, "group_size": 64},
        {"name": "k16_v8_gs64", "key_bits": 16, "value_bits": 8, "group_size": 64},
        {"name": "k8_v8_gs64", "key_bits": 8, "value_bits": 8, "group_size": 64},
        {"name": "k8_v6_gs64", "key_bits": 8, "value_bits": 6, "group_size": 64},
        {"name": "k8_v5_gs64", "key_bits": 8, "value_bits": 5, "group_size": 64},
        {"name": "k8_v8_gs64_no_wht", "key_bits": 8, "value_bits": 8, "group_size": 64, "use_wht": False},
        {"name": "k8_v8_gs64_no_sign", "key_bits": 8, "value_bits": 8, "group_size": 64, "use_sign": False},
        {"name": "k8_v8_gs32", "key_bits": 8, "value_bits": 8, "group_size": 32},
        {"name": "k8_v8_gs128", "key_bits": 8, "value_bits": 8, "group_size": 128},
    ]

    results = {
        "model_id": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ladder_results": [],
        "divergence_analysis": {},
    }

    # Get baseline (native MLX) logits
    print("Computing baseline (native MLX)...")
    baseline_logits = _get_baseline_logits(model, tokenizer, prompt, max_tokens)

    # Test each configuration
    for config in ladder_configs:
        print(f"\nTesting config: {config['name']}")
        config_result = _test_config(
            model, tokenizer, prompt, max_tokens, config, baseline_logits
        )
        results["ladder_results"].append(config_result)

    # Analyze divergence
    results["divergence_analysis"] = _analyze_divergence(results["ladder_results"])

    return results


def _get_baseline_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
) -> list[Any]:
    """Get baseline logits using native MLX without quantization."""
    import mlx.core as mx

    prompt_ids = tokenizer.encode(prompt)
    logits_list = []

    # Prefill
    y = mx.array(prompt_ids)
    prefill_logits = model(y[None])
    prefill_logits = prefill_logits[:, -1, :]
    logits_list.append(prefill_logits)

    # Decode
    for _ in range(max_tokens):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logits_list.append(logits)
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    return logits_list


def _test_config(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    config: dict[str, Any],
    baseline_logits: list[Any],
) -> dict[str, Any]:
    """Test a single quantization configuration."""
    try:
        from rfsn_v10.cache.cartesian_codec import CartesianCodec
        from rfsn_v10.cache.session import GenerationCacheSession
        from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
        from rfsn_v10.config import QuantizationConfig, RFSNConfig
        from rfsn_v10.runtime.generation import RFSNGenerator
        import mlx.core as mx
        import numpy as np

        # Configure quantization
        key_bits = config["key_bits"]
        value_bits = config["value_bits"]
        group_size = config["group_size"]
        use_wht = config.get("use_wht", True)
        use_sign = config.get("use_sign", True)

        key_codec = CartesianCodec(bits=key_bits, group_size=group_size, use_wht=use_wht, use_sign=use_sign)
        value_codec = CartesianCodec(bits=value_bits, group_size=group_size, use_wht=use_wht, use_sign=use_sign)

        session = GenerationCacheSession(
            name=f"diagnostic_{config['name']}",
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

        # Configure generator
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

        # Get quantized logits
        prompt_ids = tokenizer.encode(prompt)
        logits_list = []

        # Prefill
        y = mx.array(prompt_ids)
        prefill_logits = model(y[None], cache=cache_list)
        prefill_logits = prefill_logits[:, -1, :]
        logits_list.append(prefill_logits)

        # Decode
        for _ in range(max_tokens):
            logits = model(y[None], cache=cache_list)
            logits = logits[:, -1, :]
            logits_list.append(logits)
            token_id = int(mx.argmax(logits, axis=-1)[0])
            y = mx.array([token_id])

        # Compare with baseline
        divergences = []
        max_differences = []
        cosine_similarities = []

        for i, (quant_logit, base_logit) in enumerate(zip(logits_list, baseline_logits)):
            quant_np = np.array(quant_logit.astype(mx.float32).squeeze(0))
            base_np = np.array(base_logit.astype(mx.float32).squeeze(0))

            # Max difference
            max_diff = np.max(np.abs(quant_np - base_np))
            max_differences.append(max_diff)

            # Cosine similarity
            norm_quant = np.linalg.norm(quant_np)
            norm_base = np.linalg.norm(base_np)
            if norm_quant > 0 and norm_base > 0:
                cosine = np.dot(quant_np, base_np) / (norm_quant * norm_base)
            else:
                cosine = 0.0
            cosine_similarities.append(cosine)

            # Check for significant divergence
            if max_diff > 0.1 or cosine < 0.99:
                divergences.append({
                    "token": i,
                    "max_diff": float(max_diff),
                    "cosine": float(cosine),
                })

        return {
            "config_name": config["name"],
            "config": config,
            "max_differences": [float(d) for d in max_differences],
            "cosine_similarities": [float(c) for c in cosine_similarities],
            "divergences": divergences,
            "first_divergence_token": divergences[0]["token"] if divergences else None,
            "mean_max_diff": float(np.mean(max_differences)),
            "mean_cosine": float(np.mean(cosine_similarities)),
            "status": "PASS" if not divergences else "DIVERGENCE",
        }

    except Exception as e:
        import traceback
        return {
            "config_name": config["name"],
            "config": config,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "ERROR",
        }


def _analyze_divergence(ladder_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze divergence patterns across the ladder."""
    analysis = {
        "first_divergent_config": None,
        "divergence_progression": [],
        "quality_thresholds": {
            "acceptable_max_diff": 0.01,
            "acceptable_cosine": 0.999,
        },
    }

    for result in ladder_results:
        if result["status"] == "DIVERGENCE":
            if analysis["first_divergent_config"] is None:
                analysis["first_divergent_config"] = result["config_name"]

            analysis["divergence_progression"].append({
                "config": result["config_name"],
                "first_divergence_token": result["first_divergence_token"],
                "mean_max_diff": result["mean_max_diff"],
                "mean_cosine": result["mean_cosine"],
            })

    return analysis


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run K/V bit-width isolation ladder diagnostic"
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
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/bit_width_ladder.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== K/V Bit-Width Isolation Ladder Diagnostic ===")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    results = run_bit_width_ladder(args.model, args.prompt, args.max_tokens)

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Diagnostic skipped: MLX not available")
        return 1

    analysis = results.get("divergence_analysis", {})
    first_divergent = analysis.get("first_divergent_config")
    if first_divergent:
        print(f"\nFirst divergent config: {first_divergent}")
        print("Divergence progression:")
        for prog in analysis.get("divergence_progression", []):
            print(f"  {prog['config']}: token {prog['first_divergence_token']}, "
                  f"max_diff={prog['mean_max_diff']:.4f}, cosine={prog['mean_cosine']:.4f}")
    else:
        print("\nNo divergence detected in tested configurations")

    return 0


if __name__ == "__main__":
    sys.exit(main())
