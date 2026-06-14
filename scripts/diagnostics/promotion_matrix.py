#!/usr/bin/env python3
"""Multi-model promotion matrix infrastructure.

This script tests candidates across multiple models to create a promotion
matrix showing which candidates pass quality gates on which models.

Models to test:
- Qwen2.5-0.5B-Instruct-4bit (small, fast)
- Qwen2.5-1.5B-Instruct (medium)
- Qwen2.5-3B-Instruct (larger)

This provides evidence for promotion decisions across model sizes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_multi_model_promotion_matrix(
    models: list[str] = [
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct",
        "mlx-community/Qwen2.5-3B-Instruct",
    ],
    prompt: str = "Hello, world!",
    max_tokens: int = 50,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run promotion matrix across multiple models.

    Args:
        models: List of model IDs to test.
        prompt: Test prompt.
        max_tokens: Maximum tokens to generate.
        candidates: List of candidate configurations to test.

    Returns:
        Dictionary containing promotion matrix results.
    """
    try:
        import mlx_lm
    except ImportError as e:
        return {
            "error": f"MLX not available: {e}",
            "status": "SKIPPED_NO_MLX",
        }

    if candidates is None:
        candidates = [
            {"name": "baseline", "type": "baseline"},
            {"name": "k8_v8_gs64", "key_bits": 8, "value_bits": 8, "group_size": 64},
            {"name": "k8_v5_gs64", "key_bits": 8, "value_bits": 5, "group_size": 64},
        ]

    results = {
        "models_tested": models,
        "candidates_tested": [c["name"] for c in candidates],
        "matrix": [],
        "summary": {},
    }

    for model_id in models:
        print(f"\nTesting model: {model_id}")
        model_results = _test_model(
            model_id, prompt, max_tokens, candidates
        )
        results["matrix"].append(model_results)

    # Generate summary
    results["summary"] = _generate_promotion_summary(results["matrix"])

    return results


def _test_model(
    model_id: str,
    prompt: str,
    max_tokens: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test all candidates on a single model."""
    import mlx_lm
    import mlx.core as mx
    import numpy as np

    print(f"  Loading model...")
    model, tokenizer = mlx_lm.load(model_id)

    model_results = {
        "model_id": model_id,
        "candidate_results": [],
    }

    for candidate_config in candidates:
        print(f"  Testing candidate: {candidate_config['name']}")
        candidate_result = _test_candidate(
            model, tokenizer, prompt, max_tokens, candidate_config
        )
        model_results["candidate_results"].append(candidate_result)

    return model_results


def _test_candidate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    """Test a single candidate."""
    import mlx.core as mx
    import numpy as np

    if candidate_config.get("type") == "baseline":
        return _test_baseline(model, tokenizer, prompt, max_tokens, candidate_config)
    else:
        return _test_quantized(
            model, tokenizer, prompt, max_tokens, candidate_config
        )


def _test_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    """Test baseline candidate."""
    import mlx.core as mx
    import numpy as np

    prompt_ids = tokenizer.encode(prompt)
    logits_list = []

    # Prefill
    y = mx.array(prompt_ids)
    logits = model(y[None])
    logits = logits[:, -1, :]
    logits_list.append(np.array(logits.astype(mx.float32).squeeze(0)))

    # Decode
    for _ in range(max_tokens):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logits_list.append(np.array(logits.astype(mx.float32).squeeze(0)))
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    return {
        "candidate_name": candidate_config["name"],
        "candidate_type": "baseline",
        "logits": logits_list,
        "status": "PASS",
        "promotion_eligible": False,  # Baseline is not a compression candidate
    }


def _test_quantized(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    """Test quantized candidate."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx
    import numpy as np

    key_bits = candidate_config["key_bits"]
    value_bits = candidate_config["value_bits"]
    group_size = candidate_config["group_size"]

    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name=f"promotion_matrix_{candidate_config['name']}",
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

    prompt_ids = tokenizer.encode(prompt)
    logits_list = []

    try:
        # Prefill
        y = mx.array(prompt_ids)
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]
        logits_list.append(np.array(logits.astype(mx.float32).squeeze(0)))

        # Decode
        for _ in range(max_tokens):
            logits = model(y[None], cache=cache_list)
            logits = logits[:, -1, :]
            logits_list.append(np.array(logits.astype(mx.float32).squeeze(0)))
            token_id = int(mx.argmax(logits, axis=-1)[0])
            y = mx.array([token_id])

        # Get runtime counters
        counters = session.counters()

        return {
            "candidate_name": candidate_config["name"],
            "candidate_type": "quantized",
            "config": candidate_config,
            "logits": logits_list,
            "runtime_counters": counters,
            "status": "PASS",
            "promotion_eligible": counters.get("dense_fallback_calls", 0) == 0,
        }

    except Exception as e:
        import traceback
        return {
            "candidate_name": candidate_config["name"],
            "candidate_type": "quantized",
            "config": candidate_config,
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "promotion_eligible": False,
        }


def _generate_promotion_summary(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary of promotion matrix."""
    summary = {
        "total_models": len(matrix),
        "candidates_by_name": {},
        "promotion_matrix": {},
    }

    # Aggregate results by candidate name
    for model_result in matrix:
        model_id = model_result["model_id"]
        for candidate_result in model_result["candidate_results"]:
            candidate_name = candidate_result["candidate_name"]

            if candidate_name not in summary["candidates_by_name"]:
                summary["candidates_by_name"][candidate_name] = {
                    "total_models": 0,
                    "pass_count": 0,
                    "promotion_eligible_count": 0,
                    "models": [],
                }

            summary["candidates_by_name"][candidate_name]["total_models"] += 1
            if candidate_result["status"] == "PASS":
                summary["candidates_by_name"][candidate_name]["pass_count"] += 1
            if candidate_result.get("promotion_eligible", False):
                summary["candidates_by_name"][candidate_name]["promotion_eligible_count"] += 1

            summary["candidates_by_name"][candidate_name]["models"].append({
                "model_id": model_id,
                "status": candidate_result["status"],
                "promotion_eligible": candidate_result.get("promotion_eligible", False),
            })

    # Build promotion matrix
    for candidate_name, candidate_summary in summary["candidates_by_name"].items():
        summary["promotion_matrix"][candidate_name] = {
            "pass_rate": candidate_summary["pass_count"] / candidate_summary["total_models"],
            "promotion_eligible_rate": candidate_summary["promotion_eligible_count"] / candidate_summary["total_models"],
            "recommended": candidate_summary["promotion_eligible_count"] == candidate_summary["total_models"],
        }

    return summary


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-model promotion matrix"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "mlx-community/Qwen2.5-1.5B-Instruct",
            "mlx-community/Qwen2.5-3B-Instruct",
        ],
        help="Models to test",
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
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/promotion_matrix.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Multi-Model Promotion Matrix ===")
    print(f"Models: {args.models}")
    print(f"Prompt: {args.prompt}")

    results = run_multi_model_promotion_matrix(
        args.models, args.prompt, args.max_tokens
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Matrix skipped: MLX not available")
        return 1

    summary = results.get("summary", {})
    print(f"\nPromotion matrix summary:")
    for candidate_name, matrix_info in summary["promotion_matrix"].items():
        recommended = " ✓" if matrix_info["recommended"] else ""
        print(f"  {candidate_name}:")
        print(f"    Pass rate: {matrix_info['pass_rate']:.2%}")
        print(f"    Promotion eligible rate: {matrix_info['promotion_eligible_rate']:.2%}{recommended}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
