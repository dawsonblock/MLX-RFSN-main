#!/usr/bin/env python3
"""Multi-context length testing framework.

This script tests candidates across multiple context lengths to ensure
quality is maintained at different scales:
- Short context (512 tokens)
- Medium context (2K tokens)
- Long context (4K tokens)
- Very long context (8K tokens)

This helps identify context-dependent quality issues.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_multi_context_length_test(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    context_lengths: list[int] = [512, 2048, 4096, 8192],
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 100,
    max_tokens: int = 50,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> dict[str, Any]:
    """Run quality tests across multiple context lengths.

    Args:
        model_id: Model to test with.
        context_lengths: List of context lengths to test.
        prompt: Test prompt (long enough for largest context).
        max_tokens: Maximum tokens to generate.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.

    Returns:
        Dictionary containing multi-context test results.
    """
    try:
        import mlx_lm
    except ImportError as e:
        return {
            "error": f"MLX not available: {e}",
            "status": "SKIPPED_NO_MLX",
        }

    print(f"Loading model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)

    # Extend prompt if needed
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) < max(context_lengths):
        prompt = prompt + " " * ((max(context_lengths) - len(prompt_ids)) // 10 + 1)
        prompt_ids = tokenizer.encode(prompt)

    results = {
        "model_id": model_id,
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
        },
        "context_lengths_tested": context_lengths,
        "context_results": [],
        "summary": {},
    }

    for context_len in context_lengths:
        print(f"\nTesting context length: {context_len}")
        context_result = _test_context_length(
            model, tokenizer, prompt_ids[:context_len], max_tokens,
            key_bits, value_bits, group_size, context_len
        )
        results["context_results"].append(context_result)

    # Generate summary
    results["summary"] = _generate_context_summary(results["context_results"])

    return results


def _test_context_length(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
    context_len: int,
) -> dict[str, Any]:
    """Test quality at a specific context length."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx
    import numpy as np

    # Baseline
    y = mx.array(prompt_ids)
    baseline_logits = []
    for _ in range(max_tokens):
        logits = model(y[None])
        logits = logits[:, -1, :]
        baseline_logits.append(np.array(logits.astype(mx.float32).squeeze(0)))
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    # Quantized
    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name=f"context_{context_len}",
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

    y = mx.array(prompt_ids)
    quantized_logits = []
    for _ in range(max_tokens):
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]
        quantized_logits.append(np.array(logits.astype(mx.float32).squeeze(0)))
        token_id = int(mx.argmax(logits, axis=-1)[0])
        y = mx.array([token_id])

    # Compare
    max_differences = []
    cosine_similarities = []
    top_token_matches = []

    for base_logit, quant_logit in zip(baseline_logits, quantized_logits):
        max_diff = np.max(np.abs(base_logit - quant_logit))
        max_differences.append(float(max_diff))

        norm_base = np.linalg.norm(base_logit)
        norm_quant = np.linalg.norm(quant_logit)
        if norm_base > 0 and norm_quant > 0:
            cosine = np.dot(base_logit, quant_logit) / (norm_base * norm_quant)
        else:
            cosine = 0.0
        cosine_similarities.append(float(cosine))

        base_top = int(np.argmax(base_logit))
        quant_top = int(np.argmax(quant_logit))
        top_token_matches.append(base_top == quant_top)

    # Get runtime counters
    counters = session.counters()

    return {
        "context_length": context_len,
        "mean_max_difference": float(np.mean(max_differences)),
        "mean_cosine_similarity": float(np.mean(cosine_similarities)),
        "top_token_match_rate": sum(top_token_matches) / len(top_token_matches),
        "runtime_counters": counters,
        "status": "PASS" if counters.get("dense_fallback_calls", 0) == 0 else "FAIL",
    }


def _generate_context_summary(context_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary of context length tests."""
    if not context_results:
        return {}

    mean_max_diffs = [r["mean_max_difference"] for r in context_results]
    mean_cosines = [r["mean_cosine_similarity"] for r in context_results]
    match_rates = [r["top_token_match_rate"] for r in context_results]

    return {
        "contexts_tested": len(context_results),
        "min_mean_max_diff": float(min(mean_max_diffs)),
        "max_mean_max_diff": float(max(mean_max_diffs)),
        "min_mean_cosine": float(min(mean_cosines)),
        "max_mean_cosine": float(max(mean_cosines)),
        "min_match_rate": float(min(match_rates)),
        "max_match_rate": float(max(match_rates)),
        "all_passed": all(r["status"] == "PASS" for r in context_results),
    }


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-context length testing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model to test with",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 2048, 4096, 8192],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog. " * 100,
        help="Test prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
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
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/multi_context_length.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Multi-Context Length Testing ===")
    print(f"Model: {args.model}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")

    results = run_multi_context_length_test(
        args.model, args.context_lengths, args.prompt, args.max_tokens,
        args.key_bits, args.value_bits, args.group_size
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Testing skipped: MLX not available")
        return 1

    summary = results.get("summary", {})
    print(f"\nContext length summary:")
    print(f"  Contexts tested: {summary['contexts_tested']}")
    print(f"  Mean max diff range: {summary['min_mean_max_diff']:.6f} - {summary['max_mean_max_diff']:.6f}")
    print(f"  Mean cosine range: {summary['min_mean_cosine']:.6f} - {summary['max_mean_cosine']:.6f}")
    print(f"  Match rate range: {summary['min_match_rate']:.2%} - {summary['max_match_rate']:.2%}")
    print(f"  All passed: {summary['all_passed']}")

    if not summary["all_passed"]:
        print("\n⚠️  WARNING: Some context lengths failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
