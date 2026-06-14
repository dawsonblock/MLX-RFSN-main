#!/usr/bin/env python3
"""Teacher-forced and free-generation test framework.

This script tests candidates in two modes:
1. Teacher-forced: Ground-truth tokens are forced, measuring log-prob accuracy
2. Free-generation: Model generates freely, measuring output quality

This helps distinguish between:
- Quality issues in the attention computation itself
- Quality issues in generation sampling
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_teacher_forced_and_free_generation(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    target_text: str = "The lazy dog slept peacefully.",
    max_tokens: int = 50,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> dict[str, Any]:
    """Run teacher-forced and free-generation tests.

    Args:
        model_id: Model to test with.
        prompt: Input prompt.
        target_text: Target text for teacher-forced generation.
        max_tokens: Maximum tokens to generate.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.

    Returns:
        Dictionary containing test results for both modes.
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

    results = {
        "model_id": model_id,
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
        },
        "teacher_forced": {},
        "free_generation": {},
        "comparison": {},
    }

    # Teacher-forced test
    print("\nRunning teacher-forced test...")
    teacher_forced_results = _run_teacher_forced(
        model, tokenizer, prompt, target_text, max_tokens,
        key_bits, value_bits, group_size
    )
    results["teacher_forced"] = teacher_forced_results

    # Free-generation test
    print("\nRunning free-generation test...")
    free_generation_results = _run_free_generation(
        model, tokenizer, prompt, max_tokens,
        key_bits, value_bits, group_size
    )
    results["free_generation"] = free_generation_results

    # Compare results
    results["comparison"] = _compare_modes(teacher_forced_results, free_generation_results)

    return results


def _run_teacher_forced(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_text: str,
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Run teacher-forced generation test."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx
    import numpy as np

    # Baseline teacher-forced
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)

    # Extract generation tokens
    if len(target_ids) >= len(prompt_ids) and target_ids[:len(prompt_ids)] == prompt_ids:
        gen_ids = target_ids[len(prompt_ids):]
    else:
        gen_ids = target_ids

    if not gen_ids:
        return {"error": "No generation tokens found", "status": "ERROR"}

    baseline_logprobs = []
    y = mx.array(prompt_ids)
    for i, forced_token_id in enumerate(gen_ids):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        baseline_logprobs.append(np.array(logprobs.astype(mx.float32).squeeze(0)))
        y = mx.array([forced_token_id])

    # Quantized teacher-forced
    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name="teacher_forced",
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

    quantized_logprobs = []
    y = mx.array(prompt_ids)
    for i, forced_token_id in enumerate(gen_ids):
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        quantized_logprobs.append(np.array(logprobs.astype(mx.float32).squeeze(0)))
        y = mx.array([forced_token_id])

    # Compare logprobs
    logprob_differences = []
    for base_lp, quant_lp in zip(baseline_logprobs, quantized_logprobs):
        diff = np.max(np.abs(base_lp - quant_lp))
        logprob_differences.append(float(diff))

    counters = session.counters()

    return {
        "status": "PASS" if counters.get("dense_fallback_calls", 0) == 0 else "FAIL",
        "mean_logprob_difference": float(np.mean(logprob_differences)),
        "max_logprob_difference": float(np.max(logprob_differences)),
        "runtime_counters": counters,
    }


def _run_free_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Run free-generation test."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx

    # Baseline free-generation
    prompt_ids = tokenizer.encode(prompt)
    y = mx.array(prompt_ids)
    baseline_tokens = []
    for _ in range(max_tokens):
        logits = model(y[None])
        token_id = int(mx.argmax(logits, axis=-1)[0])
        baseline_tokens.append(token_id)
        y = mx.array([token_id])

    baseline_text = tokenizer.decode(baseline_tokens)

    # Quantized free-generation
    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name="free_generation",
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
    quantized_tokens = []
    for _ in range(max_tokens):
        logits = model(y[None], cache=cache_list)
        token_id = int(mx.argmax(logits, axis=-1)[0])
        quantized_tokens.append(token_id)
        y = mx.array([token_id])

    quantized_text = tokenizer.decode(quantized_tokens)

    # Compare outputs
    token_match_rate = sum(1 for b, q in zip(baseline_tokens, quantized_tokens) if b == q) / len(baseline_tokens)

    counters = session.counters()

    return {
        "status": "PASS" if counters.get("dense_fallback_calls", 0) == 0 else "FAIL",
        "baseline_text": baseline_text,
        "quantized_text": quantized_text,
        "token_match_rate": token_match_rate,
        "runtime_counters": counters,
    }


def _compare_modes(
    teacher_forced: dict[str, Any],
    free_generation: dict[str, Any],
) -> dict[str, Any]:
    """Compare teacher-forced and free-generation results."""
    return {
        "teacher_forced_status": teacher_forced.get("status"),
        "free_generation_status": free_generation.get("status"),
        "both_passed": (
            teacher_forced.get("status") == "PASS" and
            free_generation.get("status") == "PASS"
        ),
        "teacher_forced_mean_diff": teacher_forced.get("mean_logprob_difference"),
        "free_generation_match_rate": free_generation.get("token_match_rate"),
    }


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run teacher-forced and free-generation tests"
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
        default="The quick brown fox jumps over the lazy dog.",
        help="Input prompt",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        default="The lazy dog slept peacefully.",
        help="Target text for teacher-forced generation",
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
        default=Path("artifacts/diagnostics/teacher_forced_free_generation.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Teacher-Forced and Free-Generation Testing ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")

    results = run_teacher_forced_and_free_generation(
        args.model, args.prompt, args.target_text, args.max_tokens,
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

    comparison = results.get("comparison", {})
    print(f"\nComparison:")
    print(f"  Teacher-forced status: {comparison['teacher_forced_status']}")
    print(f"  Free-generation status: {comparison['free_generation_status']}")
    print(f"  Both passed: {comparison['both_passed']}")
    print(f"  Teacher-forced mean diff: {comparison['teacher_forced_mean_diff']:.6f}")
    print(f"  Free-generation match rate: {comparison['free_generation_match_rate']:.2%}")

    if not comparison["both_passed"]:
        print("\n⚠️  WARNING: One or both modes failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
