#!/usr/bin/env python3
"""Multi-context memory measurement script.

This script measures incremental KV memory at meaningful contexts:
512, 2K, 4K, 8K, 16K, 32K tokens.

For each context:
1. Load model once
2. Stabilize allocator
3. Reset peak counters
4. Run baseline
5. Tear down cache
6. Reset counters
7. Run candidate
8. Record incremental cache and scratch deltas

Compression should be evaluated primarily at contexts where KV memory is material.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def measure_multi_context_memory(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    contexts: list[int] = [512, 2048, 4096, 8192, 16384, 32768],
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 100,  # Long prompt to support large contexts
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> dict[str, Any]:
    """Measure memory at multiple context lengths.

    Args:
        model_id: Model to test with.
        contexts: List of context lengths to test.
        prompt: Test prompt (long enough for largest context).
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.

    Returns:
        Dictionary containing memory measurements for each context.
    """
    try:
        import mlx_lm
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
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
        },
        "contexts_tested": contexts,
        "context_measurements": [],
        "summary": {},
    }

    # Truncate prompt to fit smallest context
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) < max(contexts):
        # Extend prompt if needed
        prompt = prompt + " " * ((max(contexts) - len(prompt_ids)) // 10 + 1)
        prompt_ids = tokenizer.encode(prompt)

    for context_len in contexts:
        print(f"\nMeasuring context: {context_len} tokens")
        context_result = _measure_context(
            model, tokenizer, prompt_ids[:context_len],
            key_bits, value_bits, group_size, context_len
        )
        results["context_measurements"].append(context_result)

    # Generate summary
    results["summary"] = _generate_memory_summary(results["context_measurements"])

    return results


def _measure_context(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    key_bits: int,
    value_bits: int,
    group_size: int,
    context_len: int,
) -> dict[str, Any]:
    """Measure memory for a single context length."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx

    # Stabilize allocator
    _ = model(mx.array([1]))

    # Baseline measurement
    mx.metal.reset_peak_memory()
    baseline_peak = mx.metal.get_peak_memory() / (1024 ** 2)

    # Run baseline
    y = mx.array(prompt_ids)
    for _ in range(10):  # Generate 10 tokens
        _ = model(y[None])
        token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
        y = mx.array([token_id])

    baseline_final_peak = mx.metal.get_peak_memory() / (1024 ** 2)
    baseline_delta = baseline_final_peak - baseline_peak

    # Clear cache
    del _

    # Stabilize allocator again
    _ = model(mx.array([1]))
    mx.metal.reset_peak_memory()

    # Quantized measurement
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

    quantized_peak = mx.metal.get_peak_memory() / (1024 ** 2)

    # Run quantized
    y = mx.array(prompt_ids)
    for _ in range(10):
        _ = model(y[None], cache=cache_list)
        token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
        y = mx.array([token_id])

    quantized_final_peak = mx.metal.get_peak_memory() / (1024 ** 2)
    quantized_delta = quantized_final_peak - quantized_peak

    # Get runtime counters
    counters = session.counters()

    return {
        "context_length": context_len,
        "baseline_peak_mb": float(baseline_peak),
        "baseline_delta_mb": float(baseline_delta),
        "quantized_peak_mb": float(quantized_peak),
        "quantized_delta_mb": float(quantized_delta),
        "memory_saving_mb": float(baseline_delta - quantized_delta),
        "memory_ratio": float(quantized_delta / baseline_delta) if baseline_delta > 0 else 0.0,
        "runtime_counters": counters,
    }


def _generate_memory_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary of memory measurements across contexts."""
    if not measurements:
        return {}

    baseline_deltas = [m["baseline_delta_mb"] for m in measurements]
    quantized_deltas = [m["quantized_delta_mb"] for m in measurements]
    savings = [m["memory_saving_mb"] for m in measurements]

    return {
        "contexts_tested": len(measurements),
        "min_baseline_delta_mb": min(baseline_deltas),
        "max_baseline_delta_mb": max(baseline_deltas),
        "min_quantized_delta_mb": min(quantized_deltas),
        "max_quantized_delta_mb": max(quantized_deltas),
        "total_saving_mb": sum(savings),
        "avg_saving_mb": sum(savings) / len(savings),
        "best_saving_context": measurements[savings.index(max(savings))]["context_length"],
        "best_saving_mb": max(savings),
    }


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure memory at multiple context lengths"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model to test with",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[512, 2048, 4096, 8192, 16384, 32768],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog. " * 100,
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
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/multi_context_memory.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Multi-Context Memory Measurement ===")
    print(f"Model: {args.model}")
    print(f"Contexts: {args.contexts}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")

    results = measure_multi_context_memory(
        args.model, args.contexts, args.prompt,
        args.key_bits, args.value_bits, args.group_size
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Measurement skipped: MLX not available")
        return 1

    summary = results.get("summary", {})
    print(f"\nMemory summary:")
    print(f"  Contexts tested: {summary['contexts_tested']}")
    print(f"  Total saving: {summary['total_saving_mb']:.2f} MB")
    print(f"  Average saving: {summary['avg_saving_mb']:.2f} MB")
    print(f"  Best saving: {summary['best_saving_mb']:.2f} MB at context {summary['best_saving_context']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
