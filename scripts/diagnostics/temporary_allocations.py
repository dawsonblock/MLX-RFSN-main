#!/usr/bin/env python3
"""Temporary allocation tracking script.

This script tracks temporary allocations that should be eliminated:
- Full-cache temporary reconstructions
- Dense fallback allocations
- Scratch buffer over-allocation
- Staging buffer leaks

The goal is to identify and eliminate unnecessary temporary allocations
that increase memory pressure and reduce performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def track_temporary_allocations(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 10,
    max_tokens: int = 100,
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> dict[str, Any]:
    """Track temporary allocations during generation.

    Args:
        model_id: Model to test with.
        prompt: Test prompt.
        max_tokens: Maximum tokens to generate.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.

    Returns:
        Dictionary containing allocation tracking results.
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
        "allocation_events": [],
        "allocation_summary": {},
    }

    # Track allocations
    print("Tracking allocations...")
    allocation_events = _track_allocations(
        model, tokenizer, prompt, max_tokens,
        key_bits, value_bits, group_size
    )

    results["allocation_events"] = allocation_events
    results["allocation_summary"] = _analyze_allocations(allocation_events)

    return results


def _track_allocations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    key_bits: int,
    value_bits: int,
    group_size: int,
) -> list[dict[str, Any]]:
    """Track allocation events during generation."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx

    key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
    value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

    session = GenerationCacheSession(
        name="allocation_tracking",
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
    events = []

    # Prefill
    mx.metal.reset_peak_memory()
    prefill_peak = mx.metal.get_peak_memory() / (1024 ** 2)

    y = mx.array(prompt_ids)
    _ = model(y[None], cache=cache_list)

    prefill_final_peak = mx.metal.get_peak_memory() / (1024 ** 2)
    events.append({
        "stage": "prefill",
        "peak_before_mb": float(prefill_peak),
        "peak_after_mb": float(prefill_final_peak),
        "delta_mb": float(prefill_final_peak - prefill_peak),
    })

    # Decode
    for token_idx in range(max_tokens):
        mx.metal.reset_peak_memory()
        decode_peak = mx.metal.get_peak_memory() / (1024 ** 2)

        _ = model(y[None], cache=cache_list)

        decode_final_peak = mx.metal.get_peak_memory() / (1024 ** 2)
        events.append({
            "stage": f"decode_{token_idx}",
            "peak_before_mb": float(decode_peak),
            "peak_after_mb": float(decode_final_peak),
            "delta_mb": float(decode_final_peak - decode_peak),
        })

        token_id = int(mx.argmax(_[:, -1, :], axis=-1)[0])
        y = mx.array([token_id])

    # Get session counters
    counters = session.counters()
    events.append({
        "stage": "runtime_counters",
        "counters": counters,
    })

    return events


def _analyze_allocations(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze allocation patterns."""
    decode_events = [e for e in events if e["stage"].startswith("decode_")]

    if not decode_events:
        return {"error": "No decode events"}

    deltas = [e["delta_mb"] for e in decode_events]
    prefill_event = next((e for e in events if e["stage"] == "prefill"), None)

    runtime_counters = next((e for e in events if e["stage"] == "runtime_counters"), {})
    counters = runtime_counters.get("counters", {})

    return {
        "prefill_delta_mb": float(prefill_event["delta_mb"]) if prefill_event else 0.0,
        "decode_mean_delta_mb": float(sum(deltas) / len(deltas)),
        "decode_max_delta_mb": float(max(deltas)),
        "decode_min_delta_mb": float(min(deltas)),
        "decode_std_delta_mb": float(_std_dev(deltas)),
        "total_decode_delta_mb": float(sum(deltas)),
        "dense_fallback_calls": counters.get("dense_fallback_calls", 0),
        "packed_attention_calls": counters.get("packed_attention_calls", 0),
        "scratch_bytes_peak": counters.get("scratch_bytes_peak", 0),
        "decoded_block_bytes": counters.get("decoded_block_bytes", 0),
        "packed_blocks_created": counters.get("packed_blocks_created", 0),
    }


def _std_dev(values: list[float]) -> float:
    """Calculate standard deviation."""
    import math
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Track temporary allocations"
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
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/temporary_allocations.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== Temporary Allocation Tracking ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")

    results = track_temporary_allocations(
        args.model, args.prompt, args.max_tokens,
        args.key_bits, args.value_bits, args.group_size
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Tracking skipped: MLX not available")
        return 1

    summary = results.get("allocation_summary", {})
    print(f"\nAllocation summary:")
    print(f"  Prefill delta: {summary['prefill_delta_mb']:.2f} MB")
    print(f"  Decode mean delta: {summary['decode_mean_delta_mb']:.2f} MB")
    print(f"  Decode max delta: {summary['decode_max_delta_mb']:.2f} MB")
    print(f"  Total decode delta: {summary['total_decode_delta_mb']:.2f} MB")
    print(f"  Dense fallback calls: {summary['dense_fallback_calls']}")
    print(f"  Packed attention calls: {summary['packed_attention_calls']}")
    print(f"  Scratch bytes peak: {summary['scratch_bytes_peak']:,} bytes")
    print(f"  Packed blocks created: {summary['packed_blocks_created']}")

    if summary["dense_fallback_calls"] > 0:
        print("\n⚠️  WARNING: Dense fallback allocations detected")
        print("This indicates temporary full-cache reconstructions are occurring.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
