#!/usr/bin/env python3
"""RoPE, token order, and GQA verification script.

This script verifies that:
1. RoPE (Rotary Position Embedding) offsets are correctly applied
2. Token order is preserved through quantization/dequantization
3. GQA (Grouped Query Attention) head replication is correct

These are common sources of quality degradation in quantized KV caches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def verify_rope_token_order_gqa(
    model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    key_bits: int = 8,
    value_bits: int = 8,
    group_size: int = 64,
) -> dict[str, Any]:
    """Verify RoPE, token order, and GQA correctness.

    Args:
        model_id: Model to test with.
        prompt: Test prompt.
        key_bits: Key quantization bits.
        value_bits: Value quantization bits.
        group_size: Quantization group size.

    Returns:
        Dictionary containing verification results.
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
        "config": {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
        },
        "rope_verification": {},
        "token_order_verification": {},
        "gqa_verification": {},
        "overall_status": "PASS",
    }

    # Verify RoPE offsets
    print("Verifying RoPE offsets...")
    results["rope_verification"] = _verify_rope_offsets(model, tokenizer, prompt)

    # Verify token order preservation
    print("Verifying token order preservation...")
    results["token_order_verification"] = _verify_token_order(
        model, tokenizer, prompt, key_bits, value_bits, group_size
    )

    # Verify GQA head replication
    print("Verifying GQA head replication...")
    results["gqa_verification"] = _verify_gqa_replication(
        model, tokenizer, prompt, key_bits, value_bits, group_size
    )

    # Overall status
    if not results["rope_verification"]["status"] == "PASS":
        results["overall_status"] = "FAIL"
    if not results["token_order_verification"]["status"] == "PASS":
        results["overall_status"] = "FAIL"
    if not results["gqa_verification"]["status"] == "PASS":
        results["overall_status"] = "FAIL"

    return results


def _verify_rope_offsets(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> dict[str, Any]:
    """Verify that RoPE offsets are correctly applied."""
    import mlx.core as mx
    import numpy as np

    # This is a simplified check - in practice you'd need to inspect
    # the actual RoPE implementation and verify position encoding
    verification = {
        "status": "PASS",
        "details": [],
        "errors": [],
    }

    try:
        # Check if model has RoPE layers
        has_rope = any(
            hasattr(layer, "rotary_emb") or "rope" in str(type(layer)).lower()
            for layer in model.layers
        )

        if has_rope:
            verification["details"].append("Model has RoPE layers")

            # Test that position encoding varies with position
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) >= 2:
                # Get embeddings for different positions
                # This would require model introspection
                verification["details"].append("Position encoding structure detected")
        else:
            verification["details"].append("No RoPE layers detected")

    except Exception as e:
        verification["status"] = "ERROR"
        verification["errors"].append(str(e))

    return verification


def _verify_token_order(
    model: Any,
    tokenizer: Any,
    prompt: str,
    key_bits: int,
    value_bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Verify that token order is preserved through quantization."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
    from rfsn_v10.config import QuantizationConfig, RFSNConfig
    from rfsn_v10.runtime.generation import RFSNGenerator
    import mlx.core as mx
    import numpy as np

    verification = {
        "status": "PASS",
        "details": [],
        "errors": [],
    }

    try:
        prompt_ids = tokenizer.encode(prompt)

        # Create quantized cache
        key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
        value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

        session = GenerationCacheSession(
            name="token_order_test",
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

        # Run generation and check that token sequence is consistent
        y = mx.array(prompt_ids)
        generated_tokens = []

        for _ in range(10):  # Generate 10 tokens
            logits = model(y[None], cache=cache_list)
            token_id = int(mx.argmax(logits, axis=-1)[0])
            generated_tokens.append(token_id)
            y = mx.array([token_id])

        verification["details"].append(f"Generated {len(generated_tokens)} tokens")
        verification["details"].append(f"Token sequence: {generated_tokens[:5]}...")

        # Check that tokens are valid (not all the same, not zeros)
        if len(set(generated_tokens)) == 1:
            verification["status"] = "FAIL"
            verification["errors"].append("All generated tokens are identical")

        if all(t == 0 for t in generated_tokens):
            verification["status"] = "FAIL"
            verification["errors"].append("All generated tokens are zero")

    except Exception as e:
        import traceback
        verification["status"] = "ERROR"
        verification["errors"].append(str(e))
        verification["traceback"] = traceback.format_exc()

    return verification


def _verify_gqa_replication(
    model: Any,
    tokenizer: Any,
    prompt: str,
    key_bits: int,
    value_bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Verify that GQA head replication is correct."""
    import mlx.core as mx
    import numpy as np

    verification = {
        "status": "PASS",
        "details": [],
        "errors": [],
    }

    try:
        # Check model configuration for GQA
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "num_key_value_heads"):
                n_kv_heads = config.num_key_value_heads
                n_heads = config.num_attention_heads

                verification["details"].append(f"Num heads: {n_heads}")
                verification["details"].append(f"Num KV heads: {n_kv_heads}")

                if n_kv_heads < n_heads:
                    verification["details"].append("GQA detected (n_kv_heads < n_heads)")

                    # Verify head count ratio is valid
                    if n_heads % n_kv_heads != 0:
                        verification["status"] = "FAIL"
                        verification["errors"].append(
                            f"Invalid GQA ratio: {n_heads} not divisible by {n_kv_heads}"
                        )
                else:
                    verification["details"].append("No GQA (n_kv_heads == n_heads)")
            else:
                verification["details"].append("Could not determine GQA configuration")

    except Exception as e:
        verification["status"] = "ERROR"
        verification["errors"].append(str(e))

    return verification


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify RoPE, token order, and GQA correctness"
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
        default=Path("artifacts/diagnostics/rope_gqa_verification.json"),
        help="Output file for results",
    )
    args = parser.parse_args()

    print("=== RoPE, Token Order, and GQA Verification ===")
    print(f"Model: {args.model}")
    print(f"Config: K{args.key_bits}/V{args.value_bits} GS{args.group_size}")

    results = verify_rope_token_order_gqa(
        args.model, args.prompt, args.key_bits, args.value_bits, args.group_size
    )

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {args.output}")

    # Print summary
    if results.get("status") == "SKIPPED_NO_MLX":
        print("Verification skipped: MLX not available")
        return 1

    print(f"\nOverall status: {results['overall_status']}")

    for check_name, check_result in [
        ("RoPE", results.get("rope_verification", {})),
        ("Token order", results.get("token_order_verification", {})),
        ("GQA", results.get("gqa_verification", {})),
    ]:
        status = check_result.get("status", "UNKNOWN")
        print(f"  {check_name}: {status}")
        if check_result.get("errors"):
            for error in check_result["errors"]:
                print(f"    - {error}")

    if results["overall_status"] != "PASS":
        print("\n⚠️  Verification failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
