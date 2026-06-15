"""Benchmark harness for packed blockwise attention vs dense baseline.

Collects timing, memory, and quality evidence required for the promotion gate.

Usage::

    python -m rfsn_v10.benchmarks.packed_attention_bench \
        --model-id qwen2-0.5b \
        --prompt "Hello, world!" \
        --max-tokens 32 \
        --output results.json
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_dense_cache(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int = 1024,
) -> list[Any]:
    """Create dense FP16 KV caches for baseline comparison."""
    caches = []
    for _ in range(num_layers):
        k = mx.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=mx.float16)
        v = mx.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=mx.float16)
        caches.append((k, v))
    return caches


def _run_dense_attention(
    queries: Any,
    k_cache: Any,
    v_cache: Any,
    seq_len: int,
    scale: float,
) -> Any:
    """Run dense attention over the full cache."""
    k = k_cache[:, :, :seq_len, :]
    v = v_cache[:, :, :seq_len, :]
    return mx.fast.scaled_dot_product_attention(
        queries, k, v, scale=scale, mask=None
    )


def benchmark_dense_baseline(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: list[int],
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark dense FP16 attention at various sequence lengths."""
    if not HAS_MLX:
        raise RuntimeError("MLX not installed")

    scale = head_dim ** -0.5
    caches = _make_dense_cache(num_layers, num_heads, num_kv_heads, head_dim)
    queries = mx.random.normal(shape=(1, num_heads, 1, head_dim)).astype(mx.float16)

    results = {}
    for seq_len in seq_lens:
        # Warmup
        for _ in range(warmup):
            for layer_idx in range(num_layers):
                _run_dense_attention(
                    queries, caches[layer_idx][0], caches[layer_idx][1],
                    seq_len, scale,
                )
            mx.eval(queries)

        # Timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for layer_idx in range(num_layers):
                _run_dense_attention(
                    queries, caches[layer_idx][0], caches[layer_idx][1],
                    seq_len, scale,
                )
            mx.eval(queries)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_ms = sum(times) / len(times)
        tokens_per_sec = (num_layers * iterations) / (sum(times) / 1000)
        results[seq_len] = {
            "avg_ms": avg_ms,
            "tokens_per_sec": tokens_per_sec,
        }

    return results


def benchmark_packed_reference(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: list[int],
    key_codec: Any,
    value_codec: Any,
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark packed blockwise attention at various sequence lengths."""
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
    from rfsn_v10.cache.mlx_packed_attention_reference import attend

    if not HAS_MLX:
        raise RuntimeError("MLX not installed")

    scale = head_dim ** -0.5
    queries = mx.random.normal(shape=(1, num_heads, 1, head_dim)).astype(mx.float32)

    results = {}
    for seq_len in seq_lens:
        # Build cache with tokens up to seq_len
        caches = []
        for _ in range(num_layers):
            cache = QuantizedLayerCache(
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=64,
            )
            # Fill with random tokens
            for _ in range(seq_len // 64):
                k = mx.random.normal(shape=(1, num_kv_heads, 64, head_dim)).astype(mx.float32)
                v = mx.random.normal(shape=(1, num_kv_heads, 64, head_dim)).astype(mx.float32)
                cache.append(k, v)
            caches.append(cache)

        # Warmup
        for _ in range(warmup):
            for cache in caches:
                attend(queries, cache, scale=scale, causal=True)
            mx.eval(queries)

        # Timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for cache in caches:
                attend(queries, cache, scale=scale, causal=True)
            mx.eval(queries)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_ms = sum(times) / len(times)
        tokens_per_sec = (num_layers * iterations) / (sum(times) / 1000)

        # Memory accounting
        total_payload = sum(c.payload_bytes() for c in caches)
        total_dense = sum(c.dense_residual_bytes() for c in caches)
        total_staging = sum(c.staging_bytes() for c in caches)

        results[seq_len] = {
            "avg_ms": avg_ms,
            "tokens_per_sec": tokens_per_sec,
            "payload_bytes": total_payload,
            "dense_residual_bytes": total_dense,
            "staging_bytes": total_staging,
            "total_cache_bytes": total_payload + total_dense + total_staging,
        }

    return results


def run_evidence_collection(
    model_id: str,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: list[int] | None = None,
) -> dict[str, Any]:
    """Collect benchmark evidence for the promotion gate.

    Returns a dict that can be serialized to JSON and later loaded into
    CandidateResult for the promotion gate.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512, 1024]

    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    key_codec = CartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    value_codec = CartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    print(f"Collecting evidence for {model_id}...")
    print(f"  Layers: {num_layers}, Heads: {num_heads}/{num_kv_heads}, D: {head_dim}")

    dense_results = benchmark_dense_baseline(
        num_layers, num_heads, num_kv_heads, head_dim, seq_lens
    )
    packed_results = benchmark_packed_reference(
        num_layers, num_heads, num_kv_heads, head_dim, seq_lens,
        key_codec, value_codec,
    )

    # Compute compression and speedup at each sequence length
    comparisons = {}
    for seq_len in seq_lens:
        dense = dense_results[seq_len]
        packed = packed_results[seq_len]

        # Dense memory: 2 * layers * seq_len * kv_heads * head_dim * 2 bytes (FP16)
        dense_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * 2
        packed_bytes = packed["total_cache_bytes"]

        comparisons[seq_len] = {
            "dense_ms": dense["avg_ms"],
            "packed_ms": packed["avg_ms"],
            "speedup": dense["avg_ms"] / packed["avg_ms"] if packed["avg_ms"] > 0 else 0,
            "dense_bytes": dense_bytes,
            "packed_bytes": packed_bytes,
            "size_ratio": packed_bytes / dense_bytes if dense_bytes > 0 else 0,
            "compression_factor": dense_bytes / packed_bytes if packed_bytes > 0 else 0,
            "dense_tps": dense["tokens_per_sec"],
            "packed_tps": packed["tokens_per_sec"],
        }

    evidence = {
        "model_id": model_id,
        "architecture": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "key_bits": 8,
            "value_bits": 5,
            "group_size": 64,
        },
        "dense_results": dense_results,
        "packed_results": packed_results,
        "comparisons": comparisons,
        "codec_signature": "k8_v5_gs64_wht64",
    }

    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark packed attention")
    parser.add_argument("--model-id", default="synthetic_qwen2")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--output", default="packed_attention_evidence.json")
    args = parser.parse_args()

    evidence = run_evidence_collection(
        model_id=args.model_id,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        seq_lens=args.seq_lens,
    )

    with open(args.output, "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    print(f"Evidence written to {args.output}")

    # Print summary
    print("\n--- Summary ---")
    for seq_len, comp in evidence["comparisons"].items():
        print(
            f"seq={seq_len:4d}: "
            f"dense={comp['dense_ms']:7.3f}ms "
            f"packed={comp['packed_ms']:7.3f}ms "
            f"speedup={comp['speedup']:.2f}x "
            f"compress={comp['compression_factor']:.2f}x"
        )


if __name__ == "__main__":
    main()
