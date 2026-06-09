"""
Memory and speed benchmark: TurboQuant vs baselines.
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache
from mlx_turboquant.cache import TurboQuantKVCache


def benchmark_memory(model_id):
    """Measure actual KV cache memory at various context lengths."""
    print(f"\n{'='*70}")
    print(f"Memory Benchmark: {model_id}")
    print(f"{'='*70}")

    model, tokenizer = load(model_id)
    args = model.args
    head_dim = getattr(args, 'head_dim', args.hidden_size // args.num_attention_heads)
    num_layers = len(model.layers)
    n_kv_heads = getattr(args, 'num_key_value_heads', args.num_attention_heads)

    print(f"Layers={num_layers}, head_dim={head_dim}, n_kv_heads={n_kv_heads}\n")

    # Generate a long prompt by repeating
    base = "The quick brown fox jumps over the lazy dog. "
    seq_lens = [128, 512, 1024]

    print(f"{'Seq':>6} | {'FP16 KB':>9} | {'TQ3 KB':>9} | {'TQ3.5 KB':>9} | {'TQ4 KB':>9} | {'TQ3 ratio':>9} | {'TQ4 ratio':>9}")
    print("-" * 80)

    for target_len in seq_lens:
        text = base * (target_len // 10 + 1)
        tokens = mx.array([tokenizer.encode(text)[:target_len]])
        actual_len = tokens.shape[1]

        # FP16
        fp16_c = [mlx_cache.KVCache() for _ in range(num_layers)]
        model(tokens, cache=fp16_c)
        mx.eval(*[c.keys for c in fp16_c if c.keys is not None])
        fp16_kb = sum(c.nbytes for c in fp16_c) / 1024

        results = {}
        for bits in [3, 3.5, 4]:
            tq_c = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]
            model(tokens, cache=tq_c)
            for c in tq_c:
                if c._key_indices is not None:
                    mx.eval(c._key_indices)
            tq_kb = sum(c.nbytes for c in tq_c) / 1024
            results[bits] = tq_kb

        r3 = fp16_kb / max(results[3], 0.1)
        r4 = fp16_kb / max(results[4], 0.1)
        print(f"{actual_len:>6} | {fp16_kb:>9.0f} | {results[3]:>9.0f} | {results[3.5]:>9.0f} | {results[4]:>9.0f} | {r3:>8.1f}x | {r4:>8.1f}x")


def benchmark_speed(model_id, n_tokens=50):
    """Measure generation speed."""
    print(f"\n{'='*70}")
    print(f"Speed Benchmark: {model_id}")
    print(f"{'='*70}")

    model, tokenizer = load(model_id)
    args = model.args
    head_dim = getattr(args, 'head_dim', args.hidden_size // args.num_attention_heads)
    num_layers = len(model.layers)

    prompt = "Write a detailed explanation of how neural networks work. Start from the basics of neurons and perceptrons, then explain"
    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_len = tokens.shape[1]
    print(f"Prompt: {prompt_len} tokens, generating {n_tokens} tokens\n")

    configs = [
        ("FP16", lambda: [mlx_cache.KVCache() for _ in range(num_layers)]),
        ("TQ 3-bit", lambda: [TurboQuantKVCache(bits=3, head_dim=head_dim) for _ in range(num_layers)]),
        ("TQ 3.5-bit", lambda: [TurboQuantKVCache(bits=3.5, head_dim=head_dim) for _ in range(num_layers)]),
        ("TQ 4-bit", lambda: [TurboQuantKVCache(bits=4, head_dim=head_dim) for _ in range(num_layers)]),
    ]

    print(f"{'Method':<12} | {'Prefill':>10} | {'Decode':>10} | {'Cache KB':>10}")
    print("-" * 55)

    for name, make_cache in configs:
        cache = make_cache()
        mx.eval(model.parameters())

        t0 = time.perf_counter()
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        prefill = prompt_len / (time.perf_counter() - t0)

        next_token = mx.argmax(logits[0, -1], keepdims=True)
        t0 = time.perf_counter()
        for _ in range(n_tokens - 1):
            logits = model(next_token[None, :], cache=cache)
            next_token = mx.argmax(logits[0, -1], keepdims=True)
            mx.eval(next_token)
        decode = (n_tokens - 1) / (time.perf_counter() - t0)

        cache_kb = sum(c.nbytes for c in cache) / 1024
        print(f"{name:<12} | {prefill:>8.0f} t/s | {decode:>8.1f} t/s | {cache_kb:>10.0f}")


if __name__ == "__main__":
    for model_id in ["mlx-community/Llama-3.2-1B-Instruct-4bit",
                      "mlx-community/Llama-3.2-3B-Instruct-4bit"]:
        benchmark_memory(model_id)
        benchmark_speed(model_id, n_tokens=50)
