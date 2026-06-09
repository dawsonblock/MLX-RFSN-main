"""
Full benchmark suite for TurboQuant report.

Tests quality, memory, and speed across legacy and SOTA 2026 models.
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache
from mlx_turboquant.cache import TurboQuantKVCache

MODELS = [
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    "The following is a list of world capitals:\n1. France - Paris\n2. Germany - Berlin\n3. Japan - Tokyo\n4. Brazil - Brasilia\n5. Australia - Canberra\n6. India -",
    "In machine learning, the transformer architecture was introduced in the paper titled 'Attention is All You",
]

BIT_CONFIGS = [2, 3, 3.5, 4]


def run_quality(model, tokenizer, head_dim, num_layers, model_name):
    """Quality: logit cosine similarity and top-k accuracy."""
    results = []

    for prompt in PROMPTS:
        tokens = mx.array([tokenizer.encode(prompt)])
        n_tok = tokens.shape[1]

        fp16_cache = [mlx_cache.KVCache() for _ in range(num_layers)]
        logits_fp16 = model(tokens, cache=fp16_cache)[0, -1].astype(mx.float32)
        mx.eval(logits_fp16)
        fp16_top5 = mx.argsort(logits_fp16)[-5:][::-1]
        mx.eval(fp16_top5)
        fp16_top5_set = set(int(t) for t in fp16_top5)
        fp16_top1 = int(fp16_top5[0])

        for bits in BIT_CONFIGS:
            tq_cache = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]
            logits_tq = model(tokens, cache=tq_cache)[0, -1].astype(mx.float32)
            mx.eval(logits_tq)
            cos = float(mx.sum(logits_fp16 * logits_tq) / (mx.linalg.norm(logits_fp16) * mx.linalg.norm(logits_tq) + 1e-8))
            tq_top5 = mx.argsort(logits_tq)[-5:][::-1]
            mx.eval(tq_top5)
            top1_match = int(tq_top5[0]) == fp16_top1
            top5_overlap = len(set(int(t) for t in tq_top5) & fp16_top5_set)

            results.append({
                "model": model_name, "prompt_len": n_tok, "bits": bits,
                "cosine": cos, "top1_match": top1_match, "top5_overlap": top5_overlap,
            })

    return results


def run_memory(model, tokenizer, head_dim, num_layers, model_name):
    """Memory: actual packed cache bytes at various seq lengths."""
    results = []
    base = "The quick brown fox jumps over the lazy dog. "

    for target_len in [128, 512, 1024]:
        text = base * (target_len // 10 + 1)
        tokens = mx.array([tokenizer.encode(text)[:target_len]])
        actual_len = tokens.shape[1]

        fp16_c = [mlx_cache.KVCache() for _ in range(num_layers)]
        model(tokens, cache=fp16_c)
        mx.eval(*[c.keys for c in fp16_c if c.keys is not None])
        fp16_kb = sum(c.nbytes for c in fp16_c) / 1024

        for bits in [3, 3.5, 4]:
            tq_c = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]
            model(tokens, cache=tq_c)
            for c in tq_c:
                if c._key_indices is not None:
                    mx.eval(c._key_indices)
            tq_kb = sum(c.nbytes for c in tq_c) / 1024

            results.append({
                "model": model_name, "seq_len": actual_len, "bits": bits,
                "fp16_kb": fp16_kb, "tq_kb": tq_kb,
                "ratio": fp16_kb / max(tq_kb, 0.1),
            })

    return results


def run_speed(model, tokenizer, head_dim, num_layers, model_name, n_tokens=30):
    """Speed: prefill and decode tok/s."""
    results = []
    prompt = "Write a detailed explanation of how neural networks work. Start from the basics."
    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_len = tokens.shape[1]

    for name, make_cache in [
        ("FP16", lambda: [mlx_cache.KVCache() for _ in range(num_layers)]),
        ("TQ-3", lambda: [TurboQuantKVCache(bits=3, head_dim=head_dim) for _ in range(num_layers)]),
        ("TQ-3.5", lambda: [TurboQuantKVCache(bits=3.5, head_dim=head_dim) for _ in range(num_layers)]),
        ("TQ-4", lambda: [TurboQuantKVCache(bits=4, head_dim=head_dim) for _ in range(num_layers)]),
    ]:
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
        results.append({
            "model": model_name, "method": name,
            "prefill_tps": prefill, "decode_tps": decode, "cache_kb": cache_kb,
        })

    return results


def main():
    all_quality = []
    all_memory = []
    all_speed = []

    for model_id, model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} ({model_id})")
        print(f"{'='*60}")

        model, tokenizer = load(model_id)
        args = model.args
        head_dim = getattr(args, 'head_dim', args.hidden_size // args.num_attention_heads)
        num_layers = len(model.layers)
        n_kv = getattr(args, 'num_key_value_heads', args.num_attention_heads)
        print(f"layers={num_layers}, head_dim={head_dim}, n_kv_heads={n_kv}")

        print("  Running quality...")
        all_quality.extend(run_quality(model, tokenizer, head_dim, num_layers, model_name))
        print("  Running memory...")
        all_memory.extend(run_memory(model, tokenizer, head_dim, num_layers, model_name))
        print("  Running speed...")
        all_speed.extend(run_speed(model, tokenizer, head_dim, num_layers, model_name))

        del model, tokenizer
        mx.clear_cache()

    # Print summary tables
    print("\n\n" + "="*80)
    print("QUALITY: Logit Cosine Similarity vs FP16 (averaged over 4 prompts)")
    print("="*80)
    print(f"{'Model':<22} | {'2-bit':>7} | {'3-bit':>7} | {'3.5-bit':>7} | {'4-bit':>7} | {'Top-1 @3':>8} | {'Top-1 @4':>8}")
    print("-"*80)
    for model_id, model_name in MODELS:
        row = {}
        top1_counts = {}
        for r in all_quality:
            if r["model"] == model_name:
                b = r["bits"]
                row.setdefault(b, []).append(r["cosine"])
                top1_counts.setdefault(b, []).append(r["top1_match"])
        avgs = {b: sum(v)/len(v) for b, v in row.items()}
        t1_3 = sum(top1_counts.get(3, [])); t1_4 = sum(top1_counts.get(4, []))
        print(f"{model_name:<22} | {avgs.get(2,0):>7.4f} | {avgs.get(3,0):>7.4f} | {avgs.get(3.5,0):>7.4f} | {avgs.get(4,0):>7.4f} | {t1_3:>5}/4   | {t1_4:>5}/4  ")

    print("\n\n" + "="*80)
    print("MEMORY: KV Cache Size (KB) at 1024 tokens")
    print("="*80)
    print(f"{'Model':<22} | {'FP16':>9} | {'TQ 3-bit':>9} | {'3-bit ratio':>11} | {'TQ 4-bit':>9} | {'4-bit ratio':>11}")
    print("-"*80)
    for model_id, model_name in MODELS:
        for r in all_memory:
            if r["model"] == model_name and r["seq_len"] >= 1000 and r["bits"] == 3:
                fp16 = r["fp16_kb"]
                tq3 = r["tq_kb"]
                r3 = r["ratio"]
        for r in all_memory:
            if r["model"] == model_name and r["seq_len"] >= 1000 and r["bits"] == 4:
                tq4 = r["tq_kb"]
                r4 = r["ratio"]
        print(f"{model_name:<22} | {fp16:>9.0f} | {tq3:>9.0f} | {r3:>10.1f}x | {tq4:>9.0f} | {r4:>10.1f}x")

    print("\n\n" + "="*80)
    print("SPEED: Tokens/sec (decode)")
    print("="*80)
    print(f"{'Model':<22} | {'FP16':>8} | {'TQ 3-bit':>8} | {'TQ 3.5':>8} | {'TQ 4-bit':>8} | {'TQ3/FP16':>8}")
    print("-"*80)
    for model_id, model_name in MODELS:
        speeds = {}
        for r in all_speed:
            if r["model"] == model_name:
                speeds[r["method"]] = r["decode_tps"]
        fp = speeds.get("FP16", 0)
        t3 = speeds.get("TQ-3", 0)
        t35 = speeds.get("TQ-3.5", 0)
        t4 = speeds.get("TQ-4", 0)
        ratio = t3 / fp if fp > 0 else 0
        print(f"{model_name:<22} | {fp:>6.1f} | {t3:>8.1f} | {t35:>8.1f} | {t4:>8.1f} | {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
