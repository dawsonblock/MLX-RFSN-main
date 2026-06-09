"""
End-to-end test: TurboQuant KV cache with real LLM inference.

Compares logit quality and top-k tokens between FP16 and TurboQuant caches.
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import cache as mlx_cache
from mlx_turboquant.cache import TurboQuantKVCache


MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def compare_logits():
    """Compare last-token logits between cache implementations."""
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)

    args = model.args
    head_dim = getattr(args, 'head_dim', args.hidden_size // args.num_attention_heads)
    num_layers = len(model.layers)
    print(f"Model: {MODEL_ID}, layers={num_layers}, head_dim={head_dim}")

    prompt = "The capital of France is"
    tokens = mx.array([tokenizer.encode(prompt)])

    # FP16 baseline
    fp16_cache = [mlx_cache.KVCache() for _ in range(num_layers)]
    logits_fp16 = model(tokens, cache=fp16_cache)[0, -1].astype(mx.float32)
    mx.eval(logits_fp16)
    fp16_top5 = mx.argsort(logits_fp16)[-5:][::-1]
    mx.eval(fp16_top5)
    fp16_bytes = sum(c.nbytes for c in fp16_cache)

    print(f"\n{'Method':<25} {'Cosine':>8} {'Top-1':>10} {'Top-5':>40} {'Cache MB':>10}")
    print("-" * 100)
    print(f"{'FP16 (baseline)':<25} {'1.0000':>8} {tokenizer.decode([int(fp16_top5[0])]):>10} {str([tokenizer.decode([int(t)]) for t in fp16_top5]):>40} {fp16_bytes/1e6:>10.2f}")

    # TurboQuant at various bit widths
    for bits in [2, 3, 4]:
        tq_cache = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]
        logits_tq = model(tokens, cache=tq_cache)[0, -1].astype(mx.float32)
        mx.eval(logits_tq)

        cos = float(mx.sum(logits_fp16 * logits_tq) / (mx.linalg.norm(logits_fp16) * mx.linalg.norm(logits_tq) + 1e-8))
        tq_top5 = mx.argsort(logits_tq)[-5:][::-1]
        mx.eval(tq_top5)
        tq_bytes = sum(c.nbytes for c in tq_cache)

        label = f"TurboQuant {bits}-bit"
        top1 = tokenizer.decode([int(tq_top5[0])])
        top5_str = str([tokenizer.decode([int(t)]) for t in tq_top5])
        print(f"{label:<25} {cos:>8.4f} {top1:>10} {top5_str:>40} {tq_bytes/1e6:>10.2f}")

    # mlx-lm QuantizedKVCache for comparison
    for kv_bits in [4, 8]:
        qkv_cache = [mlx_cache.QuantizedKVCache(bits=kv_bits) for _ in range(num_layers)]
        logits_q = model(tokens, cache=qkv_cache)[0, -1].astype(mx.float32)
        mx.eval(logits_q)
        cos = float(mx.sum(logits_fp16 * logits_q) / (mx.linalg.norm(logits_fp16) * mx.linalg.norm(logits_q) + 1e-8))
        tq_top5 = mx.argsort(logits_q)[-5:][::-1]
        mx.eval(tq_top5)
        try:
            qkv_bytes = sum(c.nbytes for c in qkv_cache)
        except Exception:
            qkv_bytes = 0  # mlx-lm tree_reduce bug
        label = f"mlx-lm Quant {kv_bits}-bit"
        top1 = tokenizer.decode([int(tq_top5[0])])
        top5_str = str([tokenizer.decode([int(t)]) for t in tq_top5])
        print(f"{label:<25} {cos:>8.4f} {top1:>10} {top5_str:>40} {qkv_bytes/1e6:>10.2f}")


def compare_generation():
    """Side-by-side text generation comparison."""
    print("\n\nLoading model for generation...")
    model, tokenizer = load(MODEL_ID)

    prompt = "Explain quantum computing in one sentence:"
    print(f"Prompt: '{prompt}'\n")

    print("[FP16 cache]")
    print(f"  {generate(model, tokenizer, prompt=prompt, max_tokens=40)}\n")

    print("[mlx-lm Quant 4-bit cache]")
    print(f"  {generate(model, tokenizer, prompt=prompt, max_tokens=40, kv_bits=4)}\n")


if __name__ == "__main__":
    compare_logits()
    compare_generation()
