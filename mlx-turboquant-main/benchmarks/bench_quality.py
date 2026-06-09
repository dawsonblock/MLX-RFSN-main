"""
Quality benchmark: compare TurboQuant vs baselines on real models.

Metrics:
- Logit cosine similarity vs FP16
- Top-1 / Top-5 accuracy
- Tested across multiple prompts and models
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache
from mlx_turboquant.cache import TurboQuantKVCache


MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    "The following is a list of world capitals:\n1. France - Paris\n2. Germany - Berlin\n3. Japan - Tokyo\n4. Brazil - Brasilia\n5. Australia - Canberra\n6. India -",
    "In machine learning, the transformer architecture was introduced in the paper titled 'Attention is All You",
]


def benchmark_model(model_id):
    print(f"\n{'='*80}")
    print(f"Model: {model_id}")
    print(f"{'='*80}")

    model, tokenizer = load(model_id)
    args = model.args
    head_dim = getattr(args, 'head_dim', args.hidden_size // args.num_attention_heads)
    num_layers = len(model.layers)
    print(f"Layers: {num_layers}, head_dim: {head_dim}\n")

    for prompt in PROMPTS:
        tokens = mx.array([tokenizer.encode(prompt)])
        n_tokens = tokens.shape[1]

        # FP16 baseline
        fp16_cache = [mlx_cache.KVCache() for _ in range(num_layers)]
        logits_fp16 = model(tokens, cache=fp16_cache)[0, -1].astype(mx.float32)
        mx.eval(logits_fp16)
        fp16_top5 = mx.argsort(logits_fp16)[-5:][::-1]
        mx.eval(fp16_top5)
        fp16_top5_set = set(int(t) for t in fp16_top5)

        print(f"Prompt ({n_tokens} tok): '{prompt[:50]}...'")
        print(f"  FP16 top1: '{tokenizer.decode([int(fp16_top5[0])])}'")

        # TurboQuant
        for bits in [2, 3, 3.5, 4]:
            tq_cache = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]
            logits_tq = model(tokens, cache=tq_cache)[0, -1].astype(mx.float32)
            mx.eval(logits_tq)

            cos = float(mx.sum(logits_fp16 * logits_tq) / (mx.linalg.norm(logits_fp16) * mx.linalg.norm(logits_tq) + 1e-8))
            tq_top5 = mx.argsort(logits_tq)[-5:][::-1]
            mx.eval(tq_top5)
            top1_match = int(tq_top5[0]) == int(fp16_top5[0])
            top5_overlap = len(set(int(t) for t in tq_top5) & fp16_top5_set)

            print(f"  TQ {bits}-bit: cos={cos:.4f}  top1={'OK' if top1_match else 'MISS':>4}  top5_overlap={top5_overlap}/5  top1='{tokenizer.decode([int(tq_top5[0])])}'")

        print()


if __name__ == "__main__":
    for model_id in MODELS:
        benchmark_model(model_id)
