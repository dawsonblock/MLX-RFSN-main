# mlx-turboquant

TurboQuant KV cache compression for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Implements [PolarQuant](https://arxiv.org/abs/2504.19874) (Google, ICLR 2026) — a data-oblivious KV cache quantization scheme that achieves 3-5x memory compression with near-lossless quality.

## Results

Tested on Apple Silicon with 4 models across 4 prompts each:

| Model | 3-bit cosine | 4-bit cosine | 3-bit compression | Top-1 @4-bit |
|-------|-------------|-------------|-------------------|--------------|
| Llama 3.2-3B | **0.988** | **0.997** | 4.6x | 4/4 |
| Qwen3-4B | **0.957** | **0.995** | 4.6x | 4/4 |
| Llama 3.2-1B | 0.823 | **0.974** | 4.0x | 4/4 |
| Qwen3-1.7B | 0.128 | **0.949** | 4.6x | 4/4 |

Cosine = logit cosine similarity vs FP16 KV cache. See [REPORT.md](REPORT.md) for full methodology.

## Install

```bash
git clone https://github.com/rachittshah/mlx-turboquant
cd mlx-turboquant
uv sync --dev
```

## Usage

```python
from mlx_lm import load
from mlx_turboquant.cache import TurboQuantKVCache

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
head_dim = 128
num_layers = len(model.layers)

# Drop-in replacement for mlx-lm's KVCache
cache = [TurboQuantKVCache(bits=3, head_dim=head_dim) for _ in range(num_layers)]
tokens = mx.array([tokenizer.encode("Hello world")])
logits = model(tokens, cache=cache)
```

Supported bit widths: `2`, `3`, `3.5`, `4`. Fractional bits use channel-split (half at ceil, half at floor).

## How It Works

1. **Normalize** each KV vector and store its norm
2. **Rotate** by a fixed random orthogonal matrix (data-oblivious — same matrix for all inputs)
3. After rotation, coordinates follow a known Gaussian distribution
4. **Quantize** each coordinate using precomputed Lloyd-Max optimal codebooks
5. **Bit-pack** indices into uint32 for storage (e.g., 10 3-bit values per uint32)
6. On fetch: **unpack → lookup centroids → inverse rotate → rescale**

No calibration data needed. Near information-theoretic optimal (within 2.7x of lower bounds).

## Benchmarks

```bash
uv run python benchmarks/bench_quality.py    # Quality: cosine sim, top-k
uv run python benchmarks/bench_memory_speed.py  # Memory + speed
uv run python benchmarks/bench_full.py       # Full suite across 4 models
uv run python tests/test_core.py             # Unit tests
```

## Architecture

```
mlx_turboquant/
├── codebooks.py      # Lloyd-Max codebook loader (precomputed for N(0,1))
├── polar_quant.py    # PolarQuant: rotate + quantize + dequantize
├── qjl.py            # QJL residual correction (for future fused attention)
├── turbo_quant.py    # Combined compressor + fractional bit support
├── packing.py        # Vectorized bit-packing into uint32
├── cache.py          # TurboQuantKVCache (mlx-lm compatible)
├── attention.py      # Custom attention with QJL correction
└── integration.py    # Monkey-patch for mlx-lm SDPA dispatch
```

## Upstream PR

This is a standalone proof-of-concept. The plan is to upstream into [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) as a new cache type. See [PR_PLAN.md](PR_PLAN.md) for the integration strategy.

## License

MIT
