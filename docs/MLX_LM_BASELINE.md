# MLX-LM Production Baseline

## Principle

Do not fight MLX-LM. Use it as the stable reference.

MLX-LM is the maintained runtime baseline. RFSN candidates must beat MLX-LM on at least one useful axis:

- Lower KV memory at similar quality
- Better tokens/sec at similar quality
- Longer context at similar quality
- Better stability than MLX-LM quantized cache

## Decision Rule

| If this wins... | Then do this |
|-----------------|-------------|
| MLX-LM built-in quantized KV | Use it. It is maintained upstream. |
| TurboQuant V2 | Integrate it. It has real cache and good compression. |
| RFSN v10 | Keep it. It is stable and proven. |
| RFSN v11 | Promote it **only after** real cache injection exists. |
| None of the above | Stay on MLX-LM baseline until a candidate proves itself. |

## Quality Gate

Any candidate that claims to beat MLX-LM must prove it with:

1. Real logit comparison (`logit_cosine >= 0.999`)
2. Real memory measurement (`size_ratio < 1.0`)
3. Real cache injection (not offline simulation)
4. Reproducible benchmark artifacts

If a candidate cannot show all four, it has not beaten MLX-LM.
