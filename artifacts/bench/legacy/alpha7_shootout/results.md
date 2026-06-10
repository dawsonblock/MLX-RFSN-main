# KV-Cache Compression Shootout Results
**Metric definitions**
- `size_ratio` = compressed_size / baseline_size (lower is better)
- `compression_factor` = baseline_size / compressed_size (higher is better)
- Example: size_ratio=0.265 → *Compressed size: 26.5% of FP16 (3.77× smaller)*

**Quality gate thresholds**
- logit_cosine ≥ 0.999
- KL divergence ≤ 0.0001
- top5_overlap ≥ 0.95
- top10_overlap ≥ 0.98

## Qwen/Qwen2.5-0.5B-Instruct
| Candidate | Prompt | Gate | tokens/s | total_ms | size_ratio | compression_factor | cosine | KL | top5 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx_lm_baseline | Hello | PASS | 56.0 | 875 | — | — | 1.00000 | 0.00e+00 | 1.000 | FP16 baseline — no compression applied |
| mlx_lm_quantized_kv_b8 | Hello | PASS | 65.4 | 749 | — | — | — | — | — | MLX-LM built-in 8-bit KV quantization  [text drift |
| rfsn_v10_k8_v5_gs32 | Hello | PASS | 83.4 | 611 | — | — | — | — | — | RFSN v10 stable baseline — config=k8_v5_gs32 bits= |
| rfsn_v10_k8_v5_gs64 | Hello | PASS | 86.9 | 587 | — | — | — | — | — | RFSN v10 stable baseline — config=k8_v5_gs64 bits= |
| rfsn_v11_k8v4_gs64_wht | Hello | PASS | 69.1 | 710 | 0.398 | 2.51× | 0.99998 | — | — | RFSN v11 fusion k8v4 gs64 wht=True  KV cosine=1.00 |
| turboquant_v2_b4_gs64_rot | Hello | PASS | 108.6 | 451 | — | — | — | — | — | TurboQuant V2: b4 gs64 rotation=True  Ideas from e |
| polar_reference_b4_d128 | Hello | PASS | 107.4 | 456 | — | — | — | — | — | PolarQuant reference: b4 d128  EXPERIMENTAL — expe |
| mlx_lm_baseline | Write a Python function that a | PASS | 87.9 | 467 | — | — | 1.00000 | 0.00e+00 | 1.000 | FP16 baseline — no compression applied |
| mlx_lm_quantized_kv_b8 | Write a Python function that a | PASS | 65.8 | 624 | — | — | — | — | — | MLX-LM built-in 8-bit KV quantization  [text drift |
| rfsn_v10_k8_v5_gs32 | Write a Python function that a | PASS | 103.6 | 492 | — | — | — | — | — | RFSN v10 stable baseline — config=k8_v5_gs32 bits= |
| rfsn_v10_k8_v5_gs64 | Write a Python function that a | PASS | 107.5 | 474 | — | — | — | — | — | RFSN v10 stable baseline — config=k8_v5_gs64 bits= |
| rfsn_v11_k8v4_gs64_wht | Write a Python function that a | PASS | 50.4 | 813 | 0.398 | 2.51× | 0.99998 | — | — | RFSN v11 fusion k8v4 gs64 wht=True  KV cosine=1.00 |
| turboquant_v2_b4_gs64_rot | Write a Python function that a | PASS | 94.3 | 435 | — | — | — | — | — | TurboQuant V2: b4 gs64 rotation=True  Ideas from e |
| polar_reference_b4_d128 | Write a Python function that a | PASS | 85.1 | 482 | — | — | — | — | — | PolarQuant reference: b4 d128  EXPERIMENTAL — expe |

## Decision

**Winner: `turboquant_v2_b4_gs64_rot`** — 108.6 tokens/s, quality gate PASS

See `STRUCTURE.md` → Promotion rule for next steps.
