# Benchmark Zone

This zone documents how candidates are measured, gated, and promoted.

## Central Command

```bash
python benchmarks/kv_shootout.py --promotion-report
```

This command answers:

- Which KV-cache compression candidate is best?
- Can it be promoted?
- Why or why not?
- What evidence exists?

## Required Candidates

The shootout always includes:

- `mlx_lm_baseline`
- `mlx_lm_quantized_kv`
- `rfsn_v10_k8_v5_gs32`
- `rfsn_v10_k8_v5_gs64`
- `rfsn_v11_offline_asymmetric_kv`
- `turboquant_v2_b4_gs64_rot`
- `polar_reference_dequant_on_fetch`

## Output Sections

1. **Speed ranking**
2. **Memory ranking**
3. **Quality ranking**
4. **Promotion eligibility**

If no candidate is promotable, the benchmark must say:

> No candidate is promotion eligible.

That is better than inventing a winner.

## Honest Benchmark Table

The results always show:

| Candidate | Status | Speed | Memory | Logit gate | Real cache used | Promotion |
|-----------|--------|-------|--------|------------|-----------------|-----------|
| MLX-LM baseline | Control | measured | baseline | baseline | yes | no |
| MLX-LM quantized | Control | measured | measured | pass/fail | yes | no |
| RFSN v10 gs32 | Baseline | measured | measured | pass/fail | yes/no | maybe |
| RFSN v10 gs64 | Baseline | measured | measured | pass/fail | yes/no | maybe |
| RFSN v11 | Offline only | measured | measured | pending | no | no |
| TurboQuant V2 | Experimental | measured | measured | pending/pass/fail | yes/no | maybe |
| Polar reference | Reference | measured | measured | pending | no | no |

## Quality Gates

- `logit_cosine >= 0.999`
- `KL divergence <= 1e-4`
- `top5_overlap >= 0.95`
- `top10_overlap >= 0.98`
- `max_logit_delta <= 0.05`

A candidate missing any metric is `PENDING`, not `PASS`.
