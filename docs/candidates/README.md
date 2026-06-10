# Candidates Zone

This zone documents all compression candidates and their current status.

## Candidate Status Definitions

| Status | Meaning | Can Promote? |
|--------|---------|-------------|
| `CONTROL` | Comparison target (MLX-LM baseline, MLX-LM quantized) | No |
| `BASELINE` | Proven stable (RFSN v10 configs) | Yes |
| `EXPERIMENTAL` | Under evaluation (TurboQuant V2) | Yes |
| `OFFLINE_ONLY` | Compression measured offline; no real cache injection (RFSN v11) | No |
| `REFERENCE_ONLY` | External reference; not an internal runtime candidate (Polar) | No |
| `PROMOTION_ELIGIBLE` | Passed all gates; awaiting final decision | Yes |
| `PROMOTED` | Winner selected; integration target | N/A |
| `FAILED` | Did not pass required gates | No |

## Current Candidates

| Candidate | Status | Real Cache? | Notes |
|-----------|--------|------------|-------|
| `mlx_lm_baseline` | CONTROL | yes | FP16, no compression |
| `mlx_lm_quantized_kv` | CONTROL | yes | Built-in MLX-LM quantization |
| `rfsn_v10_k8_v5_gs32` | BASELINE | yes | Stable |
| `rfsn_v10_k8_v5_gs64` | BASELINE | yes | Stable |
| `rfsn_v11_offline_asymmetric_kv` | OFFLINE_ONLY | no | Needs real cache injection |
| `turboquant_v2_b4_gs64_rot` | EXPERIMENTAL | yes | Real cache, must pass full logit gate |
| `polar_reference_dequant_on_fetch` | REFERENCE_ONLY | yes | External reference |

## Promotion Rules

1. Only `EXPERIMENTAL` or `BASELINE` candidates can become `PROMOTED`.
2. `OFFLINE_ONLY` cannot promote until real cache injection exists.
3. `REFERENCE_ONLY` cannot promote unless upgraded into a real runtime candidate.
4. `CONTROL` does not promote; it is the comparison target.
