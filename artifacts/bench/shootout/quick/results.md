# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 52.02 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 45.71 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 71.58 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 76.09 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 44.02 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 58.91 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 15.89 | 0.121 | PENDING_LOGIT_GATE | yes | no |
