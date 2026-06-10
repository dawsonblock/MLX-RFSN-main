# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 41.13 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 46.51 | 0.500 | FAIL | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 76.78 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 89.48 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 36.11 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 68.56 | 0.281 | FAIL | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 14.74 | 0.139 | FAIL | yes | no |
