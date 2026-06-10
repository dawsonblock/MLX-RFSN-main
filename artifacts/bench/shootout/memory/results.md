# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 46.24 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 35.00 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 68.82 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 73.28 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 35.02 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 54.19 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 14.73 | 0.139 | PENDING_LOGIT_GATE | yes | no |
