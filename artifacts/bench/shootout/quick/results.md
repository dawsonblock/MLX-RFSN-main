# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 62.71 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 59.46 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 96.69 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 97.30 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 54.15 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 69.92 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 18.20 | 0.121 | PENDING_LOGIT_GATE | yes | no |
