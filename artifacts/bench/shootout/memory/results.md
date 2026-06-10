# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 66.98 | baseline | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 50.96 | baseline | PENDING_MEMORY_METRICS | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 99.33 | baseline | PENDING_MEMORY_METRICS | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 95.09 | baseline | PENDING_MEMORY_METRICS | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 42.86 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 79.40 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 18.99 | 0.139 | PENDING_MEMORY_METRICS | yes | no |
