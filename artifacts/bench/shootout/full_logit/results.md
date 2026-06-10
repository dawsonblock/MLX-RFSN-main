# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 32.82 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 35.93 | 0.500 | FAIL | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 66.13 | 0.500 | PASS_NO_PROMOTE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 65.94 | 0.500 | PASS_NO_PROMOTE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 31.84 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 52.15 | 0.281 | FAIL | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 13.57 | 0.139 | FAIL | yes | no |
