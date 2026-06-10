# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 73.71 | baseline | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 65.12 | baseline | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 103.07 | baseline | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 99.23 | baseline | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 56.10 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 81.64 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 19.76 | 0.121 | PENDING_LOGIT_GATE | yes | no |
