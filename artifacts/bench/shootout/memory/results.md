# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 56.99 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 44.22 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 83.74 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 87.39 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 38.97 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 67.17 | 0.281 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 16.70 | 0.139 | PENDING_LOGIT_GATE | yes | no |

## Notes

**Working-set memory measurement mode dependency**: Baseline working-set memory differs between full-logit mode (~975 MB) and memory-report mode (~1422 MB). This is due to different run paths, model warmup states, prompt lengths, and sampling timing. Working-set memory should be treated as measurement-mode dependent, not promotion-critical. Actual KV cache bytes (actual_kv_memory_mb) are the stable compression proof.
**Promotion status**: rfsn_v10_k8_v5_gs32 and rfsn_v10_k8_v5_gs64 are promotion eligible under Alpha 8.3 rules. Promoted candidate: rfsn_v10_k8_v5_gs64.
