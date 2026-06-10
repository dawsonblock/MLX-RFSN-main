# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 49.35 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 44.58 | 0.500 | FAIL | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 76.16 | 0.500 | PASS | yes | yes |
| rfsn_v10_k8_v5_gs64 | BASELINE | 78.82 | 0.500 | PASS | yes | yes |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 37.41 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 67.86 | 0.281 | FAIL | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 16.37 | 0.139 | FAIL | yes | no |

## Notes

**Working-set memory measurement mode dependency**: Baseline working-set memory differs between full-logit mode (~975 MB) and memory-report mode (~1422 MB). This is due to different run paths, model warmup states, prompt lengths, and sampling timing. Working-set memory should be treated as measurement-mode dependent, not promotion-critical. Actual KV cache bytes (actual_kv_memory_mb) are the stable compression proof.
**Promotion status**: rfsn_v10_k8_v5_gs32 and rfsn_v10_k8_v5_gs64 are promotion eligible under Alpha 8.3 rules. Promoted candidate: rfsn_v10_k8_v5_gs64.
