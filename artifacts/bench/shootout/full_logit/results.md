# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 50.27 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 50.25 | 0.500 | FAIL | yes | no |
| rfsn_v10_k8_v5_gs32 | BASELINE | 5.50 | 0.500 | FAIL | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 5.43 | 0.500 | FAIL | yes | no |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | 37.61 | 0.398 | PENDING_REAL_CACHE_INJECTION | no | no |
| turboquant_v2_b4_gs64_norot | EXPERIMENTAL | 82.84 | 0.281 | FAIL | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 18.40 | 0.139 | FAIL | yes | no |
| turbo_polar_k4_qjl64 | EXPERIMENTAL | 4.86 | baseline | FAIL | yes | no |

| *Summary* | — | — | — | — | — | **No candidate is promotion eligible.** |

## Notes

**Methodology:** `teacher_forced_logit_v1`  
**Promotion allowed:** False  
**Schema version:** 2.0  

**Working-set memory measurement mode dependency**: Baseline working-set memory differs between full-logit mode (~975 MB) and memory-report mode (~1422 MB). This is due to different run paths, model warmup states, prompt lengths, and sampling timing. Working-set memory should be treated as measurement-mode dependent, not promotion-critical. Actual KV cache bytes (actual_kv_memory_mb) are the stable compression proof.
**Token sequence hash:** `c2ff72f2e716ef3e30262cbaf5ec955629fe07b40ea88c6808602cb5b8b06716`  
**Current status:** No candidate is promotion eligible. Official promoted candidate: NONE.
