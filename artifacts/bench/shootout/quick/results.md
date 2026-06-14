# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| mlx_lm_baseline | CONTROL | 81.06 | 1.000 | PASS_NO_PROMOTE | yes | no |
| mlx_lm_quantized_kv_b8 | CONTROL | 72.75 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_v10_k8_v5_gs64 | REFERENCE_ONLY | 37.65 | 0.500 | PENDING_LOGIT_GATE | yes | no |
| rfsn_direct_packed_k8v8_gs64 | EXPERIMENTAL | — | baseline | ERROR | no | no |
| rfsn_v11_offline_asymmetric_kv_k8v5_gs64 | EXPERIMENTAL | — | baseline | ERROR | no | no |
| turboquant_v2_b6_gs64_norot | EXPERIMENTAL | 68.03 | 0.406 | PENDING_LOGIT_GATE | yes | no |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | 20.58 | 0.121 | PENDING_LOGIT_GATE | yes | no |
| turbo_polar_k4_qjl64 | EXPERIMENTAL | 7.94 | baseline | PENDING_LOGIT_GATE | yes | no |

| *Summary* | — | — | — | — | — | **No candidate is promotion eligible.** |

## Notes

**Methodology:** `teacher_forced_logit_v1`  
**Promotion allowed:** False  
**Schema version:** 2.0  

**Working-set memory measurement mode dependency**: Baseline working-set memory differs between full-logit mode (~975 MB) and memory-report mode (~1422 MB). This is due to different run paths, model warmup states, prompt lengths, and sampling timing. Working-set memory should be treated as measurement-mode dependent, not promotion-critical. Actual KV cache bytes (actual_kv_memory_mb) are the stable compression proof.
**Token sequence hash:** *empty* — promotion blocked until teacher-forced rerun produces a non-empty hash.
**Current status:** No candidate is promotion eligible. Official promoted candidate: NONE.
