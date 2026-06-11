# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| rfsn_v10_k8_v5_gs32 | BASELINE | 76.16 | 0.500 | PASS_NO_PROMOTE | yes | no |
| rfsn_v10_k8_v5_gs64 | BASELINE | 78.82 | 0.500 | PASS_NO_PROMOTE | yes | no |

| *Summary* | — | — | — | — | — | **No candidate is promotion eligible.** |

## Notes

**Methodology:** `teacher_forced_logit_v1`  
**Promotion allowed:** False  
**Schema version:** 2.0  

**Working-set memory measurement mode dependency**: Baseline working-set memory differs between full-logit mode (~975 MB) and memory-report mode (~1422 MB). This is due to different run paths, model warmup states, prompt lengths, and sampling timing. Working-set memory should be treated as measurement-mode dependent, not promotion-critical. Actual KV cache bytes (actual_kv_memory_mb) are the stable compression proof.
**Token sequence hash:** *empty* — promotion blocked until teacher-forced rerun produces a non-empty hash.
**Current status:** No candidate is promotion eligible. Official promoted candidate: NONE.
