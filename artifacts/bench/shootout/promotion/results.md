# KV Shootout Results

## Honest Benchmark Table

| Candidate | Status | Speed (tps) | Memory (ratio) | Logit gate | Real cache used | Promotion |
|-----------|--------|-------------|----------------|------------|-----------------|-----------|
| rfsn_v10_k8_v5_gs32 | BASELINE | 76.16 | 0.500 | PASS | yes | yes |
| rfsn_v10_k8_v5_gs64 | BASELINE | 78.82 | 0.500 | PASS | yes | yes |

## Notes

**Promotion result**: rfsn_v10_k8_v5_gs32 and rfsn_v10_k8_v5_gs64 are promotion eligible under Alpha 8.3 rules. Promoted candidate: rfsn_v10_k8_v5_gs64.
**Scope**: This is an alpha-level baseline promotion, not a beta or production-ready release. Validation is limited to Qwen/Qwen2.5-0.5B-Instruct.
**Working-set memory**: Measurement-mode dependent; actual_kv_memory_mb is the stable compression proof.
