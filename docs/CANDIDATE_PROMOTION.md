# Candidate Promotion Criteria

A research candidate in `rfsn_v11/candidates/` can be promoted to the stable
runtime (`rfsn_v10`) only after passing objective gates.

No feature is promoted because it sounds advanced or because benchmarks run fast
in isolation. All promotion decisions are based on `benchmarks/kv_shootout.py`
results recorded in `benchmarks/results/`.

## KV Compression Gate

KV compression becomes **recommended** (default on) when ALL of the following hold:

| Criterion | Threshold |
|-----------|-----------|
| Peak memory reduction | >= 20% vs baseline |
| KV cache memory reduction | >= 30% vs baseline |
| Decode speed regression | No worse than -10% vs baseline |
| Logit cosine similarity | >= 0.995 |
| Top-k overlap | >= 0.95 |
| Output drift | Acceptable on human review |
| Crash count across full matrix | 0 |

**Current status of `rfsn_v10_k8v5_gs64_wht`:** Passes memory gate. Speed and quality gates require full matrix run.

## Sparse Decode Gate

Sparse decode can become **default** only when ALL of the following hold:

| Criterion | Threshold |
|-----------|-----------|
| Decode tokens/sec improvement | >= +15% vs baseline |
| Quality loss | Below KV compression thresholds |
| Context length | Works at 8k+ without regression |
| Short prompt | Does not break prompts < 64 tokens |
| First-token latency | Does not increase by > 20% |

**Current status:** Does not pass. Disabled by default.

## QJL Gate

QJL can be re-enabled only when:

| Criterion | Threshold |
|-----------|-----------|
| Score MAE | < baseline score MAE |
| Softmax KL divergence | < baseline KL |
| Top-k overlap | Improves or stays stable |
| Decode performance | Does not collapse |

**Current status:** Disabled pending quality investigation.

## PolarQuant Gate

| Criterion | Threshold |
|-----------|-----------|
| Quantize step throughput | Comparable to v10 affine path |
| Memory reduction | >= 30% |
| Quality (logit cosine) | >= 0.995 |
| Tested head_dim values | 64, 128 |

**Current status:** Experimental. Slow on head_dim=64. Vectorized patch applied in adapter.

## v11 Fusion Gate

A v11 fusion candidate can enter `rfsn_v10` when:

- Passes all `rfsn_v10` unit tests without modification
- Passes MLX integration tests
- Beats `rfsn_v10` baseline in full shootout
- Has zero stub/placeholder code paths in the hot path
- Has a rollback flag
- Has docs

## Running the shootout

```bash
# Quick run (0.5B, 3 prompts)
python benchmarks/kv_shootout.py --quick

# Full run (1.5B, all prompts)
python benchmarks/kv_shootout.py

# Results land in benchmarks/results/
# Report generated at benchmarks/reports/latest.md
```
