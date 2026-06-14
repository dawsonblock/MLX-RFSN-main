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

**Current status of `rfsn_v10_k8v4_gs64_wht`:** Passes memory gate. Speed and quality gates require full matrix run.

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

## Promotion rule

A candidate can be promoted only if:

1. Full logit gate passes
2. Memory metrics are complete
3. Real generation path uses the candidate cache
4. No global unsafe monkey-patching
5. Benchmark artifacts exist
6. Result beats baseline on at least one real axis:
   - lower memory at same quality
   - faster decode at same quality
   - longer context at same quality

## Non-promotion labels

- `offline compression only`
- `reference only`
- `pending logit gate`
- `pending memory metrics`
- `pending real cache injection`
- `failed quality gate`

## Automated Verdict System

`benchmarks/kv_shootout.py` runs `_classify_candidate()` at the end of every
shootout and attaches a verdict to each result row in `results.json` and the
markdown report.

| Verdict | Meaning |
|---------|---------|
| `PROMOTE` | All numeric gates pass. Candidate is eligible for promotion to default-on. |
| `KEEP_EXPERIMENTAL` | Quality passes but memory or performance gate is uncertain. Leave opt-in. |
| `REGRESSION` | Candidate is meaningfully slower or uses more memory than baseline. Do not promote. |
| `REJECT` | Failed quality gate or produced no output. Broken — fix before re-running. |
| `BASELINE` | This row is the FP16 reference. No verdict required. |

### Numeric gates used by the classifier

| Gate | Threshold | Source field |
|------|-----------|-------------|
| KV memory reduction | ≥ 30% | `size_ratio` |
| Peak memory reduction | ≥ 20% | `working_set_memory_mb` vs baseline |
| Latency regression limit | ≤ −10% tokens/s | `tokens_per_sec` vs baseline |
| Logit cosine similarity | ≥ 0.999 | `logit_cosine` |
| KL divergence | ≤ 1e-4 | `kl_divergence` |
| Top-5 overlap | ≥ 0.95 | `top5_overlap` |
| Top-10 overlap | ≥ 0.98 | `top10_overlap` |
| Max logit delta | ≤ 0.05 | `max_logit_delta` |

When a metric is `None` (not yet measured), the candidate is marked
`PENDING_*` and is not promotion eligible.

## Running the shootout

```bash
# Quick run (0.5B, 2 prompts)
rfsn-bench --quick

# Full run (1.5B, one prompt per category)
rfsn-bench

# All prompts in every category
rfsn-bench --all-prompts

# Specific categories only
rfsn-bench --categories coding,math

# Results land in artifacts/bench/shootout/
#   results.json  — machine-readable with promotion_verdict field
#   results.csv   — spreadsheet-friendly
#   results.md    — human-readable with Promotion Summary table
```
