# Benchmark Methodology

## Shootout script

`benchmarks/kv_shootout.py` is the authoritative comparison tool.

It outputs:
- `artifacts/bench/shootout/results.json`
- `artifacts/bench/shootout/results.csv`
- `artifacts/bench/shootout/results.md`

## Models

| Model                        | Role                |
|------------------------------|---------------------|
| Qwen/Qwen2.5-0.5B-Instruct   | Fast iteration test |
| Qwen/Qwen2.5-1.5B-Instruct   | Representative size |

## Prompts

| Prompt                                                        | Purpose                  |
|---------------------------------------------------------------|--------------------------|
| `Hello`                                                       | Short-prompt baseline    |
| `The capital of Canada is`                                    | Factual recall           |
| `Write a Python function that adds two numbers.`              | Code generation          |
| `Explain the difference between RAM and storage.`             | Long-answer generation   |
| `Summarize this paragraph in one sentence.`                   | Summarization            |

## Metrics

### Compression

| Metric              | Definition                                     | Direction   |
|---------------------|------------------------------------------------|-------------|
| `size_ratio`        | `compressed_size / baseline_size`              | Lower better |
| `compression_factor`| `baseline_size / compressed_size`              | Higher better |

**Wording rule**: Do NOT say "0.265× compression". Say:
> Compressed size: 26.5% of FP16  (3.77× smaller)

### Memory

| Metric                 | Definition                        |
|------------------------|-----------------------------------|
| `actual_kv_memory_mb`  | Measured KV tensor memory (MB)    |
| `working_set_memory_mb`| Peak MLX metal memory during run  |

### Timing

| Metric       | Unit | Definition                  |
|--------------|------|-----------------------------|
| `prefill_ms` | ms   | Time to process input tokens |
| `decode_ms`  | ms   | Time to generate all tokens  |
| `total_ms`   | ms   | End-to-end wall time         |

### Throughput

| Metric          | Unit      | Definition                 |
|-----------------|-----------|----------------------------|
| `tokens_per_sec`| tokens/s  | Generated tokens / total_s |

### Quality vs. FP16 baseline

| Metric                  | Threshold     | Direction   |
|-------------------------|---------------|-------------|
| `logit_cosine`          | ≥ 0.999       | Higher better |
| `kl_divergence`         | ≤ 1e-4        | Lower better |
| `top1_match`            | (recorded)    | Higher better |
| `top5_overlap`          | ≥ 0.95        | Higher better |
| `top10_overlap`         | ≥ 0.98        | Higher better |
| `max_logit_delta`       | ≤ 0.05        | Lower better |
| `first_divergent_token` | (recorded)    | Later better |

## Quality gates

A candidate **passes** only if ALL of the following hold:

```
logit_cosine  >= 0.999
kl_divergence <= 1e-4
top5_overlap  >= 0.95
top10_overlap >= 0.98
max_logit_delta <= 0.05
```

A candidate that fails any gate is labelled:
```
experimental / failed quality gate
```

Failures are never hidden. The `gate_status` field in results.json
is always set honestly.

## Gate statuses

- `PASS` — all required gates passed
- `FAIL` — at least one required gate failed
- `PENDING_LOGIT_GATE` — text heuristic or quick run; real logit comparison not yet done
- `PENDING_MEMORY_METRICS` — actual KV memory or working-set memory not yet measured
- `PENDING_REAL_CACHE_INJECTION` — compression measured offline; generation does not use the candidate cache directly
- `ERROR` — candidate crashed or could not run

## Gate levels

**Text heuristic** (shootout default, text-generation mode):
- Word-level match vs. baseline output
- ≥95% word match → tentatively PASS, flagged for logit confirmation

**Full logit gate** (MLX gate, model internals required):
- Direct logit cosine, KL, top-k overlap computation
- Required for final promotion decision

## Decision rule

After the shootout:
1. Collect all candidates with `promotion_eligible=True`
   (requires: `logit_gate_passed=True`, `memory_gate_passed=True`,
   real cache path used, no unsafe global monkey-patching)
2. Winner = highest `tokens_per_sec` among eligible candidates
3. If no candidate is eligible, the promotion report says so honestly

See `docs/architecture.md` for the full promotion rule table.

## Reproducibility

- Always run on Apple Silicon with MLX backend
- Disable background processes during benchmark
- Run at least 2 passes and report the median
- Report hardware: chip type, total RAM, MLX version, mlx-lm version
