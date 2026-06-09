# Benchmarking

The primary benchmark is `benchmarks/kv_shootout.py` — a head-to-head
comparison of all KV-compression candidates against the FP16 baseline.

---

## Prerequisites

```bash
pip install -e ".[dev]"
# mlx and mlx-lm are required for actual generation
pip install mlx mlx-lm
```

---

## Running the shootout

```bash
# Quick sanity run (0.5B model, 2 prompts, 50 tokens each)
rfsn-bench --quick

# Standard full run (1.5B model, 1 prompt per category, 200 tokens each)
rfsn-bench

# Every prompt in every category
rfsn-bench --all-prompts

# Specific categories
rfsn-bench --categories coding,math,long_context

# Specific model
rfsn-bench --model Qwen/Qwen2.5-3B-Instruct

# Custom output directory
rfsn-bench --out /tmp/my-bench-run
```

---

## Prompt categories

| Category | Prompts per category | What it tests |
|----------|---------------------|---------------|
| `short_chat` | 2 | Simple 1-turn chat |
| `coding` | 3 | Code generation |
| `summarization` | 2 | Compression of knowledge |
| `long_context` | 2 | Extended reasoning |
| `math` | 2 | Step-by-step reasoning |
| `multi_turn` | 1 | Multi-turn conversation fidelity |

---

## Output files

Results land in `artifacts/bench/shootout/` by default.

| File | Description |
|------|-------------|
| `results.json` | Machine-readable; includes `promotion_verdict` per row |
| `results.csv` | Spreadsheet-friendly |
| `results.md` | Human-readable report with Promotion Summary table |

---

## Reading the results

### Quality gate

Each non-baseline candidate is evaluated with a text-heuristic gate (word
overlap vs baseline). The gate is conservative:

- Word match ≥ 95 % → `PASS` (confirm with full logit gate)
- Word match > 0 % → `PASS` with `PENDING full logit gate` note
- Empty output → `FAIL`

Full logit comparison (cosine, KL, top-k) requires model internals access
and is run separately via the MLX gate.

### Promotion verdict

See `docs/CANDIDATE_PROMOTION.md` for the full verdict table. Briefly:

- **PROMOTE** — candidate is eligible for default-on
- **KEEP_EXPERIMENTAL** — quality passes but memory/speed gate is uncertain
- **REGRESSION** — do not enable; slower than baseline by > 10 %
- **REJECT** — broken; quality gate failed

### Metric definitions

| Metric | Description |
|--------|-------------|
| `size_ratio` | compressed_size / baseline_size (lower = more compact) |
| `compression_factor` | baseline_size / compressed_size (higher = better) |
| `tokens_per_sec` | Decode throughput |
| `total_ms` | End-to-end wall time for the generation |
| `logit_cosine` | Mean cosine similarity of output logits vs baseline |
| `kl_divergence` | Mean KL divergence vs baseline softmax distribution |
| `top5_overlap` | Fraction of top-5 logit tokens that match baseline |

> **Example:** `size_ratio=0.265` means *compressed size is 26.5 % of FP16
> (3.77× smaller)*.  Do **not** say "0.265× compression".

---

## Interpreting no-MLX runs

When `mlx` and `mlx-lm` are not installed, candidates call `is_available()`
and are skipped gracefully. The baseline adapter may also be unavailable.
Use `--quick` on a machine with MLX installed to get meaningful numbers.

---

## Re-running after a fix

Delete or move old results before re-running to avoid stale data:

```bash
rm -rf artifacts/bench/shootout/
rfsn-bench
```
