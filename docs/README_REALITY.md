# MLX-RFSN — Honest Status

**Version:** v10.2 Stable Alpha
**Target:** Local Apple Silicon MLX inference research server

## What actually works

| Feature | Status |
|---------|--------|
| MLX model loading (HuggingFace or local) | Stable |
| OpenAI-compatible FastAPI server | Stable |
| Baseline dense generation | Stable |
| v10 KV compression (k8_v5 + WHT bitpack) | Stable — **off by default**; set `RFSN_ENABLE_KV_COMPRESSION=true` after benchmarking |
| Streaming SSE (queue-bridge, non-blocking) | Stable |
| CPU-safe test suite | Stable |
| Telemetry with HMAC prompt hashing | Stable |
| Local dashboard at /dashboard | Stable |
| API key enforcement (opt-in) | Stable |

## What is experimental (disabled by default)

| Feature | Status |
|---------|--------|
| Sparse decode | Experimental — no benchmark-proven speedup |
| QJL score correction | Experimental — disabled pending quality gate |
| PolarQuant KV compression | Experimental — slower on head_dim=64 models |
| IsoQuant preconditioner | Experimental — not validated |
| v11 fusion candidates | Experimental — candidate lab only |
| Adaptive sparse controller | Experimental — not validated |

## What is not implemented

| Claim | Reality |
|-------|---------|
| CUDA backend | Not implemented (stub only) |
| Production serving | This is a research alpha |
| Guaranteed speedup | Must be measured per model and context length |
| Multi-model routing | Stub only |

## How to check for yourself

```bash
# Baseline (no compression)
RFSN_ENABLE_QUANTIZED_KV=false rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# With KV compression
rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Run benchmark comparison
python benchmarks/kv_shootout.py --quick

# Release gate
python scripts/release_gate.py --cpu-only
```

## Benchmark results

See `benchmarks/results/` for the latest shootout results.
No candidate is claimed as faster until a benchmark report in `benchmarks/reports/` proves it.
