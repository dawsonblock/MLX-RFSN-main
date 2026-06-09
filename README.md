# RFSN v10 Main 28 — Alpha Candidate

## Status: RFSN v10 Main 28

**Alpha candidate.** Telemetry tickets reconciled, inference server
(FastAPI + SSE) implemented for mlx/torch backends, compile + packaging + CPU gates passing.
MLX-dependent tests pass on Apple Silicon. Docker compose runs healthcheck only (not inference).

To verify the current state locally:

```bash
python -m compileall -q rfsn_v10 tests          # must produce no output
python scripts/release_gate.py --cpu-only        # must print: Gate: 9 passed, 0 failed
```

| Path | Status |
|------|--------|
| Stable runtime (MLX 8-bit KV compression) | Alpha — validated on Apple Silicon, quality gates passing |
| Package installation (subpackages) | Fixed — `rfsn_v10.kernels`, `rfsn_v10.runtime` install correctly |
| CLI health check (`python -m rfsn_v10 healthcheck`) | Working |
| Sparse decode | Disabled by default — not end-to-end proven |
| QJL score correction | Experimental — disabled by default, requires explicit opt-in |
| Polar / hybrid quantization | Experimental — disabled by default, requires explicit opt-in |
| Adaptive sparse controller | Experimental — disabled by default, requires explicit opt-in |
| CUDA backend | **Not implemented** |
| Full portable runtime | Not implemented — MLX required for core runtime |
| End-to-end speedup | Not proven — decode TPS comparable, compression overhead makes total slower at short contexts |
| Production deployment | FastAPI server — `/v1/chat/completions` with SSE streaming (mlx/torch only) |
| Docker | Healthcheck validation + ClickHouse telemetry (CPU-only, no inference) |
| >8-bit compression | Uses raw uint32 fallback — bit-packing is real for 2-8 bit only |
| Experimental Metal | No Metal kernels exist for the experimental quantization paths |
| Experimental throughput | No experimental throughput speedup is proven |

---

## Platform Support

| Platform | Status |
|----------|--------|
| Apple Silicon + MLX | Supported (primary runtime) |
| NumPy CPU | Partial — kernel validation, config, security tests pass; MLX-dependent runtime tests skip |
| Linux / CI | CPU tests pass; MLX suites skip cleanly |
| CUDA | Not implemented |
| macOS x86 (Intel) | MLX not supported on Intel Macs — NumPy-only |

---

## Quick Start

```bash
# Requires Python 3.11 and Apple Silicon with MLX for full runtime
pip install -e .

# Verify install
python -m rfsn_v10 version

# Health check
python -m rfsn_v10 healthcheck

# Validate config
python -m rfsn_v10 validate-config --config configs/default_runtime.yaml

# CPU-safe tests (no MLX required)
pytest tests/test_config.py tests/test_config_strict.py \
       tests/test_kernels_validation.py \
       tests/test_quantization_lazy_imports.py \
       tests/test_experimental_flags.py \
       tests/test_clickhouse_security.py \
       tests/test_no_runtime_raw_sdpa.py -q

# MLX-dependent tests (Apple Silicon required)
pytest tests/test_attention.py tests/test_bitpack.py \
       tests/test_bitpack_fuzz.py tests/test_kv_manager.py \
       tests/test_drift.py tests/test_attention_causal_mask.py \
       tests/test_short_prompt_decode_drift.py \
       tests/test_prefill_decode_split.py -q
```

## Inference Server

Run the OpenAI-compatible FastAPI server locally:

```bash
export RFSN_MODEL_ID=mlx-community/Llama-3-8B-Instruct-4bit
python -m rfsn_v10.server
# Or with uvicorn directly:
# uvicorn rfsn_v10.server.app:app --host 0.0.0.0 --port 8000
```

Test the endpoint:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Or with Docker Compose (container validation + telemetry):

```bash
export CLICKHOUSE_PASSWORD=your-password
docker compose up -d
# Runs healthcheck and exits — inference server requires mlx/torch backend natively
```

---

## Architecture

### Stable Modules

- `rfsn_v10/bitpack.py` — Bit-packed quantizer (2–8 bit widths) with exact roundtrip guarantees
- `rfsn_v10/kv_manager.py` — TurboQuant KV manager with grouped symmetric quantization and WHT preconditioning
- `rfsn_v10/attention.py` — Adaptive block-sparse attention; all dense fallbacks route through `attention_reference.py`
- `rfsn_v10/attention_reference.py` — Canonical causal attention reference (always applies causal mask for T_q > 1)
- `rfsn_v10/runtime/engine.py` — Orchestrator integrating KV cache, sparse attention, audit mode, and telemetry
- `rfsn_v10/runtime/__init__.py` — Re-exports `RFSNRuntime` from `engine.py`
- `rfsn_v10/runtime/generation.py` — `RFSNGenerator` with prefill, decode, sampling, and telemetry
- `rfsn_v10/config.py` — Strict Pydantic config (extra='forbid' on all models)
- `rfsn_v10/health.py` — Health check system; returns UNHEALTHY until checks have been run
- `rfsn_v10/clickhouse_client.py` — HMAC-SHA256 prompt sanitization, recursive sanitizer, retry queue
- `rfsn_v10/model_loader.py` — Unified model/tokenizer loading (mlx-lm / transformers)
- `rfsn_v10/server/app.py` — FastAPI OpenAI-compatible server with SSE streaming

### Experimental Modules (disabled by default)

All experimental paths require explicit opt-in. The runtime **will not activate them
silently** — attempting to use an experimental feature without enabling it raises a
`RuntimeError`.

```yaml
# config.yaml
experimental:
  enable_qjl: false    # QJL score correction
  enable_polar: false  # Polar / hybrid quantization
  enable_adaptive: false  # Adaptive sparse controller
```

Or via environment:

```bash
RFSN_EXPERIMENTAL_QJL=true
RFSN_EXPERIMENTAL_POLAR=true
RFSN_EXPERIMENTAL_ADAPTIVE=true
```

**Warning:** Experimental features are not validated for production or quality-critical generation.

- `rfsn_v10/quantization/polar_quant.py` — Iterative hierarchical polar quantization
- `rfsn_v10/quantization/hybrid_polar_cartesian.py` — Hybrid polar-cartesian quantizer
- `rfsn_v10/quantization/qjl_score_correction.py` — QJL sketch-based score correction
- `rfsn_v10/quantization/isoquant_precondition.py` — IsoQuant quaternion preconditioner

---

## Validated Stable Configs

Validated at the beta level on Apple Silicon. Quality thresholds (cosine ≥ 0.998 vs.
FP32 reference) measured with `tests/test_short_prompt_decode_drift.py`.

| Config | K bits | V bits | Group size | Status |
|--------|--------|--------|------------|--------|
| `k8_v5_gs32` | 8 | 5 | 32 | **Default** — slightly better cosine |
| `k8_v5_gs64` | 8 | 5 | 64 | Validated |
| `k8_v4_gs64` | 8 | 4 | 64 | Validated |

---

## Known Limitations

1. **Sparse path is decode-focused.** Prefill always uses dense attention with causal masking.
2. **QJL correction is disabled by default.** Experimental — not validated.
3. **Experimental configs may degrade logits.** Short-prompt drift under active investigation.
4. **Full runtime requires MLX.** Apple Silicon is mandatory for the core runtime.
5. **Docker service mode is not production hardened.** CLI health check only; no HTTP service exposed.
6. **CUDA not implemented.** Do not depend on it.
7. **End-to-end speedup not proven.** Decode TPS is comparable; compression overhead makes total time slower at short contexts.
8. **macOS x86 Metal not supported.** MLX is ARM-only; Intel Mac users get NumPy backend only.

---

## Security

- Telemetry prompt text is HMAC-SHA256 hashed before leaving the process boundary.
- Sanitization is recursive — nested dicts and message lists are also cleaned.
- **`RFSN_TELEMETRY_HMAC_KEY` is required** when events with sensitive fields are written.
  An absent key raises `RuntimeError` rather than silently hashing with an empty key.
- HTTPS is required for remote ClickHouse hosts; HTTP is only allowed for localhost.
- Retry queue stores `(table, event)` tuples — routing metadata is never mixed into event payloads.

---

## Running Tests

```bash
# CPU-safe tests (no MLX required, runs on Linux CI)
RFSN_BACKEND=numpy pytest \
    tests/test_config.py \
    tests/test_config_strict.py \
    tests/test_kernels_validation.py \
    tests/test_quantization_lazy_imports.py \
    tests/test_experimental_flags.py \
    tests/test_clickhouse_security.py \
    tests/test_no_runtime_raw_sdpa.py -q

# MLX-dependent tests (Apple Silicon required)
pytest \
    tests/test_attention.py \
    tests/test_bitpack.py \
    tests/test_bitpack_fuzz.py \
    tests/test_kv_manager.py \
    tests/test_drift.py \
    tests/test_attention_causal_mask.py \
    tests/test_short_prompt_decode_drift.py \
    tests/test_prefill_decode_split.py -q

# Security tests
pytest tests/test_clickhouse_security.py \
       tests/test_clickhouse_routing.py \
       tests/test_tool_runner_security.py -q

# Full CI (mirrors .github/workflows/ci.yml linux-cpu job)
RFSN_BACKEND=numpy pytest --collect-only -q
```

---

## Benchmarks

```bash
# Fast run (attention + bitpack)
python benchmarks/run_all.py --fast

# Full benchmark suite
python benchmarks/run_all.py

# Regression check against production_baseline.json
python benchmarks/run_all.py --check
```

Results are written to `benchmarks/results/run_all_<timestamp>.json` and `benchmarks/results/latest.json`.

---

## Docker

```bash
docker build -t rfsn-qjl .

# Health check (Dockerfile default CMD now runs healthcheck)
docker run --rm rfsn-qjl

# With ClickHouse (production compose)
cp .env.example .env   # fill CLICKHOUSE_PASSWORD
docker-compose up

# With ClickHouse ports exposed (local dev only)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Requires Python 3.11. Core runtime does not function in Docker without Apple Silicon + MLX.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RFSN_BACKEND` | *(auto)* | Force backend: `mlx` or `numpy` |
| `RFSN_LOG_LEVEL` | `INFO` | Logging level |
| `RFSN_CACHE_DIR` | `~/.cache/rfsn` | KV cache directory |
| `RFSN_TELEMETRY_HMAC_KEY` | *(required if telemetry enabled)* | HMAC key for prompt sanitisation |
| `RFSN_CLICKHOUSE_HOST` | `localhost` | ClickHouse hostname |
| `RFSN_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `RFSN_CLICKHOUSE_SECURE` | `true` | Use HTTPS |
| `RFSN_CLICKHOUSE_TOKEN` | *(empty)* | Bearer token for RFSN-Auth header |
| `RFSN_EXPERIMENTAL_QJL` | `false` | Enable QJL score correction |
| `RFSN_EXPERIMENTAL_POLAR` | `false` | Enable polar/hybrid quantization |
| `RFSN_EXPERIMENTAL_ADAPTIVE` | `false` | Enable adaptive sparse controller |
