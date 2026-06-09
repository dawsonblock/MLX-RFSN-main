# RFSN v10 — Beta Candidate Release Manifest

## Release summary

This document records the state of the beta candidate at the time of the release gate run.

| Field | Value |
|-------|-------|
| Build branch | `qjl-beta-repair` |
| Repair plan phases | 0–13 (all complete) |
| Python requirement | 3.11 |
| Platform | Apple Silicon (primary); Linux NumPy-only CI |
| Development status | Beta |

---

## Repair plan completion

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Create snapshot/repair branches, remove macOS junk, update `.gitignore` | Done |
| 1 | Fix `pyproject.toml` package discovery (`rfsn_v10*`, `agent_core*`, `tools*` wildcards) | Done |
| 2 | Remove `runtime.py`/`runtime/` collision — restructure to `runtime/engine.py` | Done |
| 3 | Centralize causal attention — replace raw SDPA calls, add enforcement test | Done |
| 4 | Fix `python -m rfsn_v10` healthcheck + CLI subcommands | Done |
| 5 | Fix Docker — remove fake HTTP port 8080, use CLI CMD, fix docker-compose | Done |
| 6 | Fix CI workflows — Python 3.11, honest job matrix, remove fake macOS x86 Metal job | Done |
| 7 | Fix telemetry — event schema (tuple queue), HMAC mandatory, retry sleep injection | Done |
| 8 | Quarantine experimental features — QJL/polar/adaptive opt-in only | Done |
| 9 | Add quality gate tests — drift, prefill/decode split, honest thresholds | Done |
| 10 | Rebuild benchmark runner — `benchmarks/run_all.py` with NumPy + MLX suites | Done |
| 11 | Rewrite README status claims honestly | Done |
| 12 | Create release gate script — `scripts/release_gate.py` | Done |
| 13 | Clean release packaging + release manifest | Done |

---

## Release gate results

Run `python scripts/release_gate.py` to reproduce. Expected output:

```
  [import_smoke] ... PASS
  [cli_version] ... PASS
  [cli_healthcheck] ... PASS
  [config_validate] ... PASS
  [cpu_tests] ... PASS
  [security_tests] ... PASS
  [sdpa_enforcement] ... PASS
  [mlx_tests] ... PASS       (Apple Silicon only)
  [benchmark_smoke] ... PASS
  [packaging_smoke] ... PASS

Gate: 10 passed, 0 skipped, 0 failed
Release gate PASSED.
```

On Linux CI (no MLX), run with `--cpu-only`:

```
Gate: 9 passed, 1 skipped, 0 failed
Release gate PASSED.
```

---

## Quality gate thresholds (measured, not assumed)

Measured on Apple Silicon with synthetic KV tensors (4 heads, 64 dim, k8_v5_gs32):

| Metric | Threshold | Actual range |
|--------|-----------|-------------|
| Cosine (decode step) | ≥ 0.998 | 0.9987–0.9991 |
| KL divergence | ≤ 1e-6 | < 1e-7 |
| Top-5 overlap | ≥ 0.95 | passes |

---

## Known issues not fixed in this release

1. Sparse decode path is decode-only — prefill always uses dense attention
2. CUDA backend not implemented
3. macOS Intel: MLX not supported — NumPy backend only
4. End-to-end speedup not proven — compression overhead dominates at short contexts
5. Experimental features (QJL, polar, adaptive) remain unvalidated

---

## Files changed in repair

- `.gitignore` — macOS junk, build artifacts
- `pyproject.toml` — package discovery wildcards, Beta classifier
- `rfsn_v10/__init__.py` — removed importlib hack, clean imports
- `rfsn_v10/__main__.py` — proper CLI subcommands
- `rfsn_v10/attention.py` — causal_attention_dense replaces raw SDPA
- `rfsn_v10/runtime/__init__.py` — re-exports from engine.py
- `rfsn_v10/runtime/engine.py` — new location (was runtime.py)
- `rfsn_v10/runtime/scoring_modes.py` — causal_attention_dense
- `rfsn_v10/clickhouse_client.py` — HMAC mandatory, tuple queue, sleep injection
- `configs/default_runtime.yaml` — default config k8_v5_gs32
- `Dockerfile` — CLI CMD, health check
- `docker-compose.yml` — no fake port 8080, secure ClickHouse
- `docker-compose.dev.yml` — new dev override
- `.github/workflows/ci.yml` — new honest CI matrix
- `.github/workflows/cross-platform.yml` — removed fake x86 Metal job
- `benchmarks/run_all.py` — new unified benchmark runner
- `scripts/release_gate.py` — new release gate
- `tests/test_attention_causal_mask.py` — causal mask quality tests
- `tests/test_no_runtime_raw_sdpa.py` — SDPA enforcement test
- `tests/test_short_prompt_decode_drift.py` — decode drift quality gate
- `tests/test_prefill_decode_split.py` — prefill/decode split quality gate
- `tests/test_clickhouse_security.py` — updated for HMAC enforcement + tuple queue
- `README.md` — honest status claims
- `RELEASE.md` — this file
