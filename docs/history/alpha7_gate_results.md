# Historical Alpha 7 / v10.2 Gate Results

> These are older results from Alpha 7 / v10.2. They are preserved for reference but are **not** the current Alpha 8.2 gate status.

## Non-MLX gate (Linux / any platform) — historical

| Step | Result |
|------|--------|
| `python -m compileall -q rfsn_v10 tests` | PASS |
| `pytest --collect-only -q` | PASS — 67 files, 0 errors |
| `pytest tests/test_no_placeholder_source.py` | PASS |
| `pytest tests/test_runtime_import_contract.py` | PASS |
| `pytest tests/test_config.py tests/test_config_strict.py` | PASS |
| `pytest tests/test_health.py` | PASS |
| `pytest tests/test_no_runtime_raw_sdpa.py` | PASS |
| `pytest tests/test_experimental_flags.py` | PASS |
| `pytest tests/test_quantization_lazy_imports.py` | PASS |
| `pytest tests/test_clickhouse_security.py` | PASS (34 tests) |
| `pytest tests/test_telemetry_e2e.py` | PASS (12 tests) |
| `RFSN_BACKEND=numpy python -m rfsn_v10 healthcheck` | PASS |
| `python -m build` | PASS |
| Wheel subpackage content check | PASS |
| Wheel install + import verify (Python 3.11 venv) | PASS |

## Apple Silicon MLX gate — historical

| Step | Result |
|------|--------|
| `pytest tests/test_attention.py` | PASS (12 tests) |
| `pytest tests/test_attention_causal_mask.py` | PASS (6 tests) |
| `pytest tests/test_bitpack.py` | PASS (28 tests) |
| `pytest tests/test_bitpack_fuzz.py` | PASS (5 tests) |
| `pytest tests/test_drift.py` | PASS (3 tests) |
| `pytest tests/test_kv_manager.py` | PASS (47 tests) |
| `pytest tests/test_short_prompt_decode_drift.py` | PASS (4 tests) |
| `pytest tests/test_prefill_decode_split.py` | PASS (5 tests) |
| `pytest tests/test_short_prompt_generation_regression.py` | PASS (4 tests) |
| `pytest tests/test_server_backend_errors.py` | PASS (6 tests) |
| `pytest tests/test_version_exported.py` | PASS (3 tests) |
| `RFSN_BACKEND=mlx python -m rfsn_v10 healthcheck` | PASS |

Total gate tests (historical): **893 passed, 15 skipped, 0 failed**

## Docker gate — historical

| Step | Result |
|------|--------|
| `docker build -t rfsn-qjl .` | PASS — image builds successfully |
| `docker run --rm -e RFSN_BACKEND=numpy rfsn-qjl` | PASS — healthcheck returns degraded (expected, no MLX in container) |

Docker gate: **PASS** (healthcheck-only mode verified) — historical.

## Package gate — historical

| Step | Result |
|------|--------|
| `SETUPTOOLS_SCM_PRETEND_VERSION=10.1.0a1 python -m build` | PASS |
| `pip install dist/*.whl && python -c "import rfsn_v10; print(rfsn_v10.__version__)"` | PASS |

## Benchmark gate — historical

| Step | Result |
|------|--------|
| `benchmarks/benchmark_kv_cache.py` | PASS — cosine sim 0.99998, compression 0.266 (3.75x) |
| `benchmarks/benchmark_bitpack.py` | PASS |
| `benchmarks/benchmark_attention.py` | PASS |
| `artifacts/bench/current/results.json` | Generated (historical path) |

Quality gates (historical): **PASS** — key cosine 0.99998 ≥ 0.999 threshold.
