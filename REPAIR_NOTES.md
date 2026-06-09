# RFSN v10 — Repair Notes

## Broken snapshot

The `qjl-beta-repair-broken-snapshot` tag points to the build that contained
invalid placeholder Python files. It is preserved for reference only.

The following files contained literal placeholder text and failed `python -m compileall`:

- `rfsn_v10/isoquant_precondition.py` — entire file was `FULL COMPLETE CODE WITH FUNCTIONAL INTERFACE HERE`
- `rfsn_v10/quantization/fused_isoquant_polar.py` — entire file was `FULL CODE FOR fused_isoquant_polar.py`
- `rfsn_v10/quantization/kv_quant_manager.py` — 216 lines of real code, then `FULL CODE FOR kv_quant_manager.py` appended

These are now replaced with valid disabled stubs that raise `_ExperimentalNotImplemented`
on use. The trailing placeholder line in `kv_quant_manager.py` is removed.

## Beta promotion conditions

The classifier remains `Development Status :: 3 - Alpha` until **all** of the following
pass in a clean environment:

### Non-MLX (Linux / any platform)
```
python -m compileall -q rfsn_v10 tests
python -m pip install -e ".[dev]"
pytest --collect-only -q
pytest tests/test_no_placeholder_source.py -q
pytest tests/test_runtime_import_contract.py -q
pytest tests/test_config.py tests/test_config_strict.py -q
pytest tests/test_health.py -q
pytest tests/test_no_runtime_raw_sdpa.py -q
pytest tests/test_experimental_flags.py -q
pytest tests/test_quantization_lazy_imports.py -q
pytest tests/test_clickhouse_security.py -q
RFSN_BACKEND=numpy python -m rfsn_v10 healthcheck
python -m build
python -m pip install --force-reinstall dist/*.whl
python -c "import rfsn_v10; import rfsn_v10.kernels; import rfsn_v10.quantization; from rfsn_v10 import RFSNRuntime"
docker build -t rfsn-qjl .
docker run --rm -e RFSN_BACKEND=numpy rfsn-qjl
```

### Apple Silicon + MLX
```
pytest tests/test_attention_causal_mask.py -q
pytest tests/test_drift.py -q
pytest tests/test_bitpack_fuzz.py -q
pytest tests/test_short_prompt_decode_drift.py -q
pytest tests/test_prefill_decode_split.py -q
RFSN_BACKEND=mlx python -m rfsn_v10 healthcheck
```

## What is NOT beta-ready

- QJL score correction — not validated, disabled by default
- Polar / fused IsoQuant-Polar — not validated, disabled by default
- Adaptive sparse controller — not validated, disabled by default
- CUDA backend — not implemented
- Full portable runtime without MLX — not implemented
- Production HTTP inference server — not implemented
- End-to-end speedup guarantee — not proven
