# MLX Test Summary — Main 28

## Command Run
```bash
pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       -vv -s --tb=short \
       | tee artifacts/proof/main28/mlx_pytest_raw.log

pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       --junitxml artifacts/proof/main28/mlx_pytest_junit.xml
```

## Environment
- **Hardware**: Apple M2 Pro, 16GB RAM
- **MLX version**: 0.31.2
- **Python version**: 3.12.0

## Test Files
- `tests/test_fused_kernel_mlx.py`
- `tests/test_kernel_equivalence_mlx.py`
- `tests/test_kv_manager.py`
- `tests/test_retrieve_blocks.py`
- `tests/test_sparse_safety_gate.py`

## Results
- **Passed**: 74
- **Failed**: 0
- **Skipped**: 0

## Artifacts
- Raw log: `artifacts/proof/main28/mlx_pytest_raw.log`
- JUnit XML: `artifacts/proof/main28/mlx_pytest_junit.xml`

## Notes
Main 28 — Proof Consistency + Long-Context + Throughput Honesty. All MLX-dependent tests pass without Metal fallback. Sparse decode is disabled by default.
