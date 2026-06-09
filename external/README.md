# external/ — Read-Only References

These repositories are included for reference only. They are **not imported** by the stable runtime (`rfsn_v10`) and are **not included** in the published package.

Any idea taken from `external/` must be:
1. Re-implemented cleanly behind a candidate adapter in `rfsn_v11/candidates/`
2. Benchmarked via `benchmarks/kv_shootout.py` before use
3. Promoted only after passing the quality gate in `docs/CANDIDATE_PROMOTION.md`

## Contents

| Directory | Source | Notes |
|-----------|--------|-------|
| `turboquant-mlx/` | TurboQuant-MLX | QR rotation + MLX native quantized_matmul |
| `mlx-turboquant/` | MLX-TurboQuant | PolarQuant codebook-based KV compression |
| `vmlx/` | VMLX | Apple Silicon MLX model serving experiments |

## Import Contract

`rfsn_v10` must never import from `external/`. This is enforced by:
- `tests/test_no_v11_import_from_v10.py` (import boundary test)
- `pyproject.toml` package discovery exclusions
