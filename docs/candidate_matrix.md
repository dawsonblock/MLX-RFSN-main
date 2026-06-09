# Candidate Matrix

Current status of all KV-compression candidates.

Strength/weakness columns reflect expected behavior before shootout results.
Update this table after running `benchmarks/kv_shootout.py`.

| Candidate                  | Source                                      | Status         | Expected strength                       | Known weakness                                 |
|----------------------------|---------------------------------------------|----------------|-----------------------------------------|------------------------------------------------|
| `mlx_lm_baseline`          | mlx-lm built-in                             | stable         | Perfect quality, reference speed        | No compression                                 |
| `mlx_lm_quantized_kv`      | mlx-lm built-in (kv_bits flag)             | stable/opt-in  | Maintained, simple                      | Availability depends on mlx-lm version         |
| `rfsn_v10_k8_v5_gs32`      | rfsn_v10 stable                             | stable         | Validated cosine≥0.998, good quality    | No rotation pre-conditioning                   |
| `rfsn_v10_k8_v5_gs64`      | rfsn_v10 stable                             | stable         | Slightly better compression than gs32  | Same as above                                  |
| `rfsn_v11_fusion`          | rfsn_v11 (WHT + PolarQuant)                 | experimental   | Better compression + quality via rotation | Unvalidated on real models, may be slower     |
| `turboquant_v2`            | external/turboquant-mlx (rebuilt as adapter)| experimental   | QR rotation → uniform dist → better quant | Overhead from rotation matrix                |
| `polar_reference`          | external/mlx-turboquant (rebuilt as adapter)| experimental   | Theoretically near-optimal distortion   | Data-oblivious, slower, no production perf     |

## Promotion rule

A candidate moves from `experimental` to `stable candidate` only after:
1. Passing all quality gates (see `docs/benchmark_methodology.md`)
2. Winning or tying the shootout on tokens/sec vs. mlx_lm_baseline
3. README, manifest, and pyproject updated to reflect new status

## After shootout

Fill in the results column:

| Candidate | size_ratio | compression_factor | tokens/s | gate | verdict |
|-----------|------------|--------------------|----------|------|---------|
| (run kv_shootout.py) | | | | | |
