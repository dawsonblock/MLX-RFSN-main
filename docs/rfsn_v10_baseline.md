# RFSN v10 — Stable Baseline

## Role

`rfsn_v10` is the **stable baseline**. It is not the fusion branch.

Its job is to provide a known-good reference point for the benchmark shootout.
Every new compression candidate in `rfsn_v11/candidates/` must beat or match
these numbers before being considered for promotion.

## Validated configurations

| Config        | Type                    | Notes                     |
|---------------|-------------------------|---------------------------|
| `k8_v5_gs32`  | 8-bit KV, 5-group gs=32 | Default — recommended     |
| `k8_v5_gs64`  | 8-bit KV, 5-group gs=64 | Also validated             |

## What belongs in rfsn_v10

- `k8_v5_gs32` and `k8_v5_gs64` quantization presets
- Causal attention reference implementation
- Server healthcheck and config validation
- Benchmark harness (used by shootout as the baseline adapter)
- CPU test pass (no MLX required)
- Server error handling (400/503, never 500)

## What does NOT belong in rfsn_v10

Do not add any of the following to rfsn_v10:

- TurboQuant ideas → put in `rfsn_v11/candidates/turboquant_v2_adapter.py`
- PolarQuant / Lloyd-Max → `rfsn_v11/candidates/polar_reference_adapter.py`
- vMLX server pieces → `external/vmlx/` reference only
- KIVI / paged attention → `docs/kivi_reference.md` notes only
- Any experimental quantization scheme not yet validated

## Promotion rule

`rfsn_v10` is frozen as baseline.

If the shootout selects a winner from `rfsn_v11`, that winner is promoted to
`stable candidate` status. `rfsn_v10` then becomes `legacy baseline`.

If `rfsn_v10 k8_v5_gs32` wins the shootout, it remains the default.

See `docs/candidate_matrix.md` for the current candidate comparison.
See `benchmarks/kv_shootout.py` for the benchmark that decides the winner.
