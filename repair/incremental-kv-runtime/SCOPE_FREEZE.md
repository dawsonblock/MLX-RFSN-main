# Scope Freeze — Incremental KV Runtime Repair

## Canonical Configuration (Only Promotion-Eligible Path)

| Parameter | Value |
|-----------|-------|
| Key quantization | K8 (8-bit grouped symmetric Cartesian) |
| Value quantization | V5 (5-bit grouped symmetric Cartesian) |
| Group size | 64 |
| Preconditioner | WHT-64 (Walsh-Hadamard Transform) |
| Deterministic signs | Enabled |
| Sparse decode | **Disabled** |
| QJL score correction | **Disabled** |
| PolarQuant / TurboPolar | **Disabled** |
| v11 candidates | **Experimental / reference-only** |

## Exit Condition

Only the K8/V5 Cartesian path with WHT-64 is eligible for promotion.
All other paths are blocked from producing promotion artifacts until
this baseline earns END_TO_END_PASS on real models.

## What This Means

1. `scripts/mlx_gate.sh` will only evaluate the K8/V5 Cartesian candidate.
2. Benchmark judge will reject any candidate not using the canonical config.
3. Experimental candidates (Polar, TurboPolar, QJL, sparse) may still run in
   smoke-test mode but cannot produce `PROMOTE` or `END_TO_END_PASS` verdicts.
4. No new candidate development until the canonical path passes.

## Repair Order

| Phase | Task | Exit Condition |
|-------|------|---------------|
| 1 | Repair benchmark governance | Synthetic/fallback can't promote |
| 2 | Extract v10 codec | Stateless codec reproduces K8/V5 |
| 3 | Append-only layer cache | O(T) work, never recompresses sealed |
| 4 | Request-local sessions | Isolated per-generation cache |
| 5 | Bounded-memory attention | Never reconstructs full K/V |
| 6 | Explicit MLX-LM adapter | No monkeypatching, proof counters |
| 7 | Fix memory measurement | Actual bytes, no estimates |
| 8 | Real-model promotion tests | END_TO_END_PASS on fixed corpus |
| 9 | Metal QK kernel | Only after reference passes |
| 10 | Metal SV / online softmax | Profiling-driven |
| 11 | Server fixes | Streaming parity, stop sequences |
| 12 | Packaging | Wheel builds, importlib.resources |
| 13 | Reconsider candidates | Same cache/adapter/judge for all |
