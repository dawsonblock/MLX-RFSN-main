# MLX-RFSN Fusion — Build Status

## Current build

| Field | Value |
|-------|-------|
| Release | MLX-RFSN Fusion Alpha 8.3 |
| Branch | `mlx-rfsn-fusion-alpha-8-3` |
| Snapshot | `mlx-rfsn-fusion-alpha-8-2-snapshot` (preserved) |

## Alpha 8.2 status (frozen)

- Structure clean.
- Tests pass.
- Full-logit and memory artifact paths exist.
- No active false winner.
- No candidate is promotion eligible.
- TurboQuant V2 remains pending logit gate.
- RFSN v11 remains offline-only.

## Alpha 8.3 goals

- [ ] `mlx_gate.sh` strict (no `|| true` masking).
- [ ] TurboQuant V2 real logit metrics or explicit pending status.
- [ ] Memory metrics complete for baseline/control candidates.
- [ ] `cache_policy.py` distinguishes control / baseline / promoted.
- [ ] Manifest wording sharpened (PARTIAL vs PASS).
- [ ] Active artifacts regenerated on Apple Silicon.
- [ ] Promotion report refreshed.
- [ ] No candidate falsely promoted.

## Candidate statuses

| Candidate | Status | Blocker |
|-----------|--------|---------|
| mlx_lm_baseline | CONTROL | Not a candidate |
| mlx_lm_quantized_kv_b8 | CONTROL | Not a candidate |
| rfsn_v10_k8_v5_gs32 | BASELINE | Memory metrics incomplete |
| rfsn_v10_k8_v5_gs64 | BASELINE | Memory metrics incomplete |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | Real cache injection missing |
| turboquant_v2_b4_gs64 | EXPERIMENTAL | Logit gate pending |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | Logit + memory metrics pending |
