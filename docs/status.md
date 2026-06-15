# MLX-RFSN Fusion — Build Status

## Current build

| Field | Value |
|-------|-------|
| Release | MLX-RFSN Fusion Alpha 8.4 |
| Branch | `mlx-rfsn-fusion-alpha-8-3` |
| Snapshot | `mlx-rfsn-fusion-alpha-8-4-snapshot` (preserved) |

## Alpha 8.2 status (frozen)

- Structure clean.
- Tests pass.
- Full-logit and memory artifact paths exist.
- No active false winner.
- No candidate is promotion eligible.
- TurboQuant V2 remains pending logit gate.
- RFSN v11 remains offline-only.

## Alpha 8.4 results

### P0 Critical Fixes (Completed)

- [x] **Wheel versioning fixed**: Static `version = "10.2.0a84"` in pyproject.toml prevents `0.0.0` builds.
- [x] **Registry test fixed**: `check_available=False` parameter allows portable tests to validate declared candidates without requiring MLX.
- [x] **Metal backend honestly named**: `metal_dense_reconstruction_violates_invariant` explicitly flags that this path violates the zero-reconstruction invariant.

### P1 Implementation Progress

- [x] Direct-packed K8/V8 canonical BS64 configuration (smoke BS8 separated).
- [x] Full-history materialization honestly recorded in Metal path.
- [x] Strict Metal failures raise instead of silently falling back.
- [x] **True packed kernel scaffold created**: `true_packed_attention.metal` and `true_packed_wrapper.py` provide the framework for zero-reconstruction GPU attention (not yet functional, falls back to reference).
- [x] **Differential testing framework**: `test_true_packed_kernel.py` establishes testing levels for kernel validation.
- [x] **Execution contract recording**: `ExecutionContract` dataclass provides auditability with invariant validation.
- [x] Capability-based full-logit dispatch (no hardcoded name lists).
- [x] Runtime byte counters use actual `array.itemsize` instead of hardcoded 4.
- [x] Promotion aggregation preserves all required fields.
- [x] Promotion policy rejects `execution_backend="unknown"`.
- [x] Strict mode exits nonzero when promotion policy fails.
- [x] Release identity unified (README, release.toml, _version.py).
- [x] No candidate falsely promoted.
- [x] **Quality gate thresholds unified**: Single source of truth in `LogitGateThresholds` dataclass.

## Critical blocker discovered in Alpha 8.3

**The logit comparison methodology is flawed.**

Current approach: run two independent greedy decodes and compare per-step logits.
Problem: if token N differs between baseline and candidate, all subsequent logits are computed on divergent contexts, making comparison meaningless.

**Impact:**
- `mlx_lm_quantized_kv_b8`: FAIL (captured real logits, but cascade divergence)
- `turboquant_v2_b4_gs64`: FAIL (same reason)
- `polar_reference_offline_b4`: FAIL (same reason)
- `rfsn_v10_k8_v5`: PENDING_LOGIT_GATE (custom generator can't capture yet)

**Fix required:** Teacher-forced (prompted) logit comparison — see [roadmap_alpha9.md](roadmap_alpha9.md).

## Candidate statuses (post Alpha 8.3)

| Candidate | Status | Blocker |
|-----------|--------|---------|
| mlx_lm_baseline | CONTROL | Not a candidate |
| mlx_lm_quantized_kv_b8 | CONTROL | **Logit gate methodology flaw** |
| rfsn_v10_k8_v5_gs32 | BASELINE | Custom generator logit capture missing |
| rfsn_v10_k8_v5_gs64 | BASELINE | Custom generator logit capture missing |
| rfsn_v11_offline_asymmetric_kv_k8v4_gs64 | OFFLINE_ONLY | Real cache injection missing |
| turboquant_v2_b4_gs64 | EXPERIMENTAL | **Logit gate methodology flaw** |
| polar_reference_offline_b4_d128 | REFERENCE_ONLY | **Logit gate methodology flaw** |

## Current Limitations (Post P0/P1 Fixes)

| Component | Status | Limitation |
|-----------|--------|------------|
| **Metal Kernel** | P1 Scaffold | True packed Metal kernel exists as scaffold only; actual GPU dispatch not yet functional. Falls back to CPU reference. |
| **Dense Reconstruction** | Violates Invariant | `metal_dense_reconstruction_violates_invariant` path explicitly flagged; reconstructs full dense KV history before attention. |
| **Logit Capture** | Methodology Issue | Teacher-forced logit comparison is the correct methodology, but cascade divergence from independent greedy decodes remains a problem. |
| **Promotion** | No Candidates | No candidates are currently promotion-eligible due to incomplete proof bundles and unproven quality gates. |
| **Wheel Build** | Fixed | P0 fix ensures static versioning prevents `0.0.0` builds from source ZIP. |
| **Registry Tests** | Fixed | P0 fix separates declared vs available candidates for portable test execution. |

## Roadmap

See [roadmap_alpha9.md](roadmap_alpha9.md) for the detailed path forward.

Phase A (critical): Fix the logit gate methodology → teacher-forced comparison.
Phase B (high): Complete true packed Metal kernel implementation (vectorized QK, full decode, online softmax).
Phase C (high): Candidate hardening once measurement is honest.
Phase D (medium): Benchmark expansion (larger models, longer contexts).
Phase E (low/deferred): CUDA backend, server hardening.
Phase F (research): Sparse decode, QJL, adaptive controller — indefinite deferral.
