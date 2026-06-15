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
- [x] **True packed kernel implemented (K8/V8)**: ``PackedV4AttentionKernel`` in ``packed_v4_attention.py`` compiles via MLX inline Metal, reads real ``PackedBlockV4`` uint32 BHTW codes and BHTG scales, applies hash signs and WHT domain dot products, and passes differential tests against the blockwise reference. Gated by ``RFSN_ENABLE_TRUE_PACKED=1``.
- [x] **Incremental decode correctness (P0)**: Wrapper now merges sealed packed blocks, live staging K/V, and dense residual K/V via exported softmax statistics. Empty-cache and pre-seal behavior supported.
- [x] **Statistics crash fixed (P0.6)**: ``get_backend_stats()`` reads ``num_key_blocks`` / ``num_value_blocks`` instead of nonexistent ``num_blocks``.
- [x] **Unsupported masks rejected (P0.7)**: Non-causal/arbitrary masks raise in strict mode; fallback to reference in non-strict mode.
- [x] **Strict mode truthful (P1.1)**: In strict direct-packed mode, dense-reconstruction and reference backends are forbidden. Fallback only occurs when ``strict=False``.
- [x] **Real kernel self-test (P1.3)**: ``_self_test()`` compiles the actual ``_PACKED_V4_KERNEL_K8`` source, not a trivial placeholder.
- [x] **Complete block validation (P1.4)**: ``_validate_blocks`` enforces contiguous positions, consistent geometry, consistent codec metadata, and exact buffer shapes across all blocks.
- [x] **Old prototypes removed**: ``TruePackedMLXInline``, ``TruePackedAttentionMetalV2``, ``TruePackedAttentionMetal``, ``FusedPackedAttentionMetal``, and associated ``.metal`` shader stubs fully deleted from the repository.
- [x] **Differential testing framework**: ``test_packed_v4_attention.py`` proves exact numerical match for single block, multiple blocks, GQA, causal/non-causal, contract validation, softmax stats, and packed+dense region merging.
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
| **Metal Kernel** | P1 Implemented (K8/V8) | ``PackedV4AttentionKernel`` reads real ``PackedBlockV4`` blocks, decodes on-the-fly in WHT domain, and passes differential tests against the blockwise reference. Gated by ``RFSN_ENABLE_TRUE_PACKED=1``; not yet the default dispatch path. Incremental decode now correctly includes staging and residual. |
| **Dense Reconstruction** | Violates Invariant | `metal_dense_reconstruction_violates_invariant` path explicitly flagged; reconstructs full dense KV history before attention. Still available as non-strict fallback only. |
| **Logit Capture** | Methodology Issue | Teacher-forced logit comparison is the correct methodology, but cascade divergence from independent greedy decodes remains a problem. |
| **Promotion** | No Candidates | No candidates are currently promotion-eligible due to incomplete proof bundles and unproven quality gates. |
| **Wheel Build** | Fixed | P0 fix ensures static versioning prevents `0.0.0` builds from source ZIP. |
| **Registry Tests** | Fixed | P0 fix separates declared vs available candidates for portable test execution. |
| **Performance Architecture** | Measured | Scalar shader (one thread per q_head/q_token). Concatenation is now incremental O(T) via persistent cached arrays. ``test_true_packed_performance_vs_dense`` archives wall-clock latency; scalar prototype is currently slower than dense (expected). ``test_true_packed_proof_bundle`` generates JSON artifacts with per-layer execution contracts and zero-materialization proof. |

## Roadmap

See [roadmap_alpha9.md](roadmap_alpha9.md) for the detailed path forward.

Phase A (critical): Fix the logit gate methodology → teacher-forced comparison.
Phase B (high): Integrate ``PackedV4AttentionKernel`` as default dispatch path (remove opt-in gate after broader Apple-Silicon validation).
Phase C (high): Extend kernel to V5 and sub-byte formats (2–7 bit) once K8 is fully hardened.
Phase D (medium): Candidate hardening once measurement is honest; benchmark expansion (larger models, longer contexts).
Phase E (low/deferred): CUDA backend, server hardening.
Phase F (research): Sparse decode, QJL, adaptive controller — indefinite deferral.
