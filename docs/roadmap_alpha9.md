# MLX-RFSN Fusion — Roadmap to Alpha 9 and Beyond

> This roadmap incorporates the lessons from Alpha 8.3. It is honest about what is blocked, what is feasible, and what is premature.

## Current Reality (Alpha 8.4)

**No candidate is promotion eligible.  Teacher-forced alignment is fixed, runtime path is instrumented, and honest artifacts are regenerated.**

| Candidate | Logit Gate | Memory Gate | Real Cache | Blocker |
|-----------|------------|-------------|------------|---------|
| mlx_lm_baseline | CONTROL | — | yes | Not a candidate |
| mlx_lm_quantized_kv_b8 | **FAIL** | PASS | yes | Per-prompt gate failures (aggregate is close) |
| rfsn_v10_k8_v5_gs32 | **FAIL** | FAIL | yes | Severe quality degradation when runtime path is exercised |
| rfsn_v10_k8_v5_gs64 | **FAIL** | FAIL | yes | Severe quality degradation when runtime path is exercised |
| rfsn_v11_offline_asymmetric | PENDING | PENDING | **no** | Offline-only, no injection |
| turboquant_v2_b4_gs64 | **FAIL** | PASS | yes | Quality fails gate thresholds |
| polar_reference_offline_b4 | **FAIL** | PASS | yes | Quality fails gate thresholds |
| turbo_polar_k4_qjl64 | **FAIL** | PASS | yes | Quality fails gate thresholds |

**Critical discovery:** Previous artifacts claiming RFSN v10 "perfect 1.0 logit match" were invalid. The `capture_logprobs` adapter was silently failing (returning `None`) due to:
1. `mx.eval([c.state for c in cache_list])` crashing on fresh `KVCache` objects before prefill (`keys` is `None`)
2. After fixing prefill order, `maybe_quantize_kv_cache` replacing caches with `QuantizedKVCache` whose `update_and_fetch` returns tuples, crashing the SDPA wrapper (`'tuple' object has no attribute 'ndim'`)

After fixing both bugs, the RFSN v10 runtime path is actually exercised and produces honest metrics:
- `logit_cosine` ~0.98 (threshold 0.999)
- `KL` ~13.5 (threshold 0.1)
- `top5_overlap` ~0.02 (threshold 0.85)
- Speed ~5.5 tps vs baseline ~50 tps (10x slower)
- Working-set memory ~1526 MB vs baseline ~975 MB (higher, due to per-step full-shape cache storage)

This means the RFSN v10 runtime path is functionally broken for production use. The `BASELINE_POLICIES` registry retains the config as a historically validated preset, but it has **not** been proven to work through its own runtime instrumentation.

**Status:**
- Teacher-forced capture scaffold exists.
- Token alignment bug identified and fixed in `rfsn_v11/candidates/logit_capture.py` and `rfsn_v10_adapter.py`.
- `tests/benchmarks/test_teacher_forced_alignment.py` proves correct token feeding.
- Artifacts are wrapped with `promotion_allowed=false` and `methodology_status=TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION`.
- Markdown now correctly renders "Promotion no" when promotion is disabled.
- `token_sequence_hash` field added; promotion blocked when empty.
- Gate thresholds are single-source-of-truth in `quality_gates.py` and included in artifact metadata.
- `failed_gate_reasons` added to every failing row.
- RFSN v10 proof trace is `trace_type: runtime_instrumented` with honest non-zero counters (`cache_bytes_read_actual > 0`, `decode_quantized_fetch_events > 0`), but quality is unacceptable for promotion.
- `cache_policy.py` PROMOTED_POLICIES is empty; rfsn_v10 is in BASELINE_POLICIES only.
- `winner.json` reset to `NO_PROMOTION_ELIGIBLE_CANDIDATE`.

## Critical Finding: The Logit Comparison Methodology is Flawed

Alpha 8.3 compares logits by running two independent greedy decodes:

1. Baseline: `generate_step(model, temp=0.0)` → tokens [a, b, c, d]
2. Candidate: `generate_step(model, temp=0.0, kv_bits=8)` → tokens [a, b, x, y]

If token 2 differs (c vs x), the remaining logits are computed on **completely different input contexts**. Comparing the logit distribution at position 2 (for token "d" given "a,b,c") vs position 2 (for token "y" given "a,b,x") is meaningless.

Even a tiny quantization shift can flip the argmax at any token, causing cascade divergence. This makes the current logit gate a test of "does quantization produce *exactly* identical greedy tokens?" not "does quantization preserve logit quality?"

**The fix:** Teacher-forced (prompted) logit comparison.

1. Run baseline greedy decode → text T.
2. Feed the **exact same token sequence** T through the candidate in teacher-forced mode (no sampling, just forward pass at each position).
3. Compare per-position logits between baseline and candidate given the **same input tokens**.

This measures: "Given the same context, how much does the candidate's logit distribution differ from baseline?" — which is the actual quality question.

## Phase A — Fix the Logit Gate (Required before any promotion)

**Priority: CRITICAL. Nothing else matters until this works.**

### A1. Fix teacher-forced token alignment (COMPLETED)
- ~~Add `capture_teacher_forced_logprobs`~~ — EXISTS.
- ~~Update `benchmarks/kv_shootout.py`~~ — DONE.
- **FIXED:** Loop was feeding `gen_ids[1:]` instead of `gen_ids[:-1]`.  Now feeds the correct token at each step.
- **TEST ADDED:** `tests/benchmarks/test_teacher_forced_alignment.py` proves correct feeding.

### A2. Enable logit capture for RFSN v10 custom generator (COMPLETED — REVEALS CRITICAL BUG)
- RFSN v10 uses `RFSNGenerator.generate()` which is a custom loop.
- `capture_logprobs` in `rfsn_v10_adapter.py` was silently failing due to `KVCache.state` crash on fresh caches and `QuantizedKVCache` tuple return breaking the SDPA wrapper.
- **FIXED:** Prefill order corrected; `maybe_quantize_kv_cache` removed so RFSNRuntime's own KVManager handles quantization; wrapper updated to fall through on tuple keys/values.
- **HONEST RESULT:** RFSN v10 runtime path is now exercised and produces terrible quality (logit_cosine ~0.98, KL ~13.5, top5 ~0.02). The previous "perfect 1.0" was a measurement artifact caused by silent adapter failure.

### A3. Validate thresholds on known-good configurations (COMPLETED)
- Corrected teacher-forced comparison run on `mlx_lm_quantized_kv_b8`.
- Result: aggregate metrics are close, but strict per-prompt gate still fails on some prompts.
- This is a conservative (not buggy) result — thresholds may be too strict for upstream quantized KV.
- **Status:** Methodology is validated; thresholds may need tuning for promotion.

### A4. Re-run full logit gate with fixed methodology (COMPLETED)
- All artifacts regenerated under corrected teacher-forced gate.
- Actual outcome: mlx_lm_quantized_kv_b8 fails strict per-prompt gate (aggregate is close).
- Actual outcome: TurboQuant V2 and Polar fail for real quality reasons, not methodology artifacts.
- **Status:** Teacher-forced rerun complete; artifacts are honest.

### A5. Post-rerun promotion validation (COMPLETED — BLOCKED BY QUALITY)
- Rerun is complete; runtime counters prove the compressed path was exercised (`cache_bytes_read_actual > 0`, `decode_quantized_fetch_events > 0`).
- `methodology_status` is `TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION`.
- `token_sequence_hash` is non-empty.
- `PROMOTED_POLICIES` in `cache_policy.py` remains empty; matches `winner.json` (no winner).
- **Remaining blocker:** RFSN v10 runtime path quality is unacceptable (logit_cosine ~0.98, KL ~13.5, top5 ~0.02, 10x slower, higher memory). Promotion is impossible until the runtime quantization algorithm is fixed or replaced.

**Timeline:** Next milestone is Alpha 8.5 or 9.0 — fix RFSN v10 runtime quantization quality, OR pivot to a candidate that actually passes gates.
**Risk:** High. The RFSN v10 runtime quantization path may need fundamental redesign.

## Phase B — Candidate Hardening (Depends on Phase A)

**Priority: HIGH. Only start after Phase A validates the measurement.**

### B1. TurboQuant V2 quality tuning
- If TQ V2 fails teacher-forced logit gate, investigate:
  - Is 4-bit quantization + rotation too aggressive for small models?
  - Does the SDPA patch introduce numerical drift?
  - Should we test with head_dim=128 (rotation designed for 128) instead of 64?
- Possible fixes: increase bits to 6, test on larger models where quantization error is relatively smaller.

### B2. Polar reference quality tuning
- Same investigation as TQ V2.
- Polar dequantizes on fetch — this might introduce per-step noise that accumulates.

### B3. RFSN v11 real cache injection
- This is the only path to getting RFSN v11 out of OFFLINE_ONLY status.
- Requires: direct cache injection into MLX-LM's `generate_step`, real `nbytes` accounting.
- This is substantial engineering. Defer until at least one other candidate is promoted.

**Timeline:** 2-4 weeks per candidate.
**Risk:** Medium. May discover fundamental quality limits of specific approaches.

## Phase C — Benchmark and Artifact System Hardening

**Priority: MEDIUM. Parallel with Phase B.**

### C1. Run on larger models
- Current artifacts use Qwen2.5-0.5B (head_dim=64).
- Run on Qwen2.5-1.5B (head_dim=128) or larger.
- Larger models are more forgiving of quantization error — may reveal candidates that work at scale but fail on tiny models.

### C2. Run on longer contexts
- Current max_tokens=50.
- Test with 200-500 tokens to measure drift accumulation.
- Test with longer prompts (1K+ context) to measure prefill quality.

### C3. Cross-model validation
- Test on Mistral, Llama, Phi architectures.
- Different head_dim, different attention patterns may affect candidate quality.

### C4. Add per-prompt artifact history
- Currently artifacts are overwritten per-run.
- Keep dated artifact directories for trend tracking.

**Timeline:** Ongoing.
**Risk:** Low. Just more compute time.

## Phase D — Platform Expansion (Defer until after first promotion)

**Priority: LOW for Alpha. Required for production.**

### D1. CUDA backend
- The original plan says 3-6 months.
- This is correct — it's a major engineering project.
- **Do not start until at least one candidate is promoted on Apple Silicon.**
- Rationale: Without a proven winning candidate, CUDA investment is premature.

### D2. Enhanced CPU fallback
- 1-2 months.
- Useful for CI and non-MLX development.
- Can be done in parallel with other work since it's additive.

### D3. Production server hardening
- FastAPI server exists but is research-grade.
- Rate limiting, auth, monitoring, Docker — standard production work.
- **Do not market as production-ready until a promoted candidate exists.**

**Timeline:** 3-6 months total.
**Risk:** Low technical risk, high time investment.

## Phase E — Experimental Feature Maturation (Defer indefinitely)

**Priority: LOW / RESEARCH.**

### E1. Sparse decode
- Currently disabled by default.
- End-to-end speedup not proven.
- Requires extensive research, not engineering.

### E2. QJL score correction
- QJL fails its own artifact (MAE too high).
- May require algorithmic redesign, not tuning.

### E3. Adaptive sparse controller
- Not validated.
- Control-theory problem more than an engineering problem.

### E4. True bit-packing for >8-bit
- Currently falls back to uint32.
- Only matters if >8-bit becomes a common config.
- 8-bit is the validated sweet spot today.

**Recommendation:** Keep these as experimental flags. Do not invest heavily until Phase A-C produce a promoted baseline.

## What NOT to do

1. **Do not claim beta or production readiness.** Alpha 8.3 is honest. Stay honest.
2. **Do not add new algorithms.** Fix measurement first.
3. **Do not start CUDA backend before a candidate promotes.** That would be building a foundation for a house without a blueprint.
4. **Do not lower thresholds to fake a promotion.** If mlx_lm_quantized_kv_b8 fails teacher-forced comparison, the thresholds are wrong, not the measurement.

## Definition of Done for Alpha 9

Alpha 9 is done when:

1. Teacher-forced logit comparison is implemented and validated.
2. At least one candidate (likely mlx_lm_quantized_kv_b8 or rfsn_v10) passes the full logit gate.
3. That same candidate passes the memory gate.
4. The promotion report names at least one eligible candidate.
5. `winner.json` names the winner with honest metrics.
6. No candidate is falsely promoted.
7. CPU gates pass, benchmark tests pass, wheel builds.

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| A: Fix logit gate | 1-2 weeks | 2 weeks |
| B: Candidate hardening | 2-4 weeks | 6 weeks |
| C: Benchmark expansion | ongoing | — |
| D: Platform expansion | 3-6 months | — |
| E: Experimental maturation | indefinite | — |

## Conclusion

The path to Alpha 9 is narrow and specific: **fix the logit comparison methodology**. Everything else — CUDA, sparse decode, server hardening — is premature until the benchmark can honestly measure candidate quality.

The good news: Alpha 8.3 has all the infrastructure. The artifact system, gate rules, and candidate adapters are honest. The only missing piece is a comparison methodology that actually measures what it claims to measure.
