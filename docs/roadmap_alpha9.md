# MLX-RFSN Fusion — Roadmap to Alpha 9 and Beyond

> This roadmap incorporates the lessons from Alpha 8.3. It is honest about what is blocked, what is feasible, and what is premature.

## Current Reality (Alpha 8.3)

**No candidate is promotion eligible.**

| Candidate | Logit Gate | Memory Gate | Real Cache | Blocker |
|-----------|------------|-------------|------------|---------|
| mlx_lm_baseline | CONTROL | — | yes | Not a candidate |
| mlx_lm_quantized_kv_b8 | **FAIL** | PASS | yes | Logit comparison methodology flawed |
| rfsn_v10_k8_v5_gs32 | PENDING | PASS | yes | Custom generator cannot capture logits |
| rfsn_v10_k8_v5_gs64 | PENDING | PASS | yes | Custom generator cannot capture logits |
| rfsn_v11_offline_asymmetric | PENDING | PENDING | **no** | Offline-only, no injection |
| turboquant_v2_b4_gs64 | **FAIL** | PASS | yes | Logit comparison methodology flawed |
| polar_reference_offline_b4 | **FAIL** | PASS | yes | Logit comparison methodology flawed |

**The common failure mode:** All candidates that actually capture real logits (mlx_lm_quantized, TurboQuant V2, Polar) fail the gate with low cosine / high KL / low top-k overlap. This is suspicious — it suggests the comparison methodology itself is broken, not that every candidate is bad.

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

### A1. Implement teacher-forced logit comparison
- Add `teacher_forced_logits(model, tokenizer, prompt, target_text)` that runs a forward pass at each token position without sampling.
- Modify `capture_generation_logprobs` or add a new `capture_teacher_forced_logprobs`.
- Update `benchmarks/kv_shootout.py` full-logit-gate mode to use teacher-forced comparison.

### A2. Enable logit capture for RFSN v10 custom generator
- RFSN v10 uses `RFSNGenerator.generate()` which is a custom loop.
- Add a teacher-forced path to `RFSNGenerator` or create a standalone forward-pass utility.
- This removes the "PENDING_LOGIT_GATE" blocker for the baseline candidate.

### A3. Validate thresholds on known-good configurations
- Run teacher-forced comparison on `mlx_lm_quantized_kv_b8` (maintained upstream).
- If it still fails, the thresholds are too strict.
- If it passes, we have a validated methodology.

### A4. Re-run full logit gate with fixed methodology
- Regenerate all artifacts.
- Expected outcome: mlx_lm_quantized_kv_b8 passes (it should — it's maintained upstream).
- Expected outcome: TurboQuant V2 either passes or fails for real quality reasons, not methodology artifacts.

**Timeline:** 1-2 weeks.
**Risk:** Low. This is a measurement fix, not an algorithm change.

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
