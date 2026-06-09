# Experimental Proof Artifact Index

**Release:** experimental  
**Stable Default:** `k8_v5_gs64`  
**QJL Status:** failed, disabled  
**Promoted to Default:** false  
**Experimental Status:** research_only  
**Teacher-Forced Baseline Status:** valid (identity guard confirmed)  
**Promotion Blocked By:** stable_teacher_forced_drift_unexplained, experimental_decode_update_failure, real_gen_benchmark_methodology_discrepancy, qjl_failed  
**Decode-Update Diagnosis:** stable configs pass decode-update; experimental configs fail  
**Step-Trace Finding:** stable configs pass all 128/128 teacher-forced steps (cosine ≥ 0.995); real-gen bulk benchmark discrepancy under investigation

## Artifact Manifest

| Artifact | File | Status | Rows | Purpose |
|----------|------|--------|------|---------|
| comparison | `comparison_summary.json` | executed | — | 0.5B model quality comparison across configs |
| memory | `memory_accounting.json` | executed | — | Per-config compressed memory with real-model basis |
| throughput | `throughput.json` | executed | — | Synthetic KV throughput benchmark |
| real_generation_throughput | `real_generation_throughput.json` | executed | — | Bulk teacher-forced + free-running generation benchmark (methodology under investigation) |
| teacher_forced_step_trace | `teacher_forced_step_trace.json` | **executed** | **1536** | Per-step teacher-forced quality trace with distribution-shift diagnostics |
| decode_update_trace | `decode_update_trace.json` | executed | 70 | Step-by-step decode path quality trace (free-running greedy) |
| decode_append_kv_diff | `decode_append_kv_diff.json` | executed | 10 | Old-cache preservation and new-token K/V append diagnostics |
| kv_roundtrip_by_context | `kv_roundtrip_by_context.json` | executed | — | Direct KV quantizer roundtrip by prompt length |
| prefill_decode_split | `prefill_decode_split.json` | executed | — | Prefill-vs-decode isolation A/B/C/D (free-running greedy; reconciliation fields added) |
| short_prompt_drift_trace | `short_prompt_drift_trace.json` | executed | — | Step-by-step short-prompt logit drift trace |
| qjl | `qjl_attention_score.json` | executed | — | QJL benchmark result (failed) |
| layer_policy | `layer_policy.json` | executed | — | Conservative per-layer policy |
| qwen_1_5b | `qwen_1_5b/` | executed | — | 1.5B model validation directory |

## Decode Diagnostics Summary

### decode_update_trace.json (70 rows)

Step-level decode comparison: FP16 vs quantized, per config × prompt_tokens × decode_step.

| Config | Result |
|--------|--------|
| k8_v5_gs64 | All steps pass (cosine ≥ 0.999) |
| k8_v5_gs32 | All steps pass (cosine ≥ 0.999) |
| turbo_polar | All steps degraded (cosine 0.65–0.99, top5 0.2–0.8) |
| adaptive | All steps degraded (cosine 0.67–0.96, top5 0.4–0.8) |
| experimental_hybrid | All steps degraded (cosine 0.76–0.99, top5 0.2–1.0) |

### decode_append_kv_diff.json (10 rows)

Pre/post-append K/V analysis: old-cache preservation vs new-token quantization error.

| Config | old_k_cos | new_k_cos | cache_ok |
|--------|-----------|-----------|----------|
| k8_v5_gs64 | ≥ 0.9999 | ≥ 0.9994 | true |
| k8_v5_gs32 | ≥ 0.9999 | ≥ 0.9996 | true |
| turbo_polar | ≥ 0.9983 | ≥ 0.9573 | true |
| adaptive | ≥ 0.9983 | ≥ 0.9326 | true |
| experimental_hybrid | ≥ 0.9960 | ≥ 0.9443 | true |

**Finding:** Old-cache is not corrupted by appending. New-token K/V quantization error is the primary source of degradation in experimental configs. Stable configs have high new-token quality.

## Teacher-Forced Step Trace Summary (teacher_forced_step_trace.json, 1536 rows)

Per-step teacher-forced quality trace: same baseline FP16 token sequence fed to both FP16 and compressed paths for every decode step. Prompt source: "The quick brown fox jumps over the lazy dog." (same as real_generation_throughput.py). 128 steps per run.

| Config | Prompt Tokens | Pass Steps | Avg Cosine | Min Cosine | Step-0 Cosine | Finding |
|--------|---------------|-----------|-----------|-----------|--------------|---------|
| k8_v5_gs64 | 128 | 128/128 | 0.9999 | 0.9995 | 0.9996 | **stable** |
| k8_v5_gs64 | 512 | 128/128 | 0.9999 | 0.9995 | 0.9951 | **stable** |
| k8_v5_gs32 | 128 | 128/128 | 0.9999 | 0.9997 | 0.9997 | **stable** |
| k8_v5_gs32 | 512 | 128/128 | 0.9999 | 0.9997 | 0.9981 | **stable** |
| turbo_polar | 128 | 0/128 | ~0.81 | ~0.56 | 0.91 | degraded from step 0 |
| turbo_polar | 512 | 1/128 | ~0.71 | ~0.41 | 0.97 | degraded |
| adaptive | 128 | 0/128 | ~0.81 | ~0.55 | 0.91 | degraded from step 0 |
| adaptive | 512 | 0/128 | ~0.71 | ~0.40 | 0.97 | degraded |
| experimental_hybrid | 128 | 28/128 | ~0.98 | ~0.92 | 0.98 | marginally degraded |
| experimental_hybrid | 512 | 0/128 | ~0.91 | ~0.84 | 0.98 | degraded |

**Key finding:** Stable configs (k8_v5_gs64, k8_v5_gs32) pass all 128/128 teacher-forced steps at both prompt lengths. This directly contradicts the bulk average (cosine ~0.955) reported by `real_generation_throughput.json`. The discrepancy is under investigation.

## Real-Generation Benchmark Methodology Note

`real_generation_throughput.json` reports cosine ~0.955 for stable configs. `teacher_forced_step_trace.json` shows cosine ≥ 0.995 for all 128 steps on the same foxdog prompt.

The two benchmarks differ in methodology:
- **teacher_forced_step_trace**: Shared token sequence, fixed FP16 continuation tokens, per-step comparison  
- **real_generation_throughput teacher_forced**: Bulk average across 128 positions using `_teacher_forced_logits()` function

Both use the same foxdog text prompt and the same forced token sequence. The step trace is more granular and more trustworthy. The bulk benchmark's 0.955 figure is under investigation for a comparison artifact.

`prefill_decode_split.json` uses **free-running greedy** decode per mode (not teacher-forced) — each mode picks its own token sequence. This makes it a different measurement. Reconciliation fields (`continuation_mode`, `token_sequence_source`) have been added to all result rows.

## Classification Rules

- No experimental mode may be classified as a candidate without real-generation data.
- Teacher-forced baseline must be exact identity (cosine=1.0, top5=1.0, KL=0.0).
- If baseline identity fails, all configs are blocked as `needs_valid_teacher_forced_baseline`.
- Teacher-forced real-generation failure is a hard reject (`rejected_generation_quality`).
- Free-running divergence without teacher-forced failure is `generation_divergence_observed`.
- No config using raw `uint32` fallback (>8-bit) may be called memory-optimized.
- QJL remains disabled until its attention score benchmark passes.
- The default runtime mode is locked to `k8_v5_gs64`.

## Current Config Status

| Config | Classification |
|--------|---------------|
| baseline_fp16 | reference |
| k8_v5_gs64 | stable default, locked, teacher-forced drift under investigation |
| k8_v5_gs32 | quality candidate, teacher-forced drift under investigation |
| experimental_hybrid | free-running match, teacher-forced failed |
| turbo_polar | rejected generation divergence |
| adaptive | rejected generation divergence |
| QJL | disabled |
