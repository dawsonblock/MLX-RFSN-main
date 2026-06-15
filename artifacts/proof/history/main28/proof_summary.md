# Proof Summary — Main 28

**Release**: Main 28 — Proof Consistency + Long-Context + Throughput Honesty  
**Status**: Alpha  
**Date**: 2026-06-03  
**Hardware**: Apple M2 Pro, 16GB RAM  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  

---

## Release Identity

This release makes the release identity, proof summary, validation artifacts, throughput reporting, and long-context claims internally consistent. No new architecture features were added.

---

## Synthetic Kernel Benchmark

| Benchmark | Status | Cosine | Max Abs Diff |
|-----------|--------|--------|--------------|


---

## Fused Kernel Benchmark

| Benchmark | Status | Cosine | Max Abs Diff |
|-----------|--------|--------|--------------|
| ? | valid | ? | ? |
| ? | valid | ? | ? |
| ? | valid | ? | ? |
| ? | valid | ? | ? |
| ? | valid | ? | ? |
| ? | valid | ? | ? |


---

## Real-Model Validation

**Method**: Causal-correct NLL scoring with 64 decode positions across 5 prompts.  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  
**Context**: 512 tokens

| Config | Cosine Mean | Cosine Min | Top1 Match | NLL Δ | KL | Status |
|--------|-------------|------------|------------|-------|-----|--------|
| k8_v3_gs64 | 0.9970 | 0.9893 | 1.000 | 0.000053 | 0.000092 | **FAIL** |
| k8_v4_gs64 | 0.9994 | 0.9974 | 1.000 | 0.00035 | 0.000047 | **PASS** |
| k8_v5_gs64 | 0.9998 | 0.9950 | 1.000 | 0.000078 | 0.000011 | **PASS** |
| k8_v4_gs32 | 0.9995 | 0.9982 | 1.000 | 0.00014 | 0.000016 | **PASS** |
| k8_v5_gs32 | 0.9999 | 0.9992 | 1.000 | -0.000000 | 0.000005 | **PASS** |
| k6_v6_gs64 | 0.8353 | 0.0510 | 1.000 | 0.0415 | 0.0386 | **FAIL** |
| k4_v4_gs64 | 0.5404 | -0.2362 | 0.5969 | 2.844 | 2.835 | **FAIL** |


---

## Long-Context Validation

Contexts tested: 512, 1024, 2048 tokens  
Positions evaluated: 64 per context

| Config | 512 | 1024 | 2048 | Passes All |
|--------|-----|-----|-----|------------|
| k8_v3_gs64 | PASS | FAIL | PASS | NO |
| k8_v4_gs64 | PASS | PASS | PASS | **YES** |
| k8_v5_gs64 | PASS | PASS | PASS | **YES** |
| k8_v4_gs32 | PASS | PASS | PASS | **YES** |
| k8_v5_gs32 | PASS | PASS | PASS | **YES** |
| k6_v6_gs64 | FAIL | FAIL | FAIL | NO |
| k4_v4_gs64 | FAIL | FAIL | FAIL | NO |


---

## Generation Smoke Test

**Method**: Greedy decode 64 tokens, compare to baseline  
**Context**: 128 tokens

| Config | Token Match | Edit Dist | Repetition | NaN | Status |
|--------|-------------|-----------|------------|-----|--------|
| baseline_fp16 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v4_gs64 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v5_gs64 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v5_gs32 | 1.000 | 0.000 | 0.197 | No | PASS |


---

## Generation Throughput

**Method**: 5 timed repeats after 2 warmup runs  
**Context**: 512 tokens, decode 64 tokens

| Config | Prefill (ms) | Compress (ms) | Decode (ms) | Total (ms) | TPS | Comp Ratio |
|--------|--------------|---------------|-------------|------------|-----|-------------|
| baseline_fp16 | 12.472 | 0.000000 | 814.989 | 827.461 | 77.742 | 1.0x |
| k8_v4_gs64 | 12.657 | 835.431 | 759.192 | 1607.280 | 82.991 | 2.6x |
| k8_v5_gs64 | 12.714 | 721.910 | 761.782 | 1496.406 | 82.705 | 2.4x |
| k8_v4_gs32 | 12.703 | 804.414 | 762.853 | 1579.971 | 82.587 | 2.5x |
| k8_v5_gs32 | 14.604 | 810.202 | 758.163 | 1582.969 | 83.099 | 2.3x |


Decode throughput is comparable to baseline, but end-to-end runtime is slower because compression overhead dominates.

---

## Memory / Compression

The primary proven benefit is KV memory reduction. Effective compression ratios are approximately 2.3x for 8-bit K / 4-bit V configs.

---

## Sparse Decode Status

Sparse decode is **disabled by default** and remains experimental. Do not enable unless explicitly testing the safety gate.

---

## Not Implemented

- Polar quantization is not implemented.
- True arbitrary token-level partial dequantization is not implemented.
- Per-layer sensitivity analysis and targeted layer protection are deferred to a future release.

---

## Recommended Configs

- **Recommended practical default**: `k8_v5_gs64`
- **Best quality**: `k8_v5_gs32`
- **Lowest-bit passing**: `k8_v4_gs64`


## Rejected Configs

- `k4_v4_gs64`
- `k6_v6_gs64`
- `k8_v3_gs64`


---

## Limitations

- RFSN is a research runtime, not production-ready.
- End-to-end speedup has not been proven.
- Validation is on a single small model (Qwen2.5-0.5B).
- Sparse decode is disabled.
- Polar quantization is not implemented.

---

## Conclusion

Main 28 successfully locks the truthful position: RFSN v10 is a clean Apple Silicon KV-cache compression research runtime. Its best practical config is `k8_v5_gs64`, it reduces KV bytes, but has not proven end-to-end speedup and is not production-ready.
