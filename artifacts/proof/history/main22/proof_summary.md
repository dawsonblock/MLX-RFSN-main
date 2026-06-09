# main12 Proof Summary

Release: Main 22 — Proof-Consistent Clean Block-Aware KV Reconstruction Release
Generated: 2026-06-02T07:26:04.849859+00:00

## KV Cache Status
- Synthetic tensor proof: pass
- Compression ratio: 0.265625 (best case)
- Key quality: pass (cosine >= 0.999)
- Value quality:
  - 8-bit V path: ~0.99998 cosine
  - 3-bit V path: ~0.970 cosine (not >= 0.999; threshold is 0.90)
  - release threshold: pass above 0.90

## Sparse Decode Status
- Default: disabled
- Reason: shipped sparse audit quality has not cleared deployment threshold
- Threshold: sparse_audit_cosine >= 0.90
- Current minimum: 0.4996378421783447
- Current maximum: 0.8836325407028198
- Recommendation: dense default, sparse opt-in only

## Kernel Benchmark Status
Generated from kernel_benchmark.json

### Cross-Mode Equivalence (vs. Python-only gold with matching quant config)
- All 6 modes pass gold validation across 4 shapes
- Full-route modes: metal_multikernel_dequant_wht_sign, metal_fused_dequant_wht_sign
- Ablation modes: metal_multikernel_dequant, metal_multikernel_dequant_wht, metal_multikernel_dequant_sign
- Best speedup: ~30x (metal_multikernel_dequant vs sequential_reference)
- Typical full-route speedup: ~12-21x
- Fallback observed: no

### Internal Self-Consistency (Metal retrieve vs. Python retrieve, same config)
- All 6 modes pass internal validation across 4 shapes
- Key cosine: 1.000000, Value cosine: 1.000000 for all full-route entries
- No internal inconsistencies detected

### Fused Kernel Standalone
- See fused_kernel_benchmark.json for standalone fused-kernel proof
- Cosine vs. reference: ~1.000000, max_abs_diff: 0.0

## MLX Test Status
- Apple Silicon run: yes
- Hardware: Apple M2 Pro, 16 GB RAM, macOS Darwin 25.2.0
- MLX version: 0.31.2
- Result: 258 tests passed, 0 failed, 0 skipped
- Metal fallback: no

## Real Model Validation
- Class: tiny-random smoke test
- Limitations: Uses hf-internal-testing-tiny-random-LlamaForCausalLM with random weights
- Status: Verifies plumbing only, does not prove quality on a real LLM
- For production-quality evidence: validation with a real non-random model is required

## Files
- kv_cache_runs.json
- e2e_scenarios.json
- kernel_benchmark.json
- real_model_validation.json
- mlx_test_summary.md

## Highlights
- Fastest KV retrieve: 0.61ms (shape=(1, 8, 1024, 64)|k=8|v=8|incoherent=False)
- KV value quality (same run): cos=1.0000, rel_mae=0.0063, max_abs=0.0199
- Best sparse scenario: sparse_topk_075_sink1_recent2 miss=11.06ms, hit=7.37ms, sparse_cos=0.8836325407028198
- Worst sparse scenario: sparse_disabled_by_default hit_mode=dense_requested sparse_cos=None sparse_rel_mae=None
- Dense decode path: miss=9.68ms, hit=5.33ms, mode=dense_requested

## Absolute Quality
- Sparse quality: warn (min=0.4996378421783447, threshold=0.900)
- Quant quality: pass (min=0.9703969359397888, threshold=0.950)
- KV cache quality: pass (min=0.9699910283088684, threshold=0.900)
- Sparse default: disabled
- WARNING_UNSAFE_FOR_LLM_DEPLOYMENT
- Sparse deployment threshold met: no
- Recommended default: dense (sparse decode remains experimental and should default to disabled)

## Optimization Sweep
- Configurations tested: 8
- Shapes tested: 4
- Total rows: 32
- Best quality config: 4bit_4_4_gs64 (worst cosine ~0.99424)
- Baseline config: baseline_8_3_gs64 (worst cosine ~0.96998)
- Rejected configs: 2-bit variants (worst cosine ~0.70094 to ~0.78041)
- Optimization sweep is included as tabular JSON only.

Recommended default remains 8-bit K / 3-bit V / group_size 64 for compression-oriented testing.
4-bit K / 4-bit V / group_size 64 is the quality-oriented candidate.
2-bit configurations are rejected for current use.

## Next Checks
- Compare these artifacts against previous runs for trend regressions.
- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.
