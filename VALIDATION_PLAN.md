# Validation Plan for Additional Models and Contexts

Phase 9: Expand validation to additional models and contexts after correctness proven

## Current Status

The direct-packed K8/V8 path has been validated on:
- **Model**: Qwen2.5-0.5B-Instruct
- **Architecture**: Standard attention (num_q_heads == num_kv_heads)
- **Context**: Short to medium length (up to 512 tokens)
- **Platform**: Apple Silicon (MLX)

## Required Additional Validation

### 1. Model Family Validation

#### 1.1 GQA Models (Grouped Query Attention)
**Target Models:**
- Llama 2/3 (num_q_heads > num_kv_heads)
- Mistral (num_q_heads > num_kv_heads)
- Gemma (num_q_heads > num_kv_heads)

**Validation Steps:**
- [ ] Load GQA model successfully
- [ ] Verify packed attention handles GQA geometry correctly
- [ ] Run quality validation with token fixtures
- [ ] Verify no errors in generation
- [ ] Compare against baseline for quality metrics

**Test File:** `rfsn_v10/cache/tests/test_gqa_validation.py` (already created)

#### 1.2 Different Model Sizes
**Target Models:**
- Small: < 1B parameters (e.g., Qwen2.5-0.5B)
- Medium: 1-7B parameters (e.g., Qwen2.5-7B, Llama-3-8B)
- Large: > 7B parameters (e.g., Qwen2.5-14B, Llama-3-70B)

**Validation Steps:**
- [ ] Test small models (already done with Qwen2.5-0.5B)
- [ ] Test medium models for memory efficiency
- [ ] Test large models for scalability
- [ ] Verify packed block creation scales correctly
- [ ] Verify memory savings are proportional to model size

#### 1.3 Different Architectures
**Target Architectures:**
- Decoder-only (e.g., GPT-style)
- Encoder-decoder (e.g., T5-style)
- Mixture of Experts (e.g., Mixtral)

**Validation Steps:**
- [ ] Test decoder-only models (already done)
- [ ] Test encoder-decoder models if applicable
- [ ] Test MoE models for expert routing correctness
- [ ] Verify cache behavior for each architecture

### 2. Context Length Validation

#### 2.1 Short Context (< 128 tokens)
**Validation Steps:**
- [ ] Test with prompts < 128 tokens
- [ ] Verify block sealing behavior
- [ ] Verify quality metrics
- [ ] Verify memory accounting

#### 2.2 Medium Context (128-1024 tokens)
**Validation Steps:**
- [ ] Test with prompts 128-1024 tokens
- [ ] Verify multiple block creation
- [ ] Verify block position monotonicity
- [ ] Verify quality across block boundaries

#### 2.3 Long Context (> 1024 tokens)
**Validation Steps:**
- [ ] Test with prompts > 1024 tokens
- [ ] Verify many block creation (16+ blocks)
- [ ] Verify memory efficiency at scale
- [ ] Verify quality across many block boundaries
- [ ] Test file: `rfsn_v10/cache/tests/test_rope_validation.py` (already created)

### 3. Generation Scenario Validation

#### 3.1 Single-Turn Generation
**Validation Steps:**
- [ ] Test single prompt → response
- [ ] Verify quality metrics
- [ ] Verify runtime counters
- [ ] Verify memory accounting

#### 3.2 Multi-Turn Generation
**Validation Steps:**
- [ ] Test conversation with multiple turns
- [ ] Verify cache persistence across turns
- [ ] Verify quality metrics per turn
- [ ] Verify no requantization (requantized_token_count == 0)

#### 3.3 Streaming Generation
**Validation Steps:**
- [ ] Test streaming output
- [ ] Verify quality metrics for streaming
- [ ] Verify no quality degradation vs non-streaming
- [ ] Verify memory accounting during streaming

#### 3.4 Batch Generation
**Validation Steps:**
- [ ] Test multiple prompts in batch
- [ ] Verify cache isolation between batches
- [ ] Verify quality metrics per batch
- [ ] Verify memory accounting for batches

### 4. Bit-Width Validation

#### 4.1 Near-Lossless (K16/V16)
**Validation Steps:**
- [ ] Test with K16/V16 configuration
- [ ] Verify quality near baseline (> 0.99 cosine similarity)
- [ ] Verify memory savings
- [ ] CLI: `python benchmarks/kv_shootout.py --bit-width k16v16`

#### 4.2 Balanced (K8/V8)
**Validation Steps:**
- [ ] Test with K8/V8 configuration (default)
- [ ] Verify acceptable quality (> 0.95 cosine similarity)
- [ ] Verify memory savings
- [ ] CLI: `python benchmarks/kv_shootout.py --bit-width k8v8`

#### 4.3 Aggressive (K8/V5, K8/V6)
**Validation Steps:**
- [ ] Test with K8/V5 configuration
- [ ] Verify minimum acceptable quality (> 0.90 cosine similarity)
- [ ] Verify maximum memory savings
- [ ] CLI: `python benchmarks/kv_shootout.py --bit-width k8v5`

### 5. Platform Validation

#### 5.1 Apple Silicon (MLX)
**Validation Steps:**
- [ ] Test on M1/M2/M3 chips
- [ ] Verify Metal kernel compatibility
- [ ] Verify performance characteristics
- [ ] Environment: `requirements-apple-silicon.txt`

#### 5.2 Linux (CUDA)
**Validation Steps:**
- [ ] Port to CUDA if needed
- [ ] Test on NVIDIA GPUs
- [ ] Verify performance characteristics
- [ ] Verify numerical accuracy

#### 5.3 CPU Fallback
**Validation Steps:**
- [ ] Test CPU reference implementation
- [ ] Verify correctness
- [ ] Verify performance characteristics
- [ ] Test file: `rfsn_v10/kernels/metal/fused_packed_wrapper.py`

### 6. Quality Metrics Validation

#### 6.1 Cosine Similarity
**Thresholds:**
- K16/V16: > 0.99
- K8/V8: > 0.95
- K8/V5: > 0.90

**Validation Steps:**
- [ ] Compute cosine similarity between baseline and compressed
- [ ] Verify thresholds are met
- [ ] Log results for each bit-width

#### 6.2 KL Divergence
**Thresholds:**
- K16/V16: < 0.01
- K8/V8: < 0.05
- K8/V5: < 0.10

**Validation Steps:**
- [ ] Compute KL divergence between baseline and compressed
- [ ] Verify thresholds are met
- [ ] Log results for each bit-width

#### 6.3 Top-K Overlap
**Thresholds:**
- K16/V16: > 0.95
- K8/V8: > 0.90
- K8/V5: > 0.85

**Validation Steps:**
- [ ] Compute top-K overlap between baseline and compressed
- [ ] Verify thresholds are met
- [ ] Log results for each bit-width

### 7. Memory Validation

#### 7.1 Memory Savings
**Expected Savings:**
- K16/V16: ~50% vs dense
- K8/V8: ~75% vs dense
- K8/V5: ~85% vs dense

**Validation Steps:**
- [ ] Measure memory usage for dense baseline
- [ ] Measure memory usage for compressed
- [ ] Verify savings meet expectations
- [ ] Use RuntimeCounters.logical_payload_bytes

#### 7.2 Memory Accounting
**Validation Steps:**
- [ ] Verify logical_payload_bytes is accurate
- [ ] Verify staging_bytes_peak is accurate
- [ ] Verify dense_residual_bytes_peak is accurate
- [ ] Verify scratch_bytes_peak is accurate

### 8. Performance Validation

#### 8.1 Throughput
**Expected:**
- Packed attention should be faster than dense for long contexts
- Metal kernel should be faster than CPU reference

**Validation Steps:**
- [ ] Measure tokens/second for dense baseline
- [ ] Measure tokens/second for compressed
- [ ] Verify performance improvement
- [ ] Compare Metal vs CPU

#### 8.2 Latency
**Expected:**
- Packed attention should have lower latency for long contexts
- Block sealing overhead should be minimal

**Validation Steps:**
- [ ] Measure time-to-first-token
- [ ] Measure time-per-token
- [ ] Verify latency characteristics
- [ ] Profile block sealing overhead

### 9. Correctness Validation

#### 9.1 Invariant Checks
**Invariants:**
- `requantized_token_count == 0` for every generation
- Block positions are monotonic and non-overlapping
- No gaps in token positions

**Validation Steps:**
- [ ] Verify invariants hold for all test cases
- [ ] Add invariant checks to tests
- [ ] Log any violations

#### 9.2 Divergence Detection
**Validation Steps:**
- [ ] Use layer divergence tracing to detect issues
- [ ] Verify layer_divergence_count is 0 for correct runs
- [ ] Investigate any divergence detected
- [ ] Use RuntimeCounters.layer_divergence_count

### 10. Integration Validation

#### 10.1 Server Integration
**Validation Steps:**
- [ ] Test with FastAPI server
- [ ] Test with streaming endpoints
- [ ] Test with cancellation
- [ ] Verify no errors in production

#### 10.2 CLI Integration
**Validation Steps:**
- [ ] Test with kv_shootout benchmark
- [ ] Test with different bit-widths
- [ ] Test with strict mode
- [ ] Verify exit codes

## Validation Checklist

Before expanding to production, verify:

- [ ] All test suites pass (portable, MLX, apple_metal)
- [ ] Quality metrics meet thresholds for all bit-widths
- [ ] Memory savings meet expectations
- [ ] Performance is acceptable
- [ ] Invariants hold for all test cases
- [ ] No errors in server integration
- [ ] No errors in CLI integration
- [ ] Documentation is updated
- [ ] Release manifest is updated
- [ ] Version is synchronized

## Next Steps

1. **Immediate**: Run validation on Qwen2.5-0.5B with all bit-widths
2. **Short-term**: Add GQA model (Llama 2/3 or Mistral) to test suite
3. **Medium-term**: Add medium-sized model (7B) for scalability testing
4. **Long-term**: Add large model (14B+) for production validation
5. **Continuous**: Add new models as they become available

## References

- Test files:
  - `rfsn_v10/cache/tests/test_block_boundaries.py`
  - `rfsn_v10/cache/tests/test_gqa_validation.py`
  - `rfsn_v10/cache/tests/test_rope_validation.py`
  - `rfsn_v10/kernels/metal/tests/test_fused_packed.py`

- Configuration:
  - `release.toml` - Release manifest
  - `requirements-apple-silicon.txt` - Pinned dependencies

- Documentation:
  - `REPAIR_PROGRESS.md` - Phase completion status
  - `VALIDATION_PLAN.md` - This document
