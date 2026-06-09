# RFSN v10 Limitations

## Quantization

### >8-bit code buffers use raw uint32 fallback
- The bit-packer supports exact packing for 2–8 bit code buffers.
- Code widths above 8 store codes as raw `uint32` without word packing.
- **Policy**: Configs using >8-bit code buffers are quality experiments only. They are excluded from memory-optimized recommendations until wide-bit packing is implemented.

### QJL score correction is disabled
- QJL is implemented as a reference module but currently **fails** the shipped attention-score benchmark.
- `qjl_score_mae > base_score_mae`, `qjl_softmax_kl > base_softmax_kl`, and `qjl_topk_overlap < base_topk_overlap`.
- QJL is **not** integrated into the model attention path.
- Future QJL research (lower priority):
  - Structured Hadamard JL projection
  - Larger projection dimension
  - Layer-specific correction scale
  - Query-aware normalization
  - Keys-only correction
  - Score-space calibration

### Sparse decode
- Sparse decode is **disabled** by default.
- Block-selective retrieval exists but end-to-end sparse attention is not validated.

## Runtime

### Partial dequantization
- Selected-block reconstruction via `retrieve_blocks()` exists.
- True arbitrary token-level partial dequant remains unimplemented.

### Metal kernels
- Metal kernel path is an alpha route with strict fallback to sequential reconstruction when unsupported.
- Fused kernels are not yet implemented.

## Validation

### Model coverage
- Stable validation uses Qwen2.5-0.5B-Instruct.
- 1.5B validation is planned but not yet complete.
- No validation on >1.5B models.

### Throughput
- End-to-end speedup is not proven.
- Decode TPS is comparable to FP16, but total time is often slower due to compression overhead.

## General

- RFSN is **not production-ready**.
- Default config remains `k8_v5_gs64` on the stable path.
- Experimental quantizers exist for research but are not the default runtime.
