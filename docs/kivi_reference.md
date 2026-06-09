# KIVI Reference Notes

KIVI (arxiv: 2402.02750) is a 4-bit KV cache system for NVIDIA/vLLM.

## Key ideas (for reference only)

- Keys quantized per-channel (shared scale across the token dimension)
- Values quantized per-token (shared scale across the channel dimension)
- 4-bit quantization with small FP16 residual cache
- Fused paged attention kernel for CUDA

## Why it is NOT the Apple Silicon path

- KIVI is implemented in CUDA kernels inside vLLM
- MLX has a different memory model (unified CPU/GPU, lazy evaluation)
- Metal kernels cannot reuse CUDA kernel code directly
- MLX's `mx.quantize` / `mx.quantized_matmul` cover the core quantization
  operation in a more MLX-native way

## What is worth borrowing conceptually

1. **Asymmetric K/V quantization**: keys per-channel, values per-token — both
   `rfsn_v10` and `rfsn_v11` already follow this direction
2. **Residual cache**: keep a small FP16 buffer for recent tokens — could
   reduce quality loss at short contexts
3. **Group size selection**: gs=32 vs gs=64 trade-offs are directly applicable

## Action

No code merge. Use as conceptual reference when tuning group sizes and
asymmetric K/V schemes in `rfsn_v11/quant/`.
