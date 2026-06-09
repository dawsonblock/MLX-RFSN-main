# vMLX Benchmark Results — 2026-03-25

## Hardware
- MacBook Pro M4 Max, 128GB Unified Memory
- macOS 15.4, MLX 0.25.x

---

## Model 1: Qwen3.5-VL-35B-A3B-JANG_4K-CRACK

| Spec | Value |
|------|-------|
| Architecture | Hybrid (Gated DeltaNet + Gated Attention) + MoE + Vision |
| Parameters | 35B total, 3B active (MoE) |
| Quantization | JANG 3.98-bit |
| Model RAM | 19GB |
| Load time | 1.4s (JANG instant mmap) |
| TurboQuant | Auto-enabled (3-bit KV compression) |
| Cache | Paged blocks + SSM companion |
| API | OpenAI + Anthropic compatible |

### Context Length Scaling

| Context | TTFT | Gen Speed | Peak GPU | After Clear |
|---------|------|-----------|----------|-------------|
| 1K | 0.8s | 71 tok/s | 18.2GB | 18.0GB |
| 4K | 2.3s | 71 tok/s | 18.8GB | 18.0GB |
| 16K | 9.4s | 71 tok/s | 20.9GB | 18.0GB |
| 32K | 19.4s | 71 tok/s | 25.1GB | 18.0GB |
| 64K | 43.3s | 71 tok/s | 33.4GB | 18.0GB |
| 100K | ~68s | 71 tok/s | ~47GB* | 18.0GB |

*100K extrapolated. Peak GPU = cumulative (paged cache retains blocks in float for reuse).
During active generation TurboQuant compresses KV to ~3-bit (~5x), so live prefill uses far less.
After clear = soft-sleep releases all cached blocks back to baseline.

### Key Takeaways
- Generation speed stays **constant at 71 tok/s** regardless of context length
- Model weights only 19GB — fits on 32GB Macs with room for 4K-8K context
- 64GB Macs can handle 32K+ context comfortably
- Memory returns to 18GB baseline after conversation ends (soft-sleep or mx.clear_cache)
- Multi-turn cache hits: 2x faster TTFT on follow-up messages
- Cold TTFT: 0.48s (short prompts), Warm TTFT: 0.23s (cache hit)
- TurboQuant active during generation: ~5x KV compression (live prefill ~5.4GB for 100K vs 29GB float)

---

## Model 2: MiniMax-M2.5-JANG_2L

| Spec | Value |
|------|-------|
| Architecture | Lightning Attention + MoE (256 experts, 8 active per token) |
| Parameters | 230B total, 10B active |
| Layers | 62 |
| Context Window | 196K native |
| Quantization | JANG 2.1-bit |
| Model RAM | 62.6GB |
| TurboQuant | Auto-enabled (3-bit KV compression) |
| Reasoning | Built-in thinking (`<think>` tags, qwen3 parser) |
| Gen Speed | 5.6 tok/s |

### Context Length Scaling

| Context | TTFT | Peak GPU | KV Overhead |
|---------|------|----------|-------------|
| 1K | 3.2s | 62.7GB | +0.1GB |
| 4K | 3.0s | 63.0GB | +0.4GB |
| 16K | 2.0s | 64.1GB | +1.5GB |
| 32K | 2.3s | 66.3GB | +3.7GB |
| 64K | 5.4s | 69.6GB | +7.0GB |
| 100K | ~10s | 77.5GB | +14.9GB |

### Key Takeaways
- **230B MoE running on a MacBook** — 62.6GB at JANG 2-bit
- 100K context fits in **77.5GB total** — leaves 50GB free on 128GB machine
- KV overhead extremely efficient: ~0.15GB per 1K tokens (only 8 KV heads)
- 196K native context — could handle **200K+** before hitting 128GB
- 64GB Macs can run this with short context (1K-4K)
- 5.6 tok/s — usable for a model this size, MoE routing keeps active compute manageable

---

## Model 3: Mistral-Small-4-119B-JANG_2L

| Spec | Value |
|------|-------|
| Architecture | MoE (128 experts, 4 active) + MLA + VLM |
| Parameters | 119B total, ~24B active |
| Layers | 36 |
| Attention | MLA (kv_lora_rank=256, compressed KV latents) |
| Context Window | 1M native |
| Quantization | JANG 2.14-bit |
| Model RAM | 37.4GB |
| TurboQuant | Auto-enabled (3-bit KV compression) |
| Vision | Yes (image + text) |
| Reasoning | reasoning_effort ("none" / "high") |
| Gen Speed | **53.6 tok/s** |

### Context Length Scaling

| Context | TTFT | Peak GPU | KV Overhead |
|---------|------|----------|-------------|
| 1K | 0.9s | 37.4GB | +0.0GB |
| 4K | 1.6s | 37.5GB | +0.1GB |
| 16K | 6.4s | 37.5GB | +0.1GB |
| 32K | 14.7s | 37.5GB | +0.1GB |
| 64K | 41.4s | 37.5GB | +0.1GB |
| 100K | 87.3s | 37.5GB | +0.1GB |

### Key Takeaways
- **MLA makes KV cache essentially free** — 100K context adds <0.1GB overhead
- 119B VLM fits in **37.4GB** at JANG 2-bit — runs on 48GB and 64GB Macs
- 53.6 tok/s generation speed — fast for a 119B model
- MoE + MLA + VLM + reasoning — most feature-rich architecture, all working
- 1M native context window — limited only by prefill time, not memory
- **Best model for memory-constrained Macs** — huge context, tiny KV footprint

---

## Notes
- All benchmarks with `enable_thinking=false` (no reasoning overhead in TTFT)
- Paged cache ON (block_size=64, max_blocks=1000)
- Continuous batching ON (max_num_seqs=256)
- Peak GPU includes paged cache blocks from completed request (LRU cached for reuse)
- "After Clear" = memory after soft-sleep reclaims all cached blocks
- TurboQuant compresses KV cache ~5x during active generation (reduces peak GPU during prefill)
