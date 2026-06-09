# vMLX Benchmark Results — 2026-03-25

## Hardware
- MacBook Pro M4 Max, 128GB Unified Memory
- macOS 15.4, MLX 0.25.x
- Paged cache OFF (clean per-request memory — no cached blocks inflating numbers)
- TurboQuant auto-enabled on all models

---

## Model 1: Mistral-Small-4-119B-JANG_2L

| Spec | Value |
|------|-------|
| Architecture | MoE (128 experts, 4 active) + MLA + VLM |
| Parameters | 119B total, ~24B active |
| Layers | 36 |
| Attention | MLA (kv_lora_rank=256 — compressed KV latents) |
| Context Window | 1M native |
| Quantization | JANG 2.14-bit |
| Model RAM | 37.4GB |
| Gen Speed | **61 tok/s** |

### Context Length Scaling

| Context | TTFT | Peak GPU | KV Overhead |
|---------|------|----------|-------------|
| 1K | 0.8s | 37.5GB | +0.0GB |
| 4K | 1.8s | 37.5GB | +0.0GB |
| 16K | 6.4s | 37.6GB | +0.1GB |
| 32K | 14.4s | 37.8GB | +0.2GB |
| 64K | 38.3s | 38.2GB | +0.4GB |
| 100K | 71.8s | 38.8GB | +0.6GB |

### Key Takeaways
- **MLA makes context essentially free** — 100K adds only 0.6GB
- 119B VLM in **37.4GB** — runs on 48GB Macs
- 61 tok/s generation — fast for 119B
- Could handle **1M context** before hitting memory limits (~6GB KV at 1M)
- MoE + MLA + VLM + reasoning — full feature set
- Best architecture for memory-constrained devices

---

## Model 2: Qwen3.5-VL-35B-A3B-JANG_4K-CRACK

| Spec | Value |
|------|-------|
| Architecture | Hybrid (Gated DeltaNet + Gated Attention) + MoE + Vision |
| Parameters | 35B total, 3B active (MoE) |
| Quantization | JANG 3.98-bit |
| Model RAM | 19GB |
| Load time | 1.4s (JANG instant mmap) |
| TurboQuant | 3-bit KV compression during generation |
| Gen Speed | **71 tok/s** |

### Context Length Scaling (paged cache ON — from earlier test)

| Context | TTFT | Gen Speed | Peak GPU* |
|---------|------|-----------|----------|
| 1K | 0.8s | 71 tok/s | 18.2GB |
| 4K | 2.3s | 71 tok/s | 18.8GB |
| 16K | 9.4s | 71 tok/s | 20.9GB |
| 32K | 19.4s | 71 tok/s | 25.1GB |
| 64K | 43.3s | 71 tok/s | 33.4GB |

*Peak GPU measured with paged cache ON (retains float blocks for multi-turn reuse).
Without paged cache, memory returns to ~19GB after each request.
During active generation, TurboQuant compresses KV ~5x.

### Key Takeaways
- Generation speed constant at **71 tok/s** regardless of context
- 19GB base — fits on **32GB Macs** with room for short context
- Hybrid SSM: Mamba layers for efficiency + attention for quality
- Cache hits give 2x faster TTFT on follow-up messages
- Cold TTFT 0.48s, Warm TTFT 0.23s

---

## Model 3: MiniMax-M2.5-JANG_2L

| Spec | Value |
|------|-------|
| Architecture | Lightning Attention + MoE (256 experts, 8 active) |
| Parameters | 230B total, 10B active |
| Layers | 62 |
| Context Window | 196K native |
| Quantization | JANG 2.1-bit |
| Model RAM | 62.6GB |
| Gen Speed | (pending retest — server crashed during 100K) |

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
- 230B MoE in **62.6GB** — fits on 128GB Macs
- 100K context in 77.5GB total — leaves 50GB free
- Lightning Attention efficient but not as extreme as MLA
- 196K native context window
- Gen speed needs retest (thinking mode interfered with measurement)

---

## Summary

| Model | Params | JANG | RAM | Speed | 100K KV |
|-------|--------|------|-----|-------|---------|
| Mistral 4 119B | 119B (24B active) | 2.14-bit | 37GB | 61 tok/s | **+0.6GB** |
| Qwen3.5-VL-35B | 35B (3B active) | 3.98-bit | 19GB | 71 tok/s | ~8GB* |
| MiniMax M2.5 230B | 230B (10B active) | 2.1-bit | 63GB | TBD | +15GB |

*Qwen3.5 during active generation with TQ: ~5x less. Paged cache stores float.

## Notes
- All TTFT measured with `enable_thinking=false` / `reasoning_effort=none`
- Gen speed from streaming chunk count over wall clock time
- TurboQuant auto-enabled on all JANG models (3-bit KV compression during generation)
- "After clear" memory always returns to model baseline via soft-sleep
