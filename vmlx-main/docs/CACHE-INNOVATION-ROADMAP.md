# Cache Innovation Roadmap — MLA, Hybrid SSM, Latent MoE

Internal engineering doc. Ideas for revolutionary cache approaches for
next-gen architectures: MLA (Multi-head Latent Attention), hybrid SSM
(Mamba+Attention), and latent MoE models on Apple Silicon.

## 1. MLA-Native Compressed KV Cache

**Problem:** MLA models (Mistral 4, DeepSeek V3) store compressed KV
latents (dim=256 + k_pe dim=64) instead of full K/V heads (dim=128 per
head × 32 heads = 4096). Standard KV cache quantization destroys these
compressed representations.

**Innovation: Latent-space quantization**
- Quantize in the LATENT space, not the head space
- kv_lora_rank=256 is already compressed — apply 4-bit quant on the 256-dim
  latent rather than the expanded 4096-dim heads
- Store (kv_latent_q4, k_pe_fp16) — 256×4bit + 64×16bit per token per layer
- Memory: ~160 bytes/token/layer vs ~1024 bytes for standard 32-head KV
- **6.4x memory reduction** with minimal quality loss since latent space
  is already optimized for reconstruction

**Implementation:**
- New `LatentKVCache` class: stores (kv_latent, k_pe) instead of (keys, values)
- `update_and_fetch` returns the raw compressed pair
- embed_q/unembed_out projections happen at attention time (already the case)
- Paged cache blocks store latent-sized pages (256+64=320 per token)

## 2. Asymmetric Paged Cache Blocks

**Problem:** Paged cache uses fixed block_size=64 tokens for all layers.
MLA layers need 320 bytes/token, dense attention needs 8192 bytes/token,
SSM layers need 0 bytes/token (state-based). Uniform blocks waste memory.

**Innovation: Per-architecture block sizing**
- Dense attention: block_size=32 (8KB blocks for 128-dim × 32 heads)
- MLA: block_size=128 (40KB blocks for 320-dim latent)
- SSM: block_size=∞ (single cumulative state, not token-positional)
- GQA: block_size=64 (reduced head count → smaller blocks)

**Implementation:**
- `PagedCacheManager` accepts per-layer block_size via architecture detector
- Block allocation pools: separate free-lists per block size class
- Cross-architecture prefix matching: hash latent blocks separately
- Disk serialization: different compression strategies per block type

## 3. SSM State Checkpointing for Hybrid Models

**Problem:** Hybrid models (Qwen3.5-A3B, Falcon-H1) have SSM layers whose
states are cumulative — they can't be token-sliced like KV cache. Currently
we store SSM state at prompt boundary only. Multi-turn conversations must
re-process all SSM layers from the checkpoint.

**Innovation: Sliding SSM checkpoints**
- Store SSM state snapshots at regular intervals (every N tokens)
- On cache hit: load nearest SSM checkpoint + re-process remaining tokens
- On cache miss: full SSM re-prefill from last checkpoint
- Checkpoint compaction: merge old checkpoints, keep only most recent M

**Implementation:**
- `HybridCheckpointCache`: stores (kv_blocks, ssm_checkpoints[])
- Each ssm_checkpoint: (token_position, layer_states[], timestamp)
- Checkpoint creation: piggyback on paged cache block boundaries
- Re-prefill: seek to nearest checkpoint, prefill remaining tokens only
- Memory budget: allocate 10% of cache budget to SSM checkpoints

## 4. Cross-Request Latent Sharing for MoE

**Problem:** MoE models route different tokens to different experts.
In batched decode, the same expert may be activated for multiple sequences.
Each sequence maintains separate KV cache, but the expert computations
could share cached intermediate states.

**Innovation: Expert activation caching**
- Track which experts are activated per token per layer
- Cache expert MLP outputs for frequently co-activated combinations
- Share across sequences in the same batch when routing decisions match
- Reduce redundant expert computation by 30-50% for common routing patterns

**Implementation:**
- `ExpertActivationCache`: LRU cache keyed by (layer, expert_idx, input_hash)
- Integration point: MoE forward pass checks cache before expert compute
- Invalidation: tied to model reload / weight change
- Memory: bounded by separate budget (not competing with KV cache)

## 5. Speculative Prefill with Latent Prediction

**Problem:** Long prompts take seconds to prefill. For multi-turn chats,
re-processing the entire prompt is wasteful even with prefix cache.

**Innovation: Latent continuation prediction**
- Train a lightweight predictor that estimates next-turn KV latents from
  the previous turn's final hidden states + new tokens
- Use predicted latents as "speculative KV cache" for first pass
- Validate against actual computation; rollback if prediction diverges
- For MLA models: predict kv_latent (256-dim) is much easier than
  predicting full K/V heads (4096-dim)

**Status:** Research-stage. Requires training the predictor on representative
conversations. May only be practical for models we quantize ourselves.

## 6. Unified Memory Topology Awareness

**Problem:** Apple Silicon's unified memory means GPU VRAM and system RAM
are the same physical memory. Current cache management doesn't exploit this.

**Innovation: Metal-aware cache lifecycle**
- Use `mx.metal.device_info()` to track actual GPU allocation vs available
- Implement cache pressure callbacks: when Metal approaches allocation limit,
  proactively evict least-recently-used blocks to disk BEFORE OOM
- Hot/warm/cold tiers: GPU-resident → CPU-resident → disk (SSD NVMe)
- Zero-copy promotion: Metal's unified memory means "CPU-resident" arrays
  can be used by GPU without copy — just needs address mapping

**Implementation:**
- `UnifiedMemoryManager`: wraps PagedCacheManager with memory pressure awareness
- Tier 0 (hot): actively used by current batch — GPU-pinned
- Tier 1 (warm): recently used, in unified memory — instant GPU access
- Tier 2 (cold): on NVMe SSD — microsecond load on M-series
- Tier transitions: background thread monitors pressure, promotes/demotes

## Priority Order

1. **MLA-Native Compressed KV Cache** — biggest impact, Mistral 4 is live now
2. **Asymmetric Paged Cache Blocks** — applies to all architectures
3. **SSM State Checkpointing** — needed for hybrid model multi-turn
4. **Unified Memory Topology** — performance optimization
5. **Expert Activation Caching** — batched decode optimization
6. **Speculative Prefill** — research stage
