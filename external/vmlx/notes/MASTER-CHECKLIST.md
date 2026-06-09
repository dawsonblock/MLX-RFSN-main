# vMLX Master Audit Checklist

Last updated: 2026-03-23
Sources consolidated: AUDIT-CHECKLIST.md, CROSS-CHECK-MATRIX.md, todo-next-session.md, session-2026-03-23-issues.md, final-audit-2026-03-16.md, audit-findings-2026-03-16.md, 2026-03-22-audit-findings.md, session-j-status.md

## Status Legend

- [ ] Not checked
- [x] Verified PASS (audit name, date)
- [~] Known issue / WARN — documented, low impact
- [!] FAIL — needs fix (details)

---

## 1. CACHING SUBSYSTEM

Files: `paged_cache.py`, `prefix_cache.py`, `block_disk_store.py`, `disk_cache.py`, `memory_cache.py`, `vision_embedding_cache.py`, `utils/cache_types.py`, `utils/mamba_cache.py`

### 1.1 Paged Cache (`paged_cache.py`)

- [ ] 1.1.1 Block allocation returns valid block IDs for new sequences
- [ ] 1.1.2 Block deallocation frees blocks back to pool
- [ ] 1.1.3 Hash computation is deterministic for same token sequences
- [ ] 1.1.4 COW fork creates new physical block on write, shares on read
- [ ] 1.1.5 Eviction selects LRU blocks, not actively referenced ones
- [ ] 1.1.6 Block table grows correctly when sequence extends past allocated blocks
- [ ] 1.1.7 No off-by-one in block boundary calculations (token_idx / block_size)
- [ ] 1.1.8 Reference counting: increment on fork, decrement on free
- [ ] 1.1.9 Thread safety: concurrent block alloc/free from scheduler + eviction
- [ ] 1.1.10 Hash collision handling: verify blocks actually match after hash hit
- [ ] 1.1.11 Memory leak check: blocks allocated but never freed on request abort
- [~] 1.1.12 CacheList tag not handled in `_serialize_block` — MoE models skip disk L2 (WARN #3)
- [ ] 1.1.13 Pool exhaustion: graceful error, not crash
- [ ] 1.1.14 Single-token sequence (less than one block)
- [ ] 1.1.15 COW fork then original freed: forked block survives
- [x] 1.1.16 CacheList paged cache support (session 2026-03-21c, B16 in cross-check)

### 1.2 Prefix Cache (`prefix_cache.py`)

- [ ] 1.2.1 Store: prefix tokens keyed and stored correctly
- [ ] 1.2.2 Fetch: exact prefix match returns cached KV states
- [ ] 1.2.3 Fetch: partial prefix match returns longest matching prefix
- [ ] 1.2.4 Reconstruct: cached KV states correctly sliced and applied
- [!] 1.2.5 gen_prompt_len stripping: thinking model generation prompt tokens excluded from cache key — ROOT CAUSE FOUND, scheduler side ready, server.py computation needed
- [ ] 1.2.6 Cache key includes all relevant state (tokens, model, quantization)
- [ ] 1.2.7 Cache invalidation on model change
- [ ] 1.2.8 gen_prompt_len computation covers ALL thinking models: Qwen3, DeepSeek R1, Mistral 4, MiniMax, Kimi K2, GLM-Z1
- [ ] 1.2.9 Multi-turn conversation: second turn hits prefix cache for first turn
- [ ] 1.2.10 Multi-turn with reasoning ON then OFF: cache key diverges correctly
- [ ] 1.2.11 Multi-turn with reasoning OFF then ON: no stale reasoning cache
- [ ] 1.2.12 Two concurrent requests with overlapping prefixes
- [ ] 1.2.13 Prefix cache after model sleep/wake cycle
- [ ] 1.2.14 Prefix cache with KV quantization enabled

### 1.3 Block Disk Store (`block_disk_store.py`)

- [ ] 1.3.1 Serialize: KV blocks written to disk in correct format
- [ ] 1.3.2 Deserialize: KV blocks read from disk match originals
- [ ] 1.3.3 L2 tier: blocks evicted GPU -> disk, restored disk -> GPU
- [x] 1.3.4 Metal safety: no lazy MLX ops in serialize path — numpy round-trip (PR #19, session 2026-03-23)
- [x] 1.3.5 orig_dtype metadata preserved on roundtrip (PR #18, session 2026-03-23)
- [~] 1.3.6 CacheList tag handling gap — MoE models skip disk L2 (same as 1.1.12)
- [ ] 1.3.7 Disk full condition: graceful error
- [ ] 1.3.8 Concurrent serialize + deserialize from scheduler threads
- [x] 1.3.9 L2 disk store cleared on DELETE /v1/cache (H+ audit fix M8, 2026-03-22)

### 1.4 Prompt Disk Cache (`disk_cache.py`)

- [ ] 1.4.1 Store: full prompt cache saved to disk with correct key
- [ ] 1.4.2 Load: prompt cache restored from disk matches original
- [x] 1.4.3 Metal safety: no stale GPU pointers after disk load — numpy round-trip applied (PR #19)
- [ ] 1.4.4 Cache key includes model path + quantization config
- [ ] 1.4.5 Eviction policy: oldest or LRU when disk budget exceeded
- [ ] 1.4.6 Store/load across model sleep/wake cycle
- [ ] 1.4.7 Corrupted cache file on disk: error handling

### 1.5 KV Cache Quantization

- [ ] 1.5.1 q4 quantization: KV states quantized and dequantized correctly
- [ ] 1.5.2 q8 quantization: same as above with 8-bit
- [ ] 1.5.3 MLA guard: KV quantization DISABLED for MLA models (Mistral 4, DeepSeek V3)
- [x] 1.5.4 QuantizedKVCache list vs tuple fix for paged cache (session 2026-03-21c)
- [~] 1.5.5 q4 KV quant degrades long reasoning context on restore (WARN #5)
- [ ] 1.5.6 KV quant with hybrid SSM model: only attention layers quantized
- [ ] 1.5.7 KV quant after wake from deep sleep

### 1.6 Memory Cache (`memory_cache.py`)

- [ ] 1.6.1 Memory-aware cache respects configured limits
- [ ] 1.6.2 Eviction triggers at correct threshold
- [ ] 1.6.3 Does not evict currently-in-use blocks

### 1.7 Vision Embedding Cache (`vision_embedding_cache.py`)

- [ ] 1.7.1 Reuses embeddings across multi-turn with same image
- [ ] 1.7.2 Clears on model switch
- [ ] 1.7.3 Memory accounting

### 1.8 Mamba Cache (`utils/mamba_cache.py`)

- [x] 1.8.1 BatchMambaCache.extend() correct (session 2026-03-21)
- [x] 1.8.2 MambaCache merge in continuous batching (session 2026-03-21)
- [x] 1.8.3 QuantizedKVCache list/tuple reconstruction for 40+8 layers (session 2026-03-21c)
- [~] 1.8.4 `_ensure_batch_cache` checks ArraysCache not MambaCache directly (WARN #3)

### 1.9 Cache + Model Type Interactions

- [ ] 1.9.1 Standard KV (Llama, Gemma): paged + prefix + disk all work
- [ ] 1.9.2 MLA (Mistral 4, DeepSeek V3): paged cache with head inflation, no KV quant
- [x] 1.9.3 Hybrid SSM (Nemotron-H): MambaCache + attention layers coexist (session 2026-03-21)
- [x] 1.9.4 MoE CacheList wrapping per-expert caches (session 2026-03-21)
- [ ] 1.9.5 Each model type: start session, chat 3 turns, check cache stats
- [ ] 1.9.6 Model switch (standard -> MLA): cache completely cleared

### 1.10 Cache After Sleep/Wake

- [ ] 1.10.1 Soft sleep: cache cleared, Metal memory limit reduced
- [ ] 1.10.2 Wake from soft sleep: cache limit restored (is not None)
- [ ] 1.10.3 Deep sleep: model unloaded, all cache freed
- [ ] 1.10.4 Stale cache references: no dangling pointers after sleep
- [ ] 1.10.5 Chat -> sleep -> wake -> chat: no crash, correct generation
- [ ] 1.10.6 Concurrent request arrives during sleep transition

---

## 2. MODEL LOADING

Files: `utils/jang_loader.py`, `models/mllm.py`, `models/llm.py`, `model_config_registry.py`, `model_configs.py`, `model_registry.py`, `api/utils.py`

### 2.1 JANG v2 Text Loader (`utils/jang_loader.py`)

- [ ] 2.1.1 Weight loading: safetensors read, keys mapped to MLX module names
- [ ] 2.1.2 Quantization: QuantizedLinear repacking stays quantized in GPU memory
- [x] 2.1.3 Gate dequant: MoE gate weights dequantized to bfloat16 (confirmed multiple sessions)
- [x] 2.1.4 bfloat16 cast: MLA models + n_experts >= 128 get bfloat16 (session 2026-03-23 fix #4)
- [x] 2.1.5 `_fix_quantized_bits`: handles QuantizedMultiLinear (session 2026-03-23 fix #2)
- [x] 2.1.6 Nemotron Cascade: fc1/fc2 rename + gate dequant (confirmed 42GB/46tok/s)
- [ ] 2.1.7 Weight key mismatch: error message identifies which key is missing
- [ ] 2.1.8 No silent weight drops: all safetensor keys accounted for or explicitly skipped

### 2.2 JANG v2 VLM Loader (`models/mllm.py`)

- [x] 2.2.1 kv_b_proj split: split into correct dims for MLA (session 2026-03-23 fix #3)
- [x] 2.2.2 embed_q / unembed_out: loaded as QuantizedMultiLinear (session 2026-03-23)
- [ ] 2.2.3 Vision weights: loaded separately, not quantized
- [x] 2.2.4 Gate dequant for VLM MoE layers (session 2026-03-23 fix #5)
- [x] 2.2.5 `jang_config.json` `has_vision: true` detection + fallthrough to config.json (session 2026-03-23)
- [ ] 2.2.6 Vision encoder precision: fp16/fp32, NOT quantized
- [~] 2.2.7 JANG VL MoE not working (sanitizer conflict) — known limitation
- [!] 2.2.8 Mistral 4 RoPE error: `rope_parameters` missing in JANG v2 VLM loader config — CURRENT BLOCKER

### 2.3 Standard MLX Loader

- [ ] 2.3.1 `mlx_lm.load()` for text models: correct tokenizer + model
- [ ] 2.3.2 `mlx_vlm.load()` for VLM: correct vision encoder + text model
- [ ] 2.3.3 Model loads from HuggingFace cache path
- [ ] 2.3.4 Model loads from local directory

### 2.4 Model Config Registry (`model_config_registry.py`, `model_configs.py`)

- [x] 2.4.1 text_config.model_type disambiguation for VLM wrappers (session 2026-03-23 fix #11)
- [x] 2.4.2 Priority: text_config > config.json model_type > name regex fallback (session 2026-03-23)
- [ ] 2.4.3 All supported model families have entries in registry
- [ ] 2.4.4 No stale entries for removed/renamed model types
- [ ] 2.4.5 Unknown model_type: graceful fallback to generic config

### 2.5 is_mllm_model Detection (`api/utils.py`)

- [x] 2.5.1 JANG path: checks jang_config.has_vision THEN falls through to config.json (session 2026-03-23 fix #13)
- [x] 2.5.2 No early return on jang_config (defense in depth, session 2026-03-23)
- [ ] 2.5.3 Text-only model: returns False
- [ ] 2.5.4 VLM model: returns True

### 2.6 Bundled Python

- [ ] 2.6.1 Electron app launches with `-s` flag (bundled, not system site-packages)
- [x] 2.6.2 Bundled mlx_lm/mlx_vlm includes all patches (_Mistral4VLMBackbone, etc.) — verified session 2026-03-23
- [ ] 2.6.3 `vmlx_engine/` in bundled matches source (CRITICAL sync step before build)
- [ ] 2.6.4 No stale .pyc files in bundled that override .py changes

### 2.7 Model Inspector (`utils/model_inspector.py`)

- [ ] 2.7.1 Correctly inspects model architecture from safetensors
- [ ] 2.7.2 Reports quantization info

### 2.8 Weight Index (`utils/weight_index.py`)

- [ ] 2.8.1 Correctly maps layer -> file offset for SSD streaming

---

## 3. INFERENCE

Files: `scheduler.py`, `mllm_scheduler.py`, `engine/batched.py`, `engine/simple.py`, `engine/base.py`, `simple.py`, `utils/ssd_generate.py`, `speculative.py`, `model_runner.py`, `output_collector.py`, `worker.py`, `attention.py`, `optimizations.py`

### 3.1 LLM Scheduler (`scheduler.py`)

- [ ] 3.1.1 add_request: request queued, assigned ID, state = WAITING
- [ ] 3.1.2 Prefill: prompt tokens processed, KV cache populated
- [ ] 3.1.3 Decode: autoregressive generation produces tokens
- [ ] 3.1.4 Cleanup: finished/aborted requests have cache freed
- [ ] 3.1.5 GQA head normalization in `_detect_n_kv_heads()`
- [ ] 3.1.6 Request state machine: WAITING -> RUNNING -> FINISHED/ABORTED (no invalid transitions)
- [ ] 3.1.7 Abort during prefill: stops cleanly, frees partial cache
- [x] 3.1.8 MLA guard in scheduler (head inflation for MLA models, session 2026-03-23)
- [x] 3.1.9 Stale `_n_kv_heads` on model switch: fixed with clear() reset (WARN #8)
- [ ] 3.1.10 Single request: start to finish
- [ ] 3.1.11 Concurrent requests: both get correct independent outputs
- [ ] 3.1.12 Abort mid-generation: partial response returned, no resource leak
- [ ] 3.1.13 Max tokens reached: finish_reason = "length"
- [ ] 3.1.14 Stop token hit: finish_reason = "stop"

### 3.2 MLLM Scheduler (`mllm_scheduler.py`)

- [ ] 3.2.1 Vision encoding: images processed through vision encoder
- [ ] 3.2.2 Batch generator: vision tokens + text tokens merged correctly
- [!] 3.2.3 `_extract_cache_states` missing GQA head normalization — BUG (todo #2)
- [~] 3.2.4 MLA MLLM batch with head inflation (single-request OK, WARN #10)
- [ ] 3.2.5 Vision embedding cache: reuse across multi-turn with same image
- [ ] 3.2.6 Single image input: correct visual understanding
- [ ] 3.2.7 Multiple images in one request
- [ ] 3.2.8 Text-only request to VLM model (no image)
- [ ] 3.2.9 Abort during vision encoding phase

### 3.3 MLLM Batch Generator (`mllm_batch_generator.py`)

- [ ] 3.3.1 Vision tokens merged with text tokens in correct positions
- [ ] 3.3.2 Batch extends correctly for VLM requests
- [ ] 3.3.3 Skips Metal override in SSD streaming mode

### 3.4 Continuous Batching (`engine/batched.py`)

- [ ] 3.4.1 Batch merge: new requests added to running batch
- [ ] 3.4.2 Batch filter: finished requests removed from batch
- [ ] 3.4.3 Individual request outputs separated correctly
- [ ] 3.4.4 Batch size limits: not exceeding GPU memory per batch
- [ ] 3.4.5 gen_prompt_len computation for batched requests
- [ ] 3.4.6 2 simultaneous requests, different lengths, different stop tokens
- [ ] 3.4.7 Add request while batch is mid-decode
- [ ] 3.4.8 Continuous batching after sleep/wake

### 3.5 SimpleEngine (`engine/simple.py`, `simple.py`)

- [ ] 3.5.1 `chat()`: single-turn chat completion
- [ ] 3.5.2 `stream_chat()`: streaming token output
- [ ] 3.5.3 `generate()`: raw text generation
- [~] 3.5.4 Prefill not interruptible (stop button unresponsive during long prefills) — known limitation
- [ ] 3.5.5 Very long prompt (near context window limit)
- [ ] 3.5.6 Empty prompt: handled gracefully

### 3.6 EngineCore (`engine_core.py`)

- [ ] 3.6.1 Engine lifecycle: init, start, stop
- [ ] 3.6.2 Proper cleanup on shutdown

### 3.7 Stop Tokens

- [ ] 3.7.1 Model-specific stop tokens applied (from model config registry)
- [ ] 3.7.2 User-provided stop tokens (via API `stop` field) applied
- [ ] 3.7.3 Multiple stop tokens: first match wins
- [ ] 3.7.4 Custom stop sequence (e.g., "```") mid-word
- [ ] 3.7.5 Stop token at exactly max_tokens boundary

### 3.8 Sampling

- [ ] 3.8.1 temperature=0: deterministic (argmax)
- [ ] 3.8.2 temperature>0: random sampling
- [ ] 3.8.3 top_p: nucleus sampling
- [ ] 3.8.4 top_k: top-k sampling
- [ ] 3.8.5 min_p: minimum probability filtering
- [ ] 3.8.6 repetition_penalty: reduces probability of repeated tokens
- [ ] 3.8.7 All params combined: temperature + top_p + top_k + repetition_penalty

### 3.9 SSD Streaming (`utils/ssd_generate.py`, `utils/streaming_wrapper.py`)

- [ ] 3.9.1 Weight recycling: layers streamed from SSD, not all held in memory
- [ ] 3.9.2 Generation produces correct output despite layer cycling
- [ ] 3.9.3 Memory usage stays bounded
- [ ] 3.9.4 Disk I/O errors during streaming: graceful handling
- [ ] 3.9.5 Model larger than GPU memory: SSD streaming enables generation
- [ ] 3.9.6 Abort during SSD streaming decode
- [x] 3.9.7 All cache flags forced off in SSD mode (session 2026-03-22, C1)
- [x] 3.9.8 max_num_seqs=1 in SSD mode (session 2026-03-22, C2)
- [x] 3.9.9 speculative_model=None in SSD mode (session 2026-03-22, C3)
- [x] 3.9.10 Deep sleep preserves ALL stream settings (session 2026-03-22, C20-24)
- [x] 3.9.11 Normal mode COMPLETELY unaffected by SSD code paths (session 2026-03-22, C32)
- [x] 3.9.12 `--stream-from-disk` added to buildArgs + RESTART_REQUIRED_KEYS (session 2026-03-23)

### 3.10 Speculative Decoding (`speculative.py`)

- [ ] 3.10.1 Draft model generates candidate tokens
- [ ] 3.10.2 Target model verifies candidates in single forward pass
- [ ] 3.10.3 All candidates accepted: max speedup
- [ ] 3.10.4 Draft model OOM: fallback to normal decode
- [x] 3.10.5 Draft model unloaded on deep sleep (H+ audit fix M7, 2026-03-22)
- [x] 3.10.6 Speculative config saved to `_cli_args` for wake reconstruction (H+ audit fix M7)

### 3.11 Attention (`attention.py`)

- [ ] 3.11.1 Standard multi-head attention
- [ ] 3.11.2 Grouped-query attention (GQA)
- [ ] 3.11.3 Multi-latent attention (MLA)
- [ ] 3.11.4 Sliding window attention

### 3.12 Model Runner (`model_runner.py`)

- [ ] 3.12.1 Forward pass orchestration
- [ ] 3.12.2 Cache management integration

### 3.13 Output Collector (`output_collector.py`)

- [ ] 3.13.1 Token collection and formatting
- [ ] 3.13.2 Usage statistics computation

### 3.14 Worker (`worker.py`)

- [ ] 3.14.1 Background task execution
- [ ] 3.14.2 Thread safety

### 3.15 Optimizations (`optimizations.py`)

- [ ] 3.15.1 Metal kernel optimizations
- [ ] 3.15.2 Memory optimization strategies

### 3.16 MLX Platform (`mlx_platform.py`)

- [ ] 3.16.1 Metal device detection
- [ ] 3.16.2 Memory limit queries
- [ ] 3.16.3 connectHost() 0.0.0.0 -> 127.0.0.1 conversion

### 3.17 Request Model (`request.py`)

- [ ] 3.17.1 All request fields validated
- [ ] 3.17.2 Default values correct

---

## 4. REASONING

Files: `reasoning/base.py`, `reasoning/qwen3_parser.py`, `reasoning/deepseek_r1_parser.py`, `reasoning/mistral_parser.py`, `reasoning/gptoss_parser.py`, `reasoning/think_parser.py`

### 4.1 Reasoning Parsers (5 parsers)

- [ ] 4.1.1 Qwen3: `<think>...</think>` extraction
- [ ] 4.1.2 DeepSeek R1: `<think>...</think>` extraction
- [x] 4.1.3 Mistral: `[THINK]...[/THINK]` extraction — parser added (session 2026-03-23)
- [ ] 4.1.4 GPT-OSS: reasoning block extraction
- [ ] 4.1.5 Think: generic `<think>` tag parsing
- [ ] 4.1.6 Base: abstract interface, factory method
- [~] 4.1.7 Partial tag handling across chunk boundaries — rare char leak (WARN #6)
- [~] 4.1.8 GPT-OSS emitted_reasoning shrink edge case (WARN #7)

### 4.2 Streaming Extraction

- [ ] 4.2.1 Partial tags buffered, not emitted prematurely
- [ ] 4.2.2 Complete tags extracted and routed to reasoning_content
- [ ] 4.2.3 Concurrent requests: each gets own parser state
- [~] 4.2.4 Reasoning ON but no think tags: fallback re-emit delay (WARN #4)
- [ ] 4.2.5 Chunk boundary splits `<thi` | `nk>`: correctly assembled
- [ ] 4.2.6 Single-character chunks: each char of `<think>` arrives separately

### 4.3 enable_thinking / reasoning_effort

- [ ] 4.3.1 enable_thinking=true: thinking tags included in generation
- [ ] 4.3.2 enable_thinking=false: thinking suppressed
- [x] 4.3.3 Mistral 4: enable_thinking auto-maps to reasoning_effort (session 2026-03-23)
- [ ] 4.3.4 reasoning_effort="none": no thinking (Mistral 4 only supports "none"/"high")
- [ ] 4.3.5 reasoning_effort="high": full thinking (Mistral 4)
- [ ] 4.3.6 think_in_template detection: "always-thinks" vs "completes-thinking"
- [ ] 4.3.7 suppress_reasoning: reasoning consumed but hidden from response
- [ ] 4.3.8 Toggle reasoning ON -> OFF -> ON across multi-turn: each turn correct
- [x] 4.3.9 `[THINK]` -> `<think>` normalization in chat.ts (session 2026-03-23, fix #1 const->let)
- [x] 4.3.10 `[/THINK]` implicit strip in tool parser fixed (session 2026-03-23)
- [x] 4.3.11 Harmony reset_state() after parser clone in non-streaming paths (H+ audit fix L13)

---

## 5. TOOL CALLING

Files: `tool_parsers/` (14 parsers), `api/tool_calling.py`, `mcp/` (6 files), `panel/src/main/tools/` (2 files)

### 5.1 Tool Parsers (14 parsers)

- [ ] 5.1.1 Mistral: `[TOOL_CALLS]` marker + JSON
- [ ] 5.1.2 Llama: `<|python_tag|>` or JSON block
- [ ] 5.1.3 Hermes: `<tool_call>...</tool_call>` XML
- [ ] 5.1.4 DeepSeek: function call JSON
- [ ] 5.1.5 Qwen: function call extraction
- [ ] 5.1.6 Functionary: v3 format
- [ ] 5.1.7 Granite: IBM granite format
- [ ] 5.1.8 GLM-47: ChatGLM tool format
- [ ] 5.1.9 Kimi: Kimi K2 tool format
- [ ] 5.1.10 MiniMax: MiniMax tool format
- [ ] 5.1.11 Nemotron: Nemotron tool format
- [ ] 5.1.12 Step 3.5: Step tool format
- [ ] 5.1.13 xLAM: Salesforce xLAM format
- [ ] 5.1.14 Auto: automatic parser selection based on model config
- [ ] 5.1.15 All concrete parsers implement abstract base class methods
- [ ] 5.1.16 Single tool call works
- [ ] 5.1.17 Multiple tool calls in one response (parallel tool use)
- [ ] 5.1.18 Tool call with nested JSON arguments
- [ ] 5.1.19 Malformed JSON in tool call: error handling
- [ ] 5.1.20 Tool call + text before/after: text preserved

### 5.2 Tool Call + Reasoning Interaction

- [ ] 5.2.1 Model thinks, then makes tool call: both extracted correctly
- [ ] 5.2.2 Tool result fed back, model reasons about result
- [ ] 5.2.3 Stop during tool call mid-generation

### 5.3 tool_choice

- [ ] 5.3.1 `auto`: model decides
- [ ] 5.3.2 `none`: tools disabled
- [ ] 5.3.3 `required`: model must call a tool
- [ ] 5.3.4 Forced specific function
- [ ] 5.3.5 tool_choice=required but model generates text: forced tool call

### 5.4 MCP Integration (`mcp/`)

Files: `mcp/client.py`, `mcp/executor.py`, `mcp/manager.py`, `mcp/security.py`, `mcp/tools.py`, `mcp/types.py`, `mcp/config.py`

- [ ] 5.4.1 Client connects to MCP servers
- [ ] 5.4.2 Executor executes tool calls via MCP
- [ ] 5.4.3 Manager lifecycle management of MCP connections
- [ ] 5.4.4 Security: input validation, sandboxing
- [ ] 5.4.5 MCP server crash: client handles disconnection gracefully
- [ ] 5.4.6 Timeout on MCP tool execution: no hanging
- [x] 5.4.7 Rate limiting on `/v1/mcp/execute` (H+ audit fix L18)

### 5.5 Built-in Coding Tools (Electron `tools/`)

Files: `panel/src/main/tools/executor.ts`, `panel/src/main/tools/registry.ts`

- [ ] 5.5.1 Tool executor dispatches tool calls
- [ ] 5.5.2 Tool registry registers available tools
- [ ] 5.5.3 Coding tool integration with Claude Code
- [ ] 5.5.4 Coding tool integration with Codex / OpenCode

---

## 6. API COMPATIBILITY

Files: `server.py`, `api/anthropic_adapter.py`, `api/streaming.py`, `api/tool_calling.py`, `api/models.py`, `api/utils.py`

### 6.1 /v1/chat/completions

- [ ] 6.1.1 Non-streaming: returns complete response
- [ ] 6.1.2 Streaming (SSE): returns chunk-by-chunk with `data:` prefix
- [ ] 6.1.3 Tools: function_call in response when tools invoked
- [ ] 6.1.4 Reasoning: reasoning_content field populated
- [ ] 6.1.5 VLM: image_url content parts processed
- [ ] 6.1.6 Multi-turn: messages array with history
- [ ] 6.1.7 Response format: matches OpenAI spec (id, object, created, model, choices, usage)
- [ ] 6.1.8 Usage counting: prompt_tokens + completion_tokens = total_tokens
- [ ] 6.1.9 finish_reason: "stop", "length", "tool_calls" correctly set
- [ ] 6.1.10 Cancel endpoint: `/v1/chat/completions/{id}/cancel`
- [ ] 6.1.11 Stream=true with tools: tool call chunks formatted correctly

### 6.2 /v1/messages (Anthropic)

- [ ] 6.2.1 Basic message: Anthropic format in -> internal -> Anthropic format out
- [ ] 6.2.2 Streaming: SSE with Anthropic event types
- [ ] 6.2.3 Thinking blocks: `type: "thinking"` in response
- [ ] 6.2.4 tool_use: Anthropic tool format in/out
- [ ] 6.2.5 `AnthropicRequest` -> `ChatCompletionRequest` conversion
- [ ] 6.2.6 `AnthropicStreamAdapter`: all event types mapped
- [ ] 6.2.7 Claude Code connecting as Anthropic client
- [ ] 6.2.8 Anthropic streaming + stop button

### 6.3 /v1/responses (OpenAI Agents)

- [ ] 6.3.1 Event types: response.created, response.output_item.added, etc.
- [ ] 6.3.2 Function calls: call_id, name, arguments
- [ ] 6.3.3 Streaming: SSE with Responses API event format
- [ ] 6.3.4 OpenAI Agents SDK connecting via Responses API

### 6.4 /v1/completions

- [ ] 6.4.1 Text completion: prompt -> generated text
- [ ] 6.4.2 Streaming: SSE chunks
- [ ] 6.4.3 Empty prompt: handled
- [ ] 6.4.4 Very long prompt (near context limit)

### 6.5 /v1/images/generations

- [ ] 6.5.1 Schnell: text -> image
- [ ] 6.5.2 Dev: text -> image
- [ ] 6.5.3 Z-Image-Turbo: text -> image
- [ ] 6.5.4 Size parameter: various resolutions
- [ ] 6.5.5 Seed parameter: reproducible output
- [ ] 6.5.6 Response format: OpenAI images response
- [ ] 6.5.7 Invalid size: error handling
- [ ] 6.5.8 Model not loaded: appropriate error

### 6.6 /v1/images/edits

- [ ] 6.6.1 Image + instruction -> edited image (Qwen Image Edit)
- [ ] 6.6.2 Mask parameter: applies mask correctly
- [ ] 6.6.3 Strength parameter: controls edit intensity
- [ ] 6.6.4 Image without mask
- [ ] 6.6.5 Non-square image

### 6.7 /v1/embeddings

- [ ] 6.7.1 Single input: returns embedding vector
- [ ] 6.7.2 Batch input: returns multiple vectors
- [ ] 6.7.3 Model swap: loads embedding model if text model loaded
- [ ] 6.7.4 Embedding dimensions match model spec
- [ ] 6.7.5 Very long input text (beyond model max)

### 6.8 /v1/rerank

- [!] 6.8.1 Causal path TokenizerWrapper `__call__` bug — line 187 in `_score_causal()` (BUG todo #1)
- [ ] 6.8.2 Encoder path: bi-encoder reranking
- [ ] 6.8.3 Late-interaction path (PR #22 Jina v3 — waiting on contributor)
- [x] 6.8.4 JSON parse in /v1/rerank (fixed session 2026-03-21c)
- [~] 6.8.5 Thread safety in reranker `_load()` — no mutex (WARN #12)
- [ ] 6.8.6 Rerank 100+ documents: context length guard

### 6.9 /v1/audio/*

- [ ] 6.9.1 `/v1/audio/transcriptions`: audio file -> text (STT)
- [ ] 6.9.2 `/v1/audio/speech`: text -> audio (TTS)
- [ ] 6.9.3 `/v1/audio/voices`: list available voices
- [x] 6.9.4 STT lock: `_stt_lock` prevents race condition (H+ audit fix H3)
- [x] 6.9.5 TTS lock: `_tts_lock` prevents race condition (H+ audit fix H4)
- [x] 6.9.6 `.unload()` called before model reassignment in both STT and TTS (H+ audit fix)

### 6.10 Utility Endpoints

- [ ] 6.10.1 `/health`: returns status, model info, memory info
- [ ] 6.10.2 `/v1/models`: returns loaded model name
- [ ] 6.10.3 `/v1/cache/stats`: returns hit rate, size, entries
- [ ] 6.10.4 `/v1/cache/entries`: returns individual cache entries
- [ ] 6.10.5 `/v1/cache/warm`: pre-populates cache with prompt
- [ ] 6.10.6 `DELETE /v1/cache`: clears cache (including L2 disk store)
- [ ] 6.10.7 /health during model loading: returns "loading" state
- [ ] 6.10.8 /health after sleep: returns "sleeping" state

### 6.11 Admin Endpoints

- [ ] 6.11.1 `/admin/soft-sleep`: clears cache, reduces Metal limit
- [ ] 6.11.2 `/admin/deep-sleep`: unloads model
- [ ] 6.11.3 `/admin/wake`: reloads model
- [ ] 6.11.4 Wake from soft-sleep: model responds correctly
- [ ] 6.11.5 Wake from deep-sleep: model reloaded and responds
- [ ] 6.11.6 Sleep during active generation: current request handled
- [x] 6.11.7 Wake failure sets `_standby_state = None` + `_model_load_error` (H+ audit fix F2)

### 6.12 Rate Limiting & Auth

- [ ] 6.12.1 API key verification: correct key passes, wrong key rejected
- [ ] 6.12.2 Rate limiting: excessive requests throttled
- [ ] 6.12.3 No API key configured: all requests pass
- [x] 6.12.4 Server-side output getattr safety for prompt_tokens/completion_tokens (H+ audit fix L19)

---

## 7. ELECTRON APP

### 7.1 Session Lifecycle

Files: `panel/src/main/ipc/sessions.ts`, `panel/src/main/sessions.ts`

- [ ] 7.1.1 Create session: DB entry created, settings initialized
- [ ] 7.1.2 Start session: engine process spawned, health check passes
- [ ] 7.1.3 Stop session: engine process killed, port freed
- [ ] 7.1.4 Delete session: DB entry removed, chat history cleared
- [ ] 7.1.5 Sleep session: soft-sleep or deep-sleep via admin endpoint
- [ ] 7.1.6 Wake session: wake via admin endpoint or JIT on request
- [ ] 7.1.7 Zombie process cleanup: orphaned vmlx-engine processes killed
- [ ] 7.1.8 Port allocation: no port conflicts between sessions
- [ ] 7.1.9 DB state sync: session state in DB matches actual process state
- [ ] 7.1.10 Create -> start -> chat -> stop -> delete: full lifecycle
- [ ] 7.1.11 Multiple sessions running simultaneously
- [ ] 7.1.12 JIT wake: send chat to sleeping session, auto-wakes
- [x] 7.1.13 Loading progress regex matches JANG v1/v2 patterns (H+ audit fix M9)
- [x] 7.1.14 Loading progress regex case-insensitive for BatchedEngine (H+ audit fix M10)

### 7.2 Chat Interface

Files: `panel/src/main/ipc/chat.ts`, `panel/src/renderer/src/components/chat/` (10 files)

- [ ] 7.2.1 `ChatInterface.tsx`: message input, send, receive display
- [ ] 7.2.2 `MessageBubble.tsx`: user/assistant message rendering
- [ ] 7.2.3 `MessageList.tsx`: scrollable message list with auto-scroll
- [ ] 7.2.4 `InputBox.tsx`: text input, send button, keyboard shortcuts
- [ ] 7.2.5 `ChatList.tsx`: conversation list in sidebar
- [ ] 7.2.6 `ChatSettings.tsx`: per-chat settings
- [ ] 7.2.7 `ReasoningBox.tsx`: collapsible reasoning display
- [ ] 7.2.8 `ToolCallStatus.tsx`: tool call progress/result display
- [ ] 7.2.9 `InlineToolCall.tsx`: inline tool call rendering
- [ ] 7.2.10 `VoiceChat.tsx`: voice input/output
- [ ] 7.2.11 `chat-utils.ts`: shared chat helper functions
- [x] 7.2.12 Streaming typewriter: renderer-side implementation (DONE, never touch main process)
- [ ] 7.2.13 Auto-scroll: scrolls during streaming, pauses when user scrolls up
- [x] 7.2.14 `[THINK]` -> `<think>` normalization + client-side fallback parser (session 2026-03-23)
- [ ] 7.2.15 Very long message: rendering performance
- [ ] 7.2.16 Code blocks: syntax highlighting
- [ ] 7.2.17 Rapid send: multiple messages before first response
- [ ] 7.2.18 Chat history persistence across app restart

### 7.3 Image Tab

Files: `panel/src/main/ipc/image.ts`, `panel/src/renderer/src/components/image/` (8 files), `panel/src/shared/imageModels.ts`

- [ ] 7.3.1 `ImageTab.tsx`: main image generation interface
- [ ] 7.3.2 `ImageModelPicker.tsx`: model selection dropdown
- [ ] 7.3.3 `ImagePromptBar.tsx`: prompt input
- [ ] 7.3.4 `ImageGallery.tsx`: generated images display
- [ ] 7.3.5 `ImageHistory.tsx`: past generations
- [ ] 7.3.6 `ImageSettings.tsx`: steps, size, seed, quantization
- [ ] 7.3.7 `ImageTopBar.tsx`: top controls
- [ ] 7.3.8 `MaskPainter.tsx`: mask drawing for image editing
- [ ] 7.3.9 Redo buttons always visible below each image card
- [ ] 7.3.10 Image session shows only Server Settings (no text inference settings)
- [~] 7.3.11 ImageSettings quantize dropdown dead after server start (by design)
- [ ] 7.3.12 `imageMode` explicit setting (NO regex for model detection)
- [ ] 7.3.13 Select model -> auto-start -> prompt -> generate: full flow

### 7.4 Server Tab

Files: `panel/src/renderer/src/components/sessions/` (10 files)

- [ ] 7.4.1 `SessionView.tsx`: server session dashboard
- [ ] 7.4.2 `SessionDashboard.tsx`: overview with status
- [ ] 7.4.3 `SessionSettings.tsx`: model settings
- [ ] 7.4.4 `ServerSettingsDrawer.tsx`: server config panel
- [ ] 7.4.5 `SessionConfigForm.tsx`: configuration form
- [ ] 7.4.6 `LogsPanel.tsx`: server log viewer
- [ ] 7.4.7 `SessionCard.tsx`: session card in list
- [ ] 7.4.8 `CreateSession.tsx`: new session creation
- [ ] 7.4.9 `BenchmarkPanel.tsx`: benchmark display
- [ ] 7.4.10 `CachePanel.tsx`: cache stats display
- [ ] 7.4.11 `EmbeddingsPanel.tsx`: embedding generation UI
- [ ] 7.4.12 `PerformancePanel.tsx`: performance metrics
- [ ] 7.4.13 `DirectoryManager.tsx`: model directory management
- [ ] 7.4.14 `DownloadTab.tsx`: model downloads
- [x] 7.4.15 SSD-related InfoNotes guarded with `!ssdActive &&` (H+ audit fix L21)
- [x] 7.4.16 Draft model input/slider disabled when SSD active (H+ audit fix M11)

### 7.5 Downloads

Files: `panel/src/renderer/src/components/DownloadsView.tsx`, `DownloadStatusBar.tsx`

- [ ] 7.5.1 HuggingFace model search: results displayed
- [ ] 7.5.2 Download progress: percentage and speed
- [ ] 7.5.3 Pause/resume: download resumes from where it left off
- [ ] 7.5.4 DownloadStatusBar auto-expands (NO silent downloads EVER)
- [ ] 7.5.5 No duplicate downloads for same model
- [ ] 7.5.6 Download cleanup on cancel: partial files removed
- [ ] 7.5.7 Download during active inference

### 7.6 Tools Tab

Files: `panel/src/renderer/src/components/tools/` (5 files)

- [ ] 7.6.1 `ToolsDashboard.tsx`: tools overview
- [ ] 7.6.2 `ModelConverter.tsx`: JANG conversion UI
- [ ] 7.6.3 `ModelDoctor.tsx`: model health check
- [ ] 7.6.4 `ModelInspector.tsx`: model architecture viewer
- [ ] 7.6.5 `LogViewer.tsx`: log analysis tool
- [ ] 7.6.6 `useStreamingOperation.ts`: streaming hook for long ops

### 7.7 API Dashboard

Files: `panel/src/renderer/src/components/api/` (4 files)

- [ ] 7.7.1 `ApiDashboard.tsx`: API status and documentation
- [ ] 7.7.2 `CodeSnippets.tsx`: copy-paste code examples
- [ ] 7.7.3 `EndpointList.tsx`: all endpoints listed
- [ ] 7.7.4 `CodingToolIntegration.tsx`: IDE integration setup

### 7.8 Layout & Navigation

Files: `panel/src/renderer/src/components/layout/` (5 files)

- [ ] 7.8.1 `Sidebar.tsx`: navigation between modes (Chat, Server, Image, Tools, API)
- [ ] 7.8.2 `SidebarHeader.tsx`: branding, version
- [ ] 7.8.3 `TitleBar.tsx`: window controls, flag button, language picker
- [ ] 7.8.4 `ChatHistory.tsx`: conversation history in sidebar
- [ ] 7.8.5 `ChatModeToolbar.tsx`: mode-specific toolbar
- [ ] 7.8.6 Switch between all 5 modes: state preserved
- [ ] 7.8.7 Dark/light theme toggle

### 7.9 Other UI Components

- [ ] 7.9.1 `SetupScreen.tsx`: first-run setup
- [ ] 7.9.2 `UpdateBanner.tsx` / `UpdateNotice.tsx`: update notifications
- [ ] 7.9.3 `Toast.tsx`: notification toasts
- [ ] 7.9.4 `Modal.tsx`: modal dialogs
- [ ] 7.9.5 `App.tsx`: root component, mode routing

### 7.10 IPC Channels (14 files)

Files: `panel/src/main/ipc/` (14 .ts files)

- [ ] 7.10.1 `chat.ts`: message send/receive, streaming, abort
- [ ] 7.10.2 `sessions.ts`: CRUD, start/stop, sleep/wake
- [ ] 7.10.3 `models.ts`: model list, download, delete
- [ ] 7.10.4 `image.ts`: image generation, editing
- [ ] 7.10.5 `engine.ts`: engine lifecycle
- [ ] 7.10.6 `cache.ts`: cache operations
- [ ] 7.10.7 `audio.ts`: audio recording, playback
- [ ] 7.10.8 `benchmark.ts`: benchmark operations
- [ ] 7.10.9 `embeddings.ts`: embedding operations
- [ ] 7.10.10 `developer.ts`: dev tools, debug info
- [ ] 7.10.11 `export.ts`: chat export
- [ ] 7.10.12 `performance.ts`: perf metrics
- [ ] 7.10.13 `coding-tools.ts`: coding tool integration
- [ ] 7.10.14 `utils.ts`: utility IPC calls
- [ ] 7.10.15 Three-layer IPC integrity: Main -> Preload -> Renderer

### 7.11 Main Process

Files: `panel/src/main/` (10 .ts files)

- [ ] 7.11.1 `index.ts`: app lifecycle, window management
- [ ] 7.11.2 `database.ts`: SQLite WAL mode, schema migrations
- [ ] 7.11.3 `process-manager.ts`: vmlx-engine process spawn/kill
- [ ] 7.11.4 `engine-manager.ts`: engine coordination
- [ ] 7.11.5 `sessions.ts` (main): session state management
- [ ] 7.11.6 `server.ts` (main): local server for renderer
- [ ] 7.11.7 `tray.ts`: system tray (listens to BOTH ProcessManager AND SessionManager)
- [ ] 7.11.8 `memory-enforcer.ts`: Metal memory monitoring
- [ ] 7.11.9 `model-config-registry.ts`: Electron-side model config
- [ ] 7.11.10 `update-checker.ts`: auto-update check against latest.json
- [ ] 7.11.11 `db/model-settings.ts`: per-model settings persistence

### 7.12 Contexts & Shared

Files: `panel/src/renderer/src/contexts/` (2 files), `panel/src/shared/` (2 files), `panel/src/preload/index.ts`

- [ ] 7.12.1 `AppStateContext.tsx`: global app state
- [ ] 7.12.2 `SessionsContext.tsx`: session state (includes config field on SessionSummary)
- [ ] 7.12.3 `imageModels.ts`: image model definitions
- [ ] 7.12.4 `sessionUtils.ts`: shared session utilities
- [ ] 7.12.5 `preload/index.ts`: safe API bridge
- [ ] 7.12.6 `env.d.ts`: TypeScript declarations match actual IPC channels

### 7.13 i18n

- [x] 7.13.1 I18nProvider + locale files rebuilt (session 2026-03-23)
- [~] 7.13.2 Only ~5% of UI uses translations — LOW priority cosmetic
- [~] 7.13.3 ~300+ hardcoded strings across 50+ components
- [!] 7.13.4 Dead i18n file: `panel/src/renderer/src/i18n/index.tsx` — delete it (BUG todo #4)

---

## 8. MODEL-SPECIFIC COMPATIBILITY

### 8.1 Mistral 4

- [x] 8.1.1 MLA: kv_b_proj split, head inflation, no KV quant (session 2026-03-23)
- [x] 8.1.2 MoE: expert routing, gate dequant in both text + VLM loaders (session 2026-03-23)
- [x] 8.1.3 VLM: image_token_index=10 [IMG], vision encoder, _Mistral4VLMBackbone (session 2026-03-23)
- [ ] 8.1.4 Reasoning: [THINK]/[/THINK], reasoning_effort "none"/"high" — verify end-to-end
- [ ] 8.1.5 Tool calling: mistral tool parser native format
- [x] 8.1.6 think_in_template=False mapping (session 2026-03-23)
- [ ] 8.1.7 JANG 2L/4M quantization targets
- [~] 8.1.8 2-bit quantized vision encoder poor quality — known model limitation
- [!] 8.1.9 RoPE error in JANG v2 VLM loader — rope_parameters missing from config — CURRENT BLOCKER
- [ ] 8.1.10 Prefix cache gen_prompt_len for Mistral reasoning

### 8.2 Nemotron-H (Hybrid SSM)

- [x] 8.2.1 MambaCache layers + attention layers coexistence (session 2026-03-21)
- [x] 8.2.2 Chunked prefill: broadcast fix (root cause session 2026-03-21)
- [x] 8.2.3 MambaCache merge in continuous batching (session 2026-03-21)
- [x] 8.2.4 CacheList for MoE layers within hybrid SSM (session 2026-03-21)
- [x] 8.2.5 QuantizedKVCache list/tuple reconstruction for 40+8 layers (session 2026-03-21c)
- [ ] 8.2.6 Boundary snapshots for MambaCache state

### 8.3 Nemotron Cascade

- [x] 8.3.1 Gate dequant: 8-bit high-to-low (confirmed)
- [x] 8.3.2 fc1/fc2 rename (confirmed)
- [x] 8.3.3 Confirmed: 42GB / 46 tok/s

### 8.4 DeepSeek V3

- [ ] 8.4.1 MLA: absorbed attention, latent KV
- [ ] 8.4.2 MoE: shared expert + routed experts
- [ ] 8.4.3 CacheList: per-expert cache management
- [ ] 8.4.4 bfloat16 computation

### 8.5 Qwen3 / Qwen3-VL

- [ ] 8.5.1 Thinking: `<think>` tags, enable_thinking toggle
- [ ] 8.5.2 Tool calling: qwen tool parser
- [ ] 8.5.3 VLM: vision_config, image processing
- [ ] 8.5.4 Multi-turn with thinking on/off

### 8.6 MiniMax

- [ ] 8.6.1 Always-thinks template: think_in_template detection
- [ ] 8.6.2 No thinking toggle (always produces thinking)

### 8.7 Kimi K2

- [ ] 8.7.1 MoE: expert routing
- [ ] 8.7.2 Thinking support
- [ ] 8.7.3 Tool calling: kimi tool parser

### 8.8 Llama

- [ ] 8.8.1 Standard attention: GQA
- [ ] 8.8.2 Tool calling: llama tool parser
- [ ] 8.8.3 All quantization levels (JANG + standard)

### 8.9 Gemma

- [ ] 8.9.1 Standard attention
- [ ] 8.9.2 Sliding window (if applicable)

### 8.10 GLM-Z1

- [ ] 8.10.1 Harmony protocol
- [ ] 8.10.2 GLM-47 tool parser

---

## 9. IMAGE GENERATION

Files: `image_gen.py`, `panel/src/main/ipc/image.ts`, image components

### 9.1 mflux Models (`image_gen.py`)

- [ ] 9.1.1 Schnell (`Flux1`): dual encoder, local loading
- [ ] 9.1.2 Dev (`Flux1`): dual encoder, local loading
- [ ] 9.1.3 Z-Image-Turbo (`ZImage`): single encoder, local loading
- [~] 9.1.4 Klein: REMOVED (mflux single-encoder limitation)
- [~] 9.1.5 mflux 0.16.9 quantized model loading broken (MLX version conflict)
- [ ] 9.1.6 Generate at 512x512, 1024x1024, custom sizes
- [ ] 9.1.7 Same seed produces same image

### 9.2 Image Editing (Qwen Image Edit)

- [ ] 9.2.1 Instruction-based editing: image + text -> edited image
- [ ] 9.2.2 Full precision only (~54GB requirement)
- [ ] 9.2.3 Edit with mask via MaskPainter

### 9.3 Model Detection

- [ ] 9.3.1 `model_index.json` detection for diffusion models
- [ ] 9.3.2 Single encoder vs dual encoder detection
- [ ] 9.3.3 imageMode explicit setting — no regex

---

## 10. POWER MANAGEMENT

Files: `server.py` (admin endpoints), `panel/src/main/sessions.ts` (sleep/wake IPC), `panel/src/main/memory-enforcer.ts`

### 10.1 Soft Sleep

- [ ] 10.1.1 Cache cleared on soft sleep
- [ ] 10.1.2 Metal memory limit reduced
- [ ] 10.1.3 Model stays loaded (weights in memory)
- [ ] 10.1.4 Server process stays alive
- [ ] 10.1.5 Soft sleep -> chat request: auto-wake
- [x] 10.1.6 Soft sleep + stream mode handles None caches (session 2026-03-22)

### 10.2 Deep Sleep

- [ ] 10.2.1 Model fully unloaded from GPU memory
- [ ] 10.2.2 Server process stays alive
- [ ] 10.2.3 Memory reclaimed (Metal memory drops)
- [ ] 10.2.4 Deep sleep -> JIT wake -> chat
- [x] 10.2.5 Deep sleep + stream mode: lazy re-applied (session 2026-03-22)
- [x] 10.2.6 Deep sleep + JANG: gate dequant re-runs (session 2026-03-22)
- [x] 10.2.7 Speculative draft model unloaded on deep sleep (H+ audit fix M7)

### 10.3 JIT Wake

- [ ] 10.3.1 Chat request to sleeping session triggers auto-load
- [ ] 10.3.2 API request to sleeping session triggers auto-load
- [ ] 10.3.3 Wake completes before response generation starts
- [ ] 10.3.4 Multiple JIT wakes: no double-load race condition
- [ ] 10.3.5 Multiple sessions with JIT: no cross-contamination
- [x] 10.3.6 Wake failure sets error state, prevents infinite retry loop (H+ audit fix F2)

### 10.4 Idle Timer

- [ ] 10.4.1 Per-session idle timer triggers sleep after configured duration
- [ ] 10.4.2 Timer reset on activity
- [ ] 10.4.3 Timer fires during active generation: should NOT sleep
- [ ] 10.4.4 Set idle timer, send request just before expiry: timer resets

### 10.5 State Tracking

- [ ] 10.5.1 DB state matches actual process state
- [ ] 10.5.2 Server health endpoint reflects sleep state
- [ ] 10.5.3 UI shows correct status icon/text

---

## 11. CLI

Files: `cli.py`, `commands/convert.py`, `commands/doctor.py`, `commands/info.py`, `commands/list.py`

- [ ] 11.1 `vmlx` command: main entry point
- [ ] 11.2 `vmlx-serve`: server mode
- [ ] 11.3 `vmlx-engine`: engine mode
- [ ] 11.4 `convert` command: model conversion
- [ ] 11.5 `doctor` command: model health check
- [ ] 11.6 `info` command: model info display
- [ ] 11.7 `list` command: list available models
- [ ] 11.8 CLI quantize auto-detection checks both model name and path dir
- [ ] 11.9 `--image-mode` flag dispatches to load_edit_model

---

## 12. ADDITIONAL SUBSYSTEMS

### 12.1 Embedding (`embedding.py`)

- [ ] 12.1.1 Encoder model loading
- [ ] 12.1.2 Batch embedding computation
- [ ] 12.1.3 Dimension normalization
- [~] 12.1.4 `_embedding_lock` lazy init (theoretical race, practically impossible)

### 12.2 Reranker (`reranker.py`)

- [!] 12.2.1 Causal path `_score_causal()` TokenizerWrapper `__call__` bug — line 187 needs unwrap (BUG todo #1)
- [ ] 12.2.2 Encoder path: bi-encoder reranking
- [~] 12.2.3 Thread safety in `_load()` — no mutex (WARN #12)
- [x] 12.2.4 Reranker model_path uses lock-captured local ref (H+ audit fix L17)

### 12.3 Multimodal Processor (`multimodal_processor.py`)

- [ ] 12.3.1 Image preprocessing: resize, normalize
- [ ] 12.3.2 Multi-image handling

### 12.4 Audio (`audio/processor.py`, `audio/stt.py`, `audio/tts.py`)

- [ ] 12.4.1 Audio processor: format conversion
- [ ] 12.4.2 STT: model loading, transcription
- [ ] 12.4.3 TTS: voice selection, audio generation
- [ ] 12.4.4 WAV, MP3, M4A input formats

### 12.5 Chat Templates (`utils/chat_templates.py`)

- [ ] 12.5.1 Jinja2 template application
- [ ] 12.5.2 System prompt injection
- [ ] 12.5.3 Multi-turn formatting

### 12.6 Tokenizer Utils (`utils/tokenizer.py`)

- [ ] 12.6.1 Tokenizer loading
- [ ] 12.6.2 TokenizerWrapper: encode/decode

### 12.7 Nemotron Latent MoE (`utils/nemotron_latent_moe.py`)

- [ ] 12.7.1 Nemotron latent MoE expert routing
- [ ] 12.7.2 Integration with scheduler

### 12.8 API Models (`api/models.py`)

- [ ] 12.8.1 Pydantic request/response models
- [ ] 12.8.2 All fields with correct types and defaults

### 12.9 Benchmark (`benchmark.py`)

- [ ] 12.9.1 Throughput measurement: tokens/sec
- [ ] 12.9.2 Latency measurement: time-to-first-token
- [ ] 12.9.3 Memory measurement: peak GPU usage

### 12.10 Gradio Apps (`gradio_app.py`, `gradio_text_app.py`)

- [ ] 12.10.1 Web UI for standalone usage
- [ ] 12.10.2 Text-only mode

### 12.11 Plugin System (`plugin.py`)

- [ ] 12.11.1 Plugin loading and registration
- [ ] 12.11.2 Plugin lifecycle management

### 12.12 Model Registry (`model_registry.py`)

- [ ] 12.12.1 Recommended models list
- [ ] 12.12.2 Model family classification

---

## 13. BUILD & RELEASE

- [ ] 13.1 Source -> bundled sync: `cp -R vmlx_engine/* panel/bundled-python/...`
- [ ] 13.2 `npm run build`: no TypeScript errors
- [ ] 13.3 `npx electron-builder --mac --dir`: produces .app
- [ ] 13.4 App launches from /Applications
- [ ] 13.5 DMG build with notarization: `source .env.signing && npx electron-builder --mac dmg`
- [ ] 13.6 Apple notarization passes (Gatekeeper won't block)
- [ ] 13.7 DMG uploaded to GitHub release (mlxstudio repo)
- [ ] 13.8 latest.json updated on mlxstudio repo (auto-updater)
- [ ] 13.9 PyPI package: `pip install vmlx` installs correctly
- [ ] 13.10 Version bumped in all locations
- [ ] 13.11 Python tests: `.venv/bin/pytest tests/ -k "not Async" -v` (2000+ tests)
- [ ] 13.12 Panel tests: `cd panel && npx vitest run` (1545+ tests)

---

## 14. SECURITY

- [ ] 14.1 API key not logged in plaintext
- [ ] 14.2 Rate limiting prevents abuse
- [ ] 14.3 Input validation on all endpoints (no injection)
- [ ] 14.4 MCP tool execution sandboxed
- [ ] 14.5 No `nodeIntegration: true` in renderer
- [ ] 14.6 Context isolation enabled
- [ ] 14.7 Preload script exposes only safe APIs
- [ ] 14.8 No credentials in source code (`.env.signing` gitignored)
- [ ] 14.9 API keys not stored in plaintext in DB

---

## 15. NAMING CONSISTENCY (CLI <-> Python <-> TypeScript)

| Python | CLI | TypeScript | Status |
|--------|-----|-----------|--------|
| stream_from_disk | --stream-from-disk | streamFromDisk | [x] Added to buildArgs + RESTART_REQUIRED_KEYS |
| stream_memory_percent | --stream-memory-percent | streamMemoryPercent | [ ] Verify |
| enable_prefix_cache | --enable-prefix-cache | enablePrefixCache | [ ] Verify |
| use_paged_cache | --use-paged-cache | usePagedCache | [ ] Verify |
| kv_cache_quantization | --kv-cache-quantization | kvCacheQuantization | [ ] Verify |
| enable_disk_cache | --enable-disk-cache | enableDiskCache | [ ] Verify |
| enable_block_disk_cache | --enable-block-disk-cache | enableBlockDiskCache | [ ] Verify |
| max_num_seqs | --max-num-seqs | maxNumSeqs | [ ] Verify |
| tool_call_parser | --tool-call-parser | toolCallParser | [ ] Verify |
| reasoning_parser | --reasoning-parser | reasoningParser | [ ] Verify |

---

## 16. CROSS-CUTTING CONCERNS

### 16.1 Sleep/Wake Matrix

Test every subsystem after sleep/wake:
- [ ] 16.1.1 Prefix cache after soft sleep/wake
- [ ] 16.1.2 KV cache quantization after sleep/wake
- [ ] 16.1.3 Continuous batching after sleep/wake
- [ ] 16.1.4 Paged cache after sleep/wake
- [ ] 16.1.5 Block disk store after sleep/wake
- [ ] 16.1.6 Tool parsers after sleep/wake (stateless or stale state?)
- [ ] 16.1.7 Reasoning parsers after sleep/wake

### 16.2 Model Switch Matrix

- [ ] 16.2.1 Standard -> MLA: cache fully cleared, head config updated
- [ ] 16.2.2 MLA -> Hybrid SSM: MambaCache layers initialized
- [ ] 16.2.3 Text -> VLM: vision encoder loaded
- [ ] 16.2.4 VLM -> Text: vision encoder unloaded, no memory leak

### 16.3 Error Recovery

- [ ] 16.3.1 Engine crash during generation: error reported to UI, session restartable
- [ ] 16.3.2 Network error during download: retry with resume
- [ ] 16.3.3 Corrupt model files: meaningful error message
- [x] 16.3.4 Metal kernel panic: numpy round-trip fix applied (issues #5, #7, #11)
- [ ] 16.3.5 OOM during model load: graceful failure, no zombie process

### 16.4 Concurrency

- [ ] 16.4.1 Two chat sessions to same model: independent outputs
- [ ] 16.4.2 Chat + image generation simultaneously
- [ ] 16.4.3 Download + inference simultaneously
- [ ] 16.4.4 Sleep one session while another is active
- [ ] 16.4.5 JIT wake race: two requests arrive for sleeping session

---

## KNOWN BUGS — MUST FIX

| # | Bug | File | Status |
|---|-----|------|--------|
| B1 | Reranker causal path TokenizerWrapper `__call__` bug | `reranker.py:187` | OPEN |
| B2 | MLLM scheduler `_extract_cache_states` missing GQA head normalization | `mllm_scheduler.py` | OPEN |
| B3 | block_disk_store `_serialize_block` CacheList tag not handled | `block_disk_store.py` | OPEN (low impact) |
| B4 | Dead i18n file `index.tsx` (TS resolves `index.ts` first) | `i18n/index.tsx` | OPEN (trivial) |
| B5 | Mistral 4 RoPE error in JANG v2 VLM loader | `utils/jang_loader.py` | OPEN (BLOCKER) |
| B6 | gen_prompt_len not computed in server.py for prefix cache | `server.py`, `scheduler.py` | OPEN |

## KNOWN LIMITATIONS — NOT FIXABLE

| # | Limitation | Notes |
|---|-----------|-------|
| K1 | SimpleEngine prefill not interruptible | Architecture limitation |
| K2 | mflux 0.16.9 quantized model loading broken | MLX version conflict |
| K3 | JANG VL MoE not working | Sanitizer conflict |
| K4 | 2-bit quantized vision encoders poor quality | Model limitation at 2-bit |
| K5 | macOS 15+ required for Metal language v4 | OS requirement |

## KNOWN WARNS — LOW IMPACT, DOCUMENTED

| # | Warn | Impact |
|---|------|--------|
| W1 | bfloat16 for all MLA models | Correct by design |
| W2 | numpy block_slice skips CacheList | MLX path handles it |
| W3 | `_ensure_batch_cache` checks ArraysCache not MambaCache | Low impact |
| W4 | Reasoning ON but no think tags -> fallback re-emit delay | Rare UX hiccup |
| W5 | q4 KV quant degrades long reasoning on restore | Quality tradeoff |
| W6 | Partial think tags across chunks -> rare char leak | Very rare |
| W7 | GPT-OSS emitted_reasoning shrink edge case | Very rare |
| W8 | CacheList numpy path always "skip" | MLX path works |
| W9 | MLA MLLM batch with head inflation | Single-request OK |
| W10 | `_resolve_model_path` dead code in PR #22 | Contributor code |
| W11 | Thread safety in reranker `_load()` (no mutex) | Theoretical race |

---

## REMAINING FROM 2026-03-16 FINAL AUDIT

These items were identified as open in the 2026-03-16 audit. Check if still applicable:

### HIGH
- [ ] H1: image.ts saveFile has no source path validation
- [ ] H2: Custom model path always gets 'generate' category — needs imageMode selector
- [ ] H3: PerformancePanel hardcodes "JANG" prefix for all quantization display
- [ ] H4: CodeSnippets image edit/gen curl omit `model` field
- [ ] H5: DownloadStatusBar: download errors silently clear bar
- [ ] H6: getSizeEstimate missing all edit models (returns wrong ~12GB for 37GB qwen)
- [ ] H7: image.ts getModelStatus always returns downloaded:false (dead handler)

### MEDIUM
- [ ] M1: Edit size parsing silently defaults to 1024x1024 (gen endpoint raises 400)
- [ ] M2: EmbeddingsPanel hardcoded model list, no custom model input
- [ ] M3: Temperature 0.0 unreachable via server settings slider
- [ ] M4: DownloadStatusBar: filesProgress never rendered
- [ ] M5: DownloadTab: "No MLX models found" shows in Image mode too
- [ ] M6: Session/server mode mismatch clicking edit session with gen server running

### LOW
- [ ] L1: Error state shows as "Stopped" in ChatModeToolbar dropdown
- [ ] L2: flux2-klein-edit in getDefaultSteps but not in NAMED_MODELS
- [ ] L3: ImageHistory: no session rename, long paths display poorly
- [ ] L4: Scroll dead zone between 100-200px from bottom
- [ ] L5: Cancel endpoints lack auth badge in EndpointList
- [ ] L6: convert.py suggests 'convert-gguf-to-hf' which may not exist
- [ ] L7: Redundant require('os')/require('path') in readFile handler

---

## FEATURE REQUESTS (from GitHub Issues)

- [ ] F1: Single port serve all models (mlxstudio #25) — HIGH priority
- [ ] F2: Broader API support: Ollama compat, LM Studio API (vmlx #23)
- [ ] F3: Cluster support: multi-node MLX RDMA (mlxstudio #26) — LONG TERM

---

## STATISTICS

| Category | Total Items | Verified [x] | Bugs [!] | Warns [~] | Unchecked [ ] |
|----------|------------|--------------|----------|-----------|--------------|
| Caching | 57 | 10 | 1 | 4 | 42 |
| Model Loading | 30 | 14 | 1 | 1 | 14 |
| Inference | 64 | 14 | 1 | 2 | 47 |
| Reasoning | 21 | 6 | 0 | 3 | 12 |
| Tool Calling | 29 | 1 | 0 | 0 | 28 |
| API Compat | 62 | 6 | 1 | 1 | 54 |
| Electron App | 101 | 8 | 1 | 3 | 89 |
| Model-Specific | 33 | 12 | 1 | 1 | 19 |
| Image Gen | 16 | 0 | 0 | 2 | 14 |
| Power Mgmt | 26 | 5 | 0 | 0 | 21 |
| CLI | 9 | 0 | 0 | 0 | 9 |
| Additional | 24 | 1 | 1 | 2 | 20 |
| Build/Release | 12 | 0 | 0 | 0 | 12 |
| Security | 9 | 0 | 0 | 0 | 9 |
| Naming | 10 | 1 | 0 | 0 | 9 |
| Cross-Cutting | 17 | 1 | 0 | 0 | 16 |
| 2016-03-16 Remaining | 14 | 0 | 0 | 0 | 14 |
| **TOTAL** | **534** | **79** | **7** | **19** | **429** |
