# Session 2026-03-24 — Audit Checklist with Full Trace Paths

## Changes This Session

### FIX-A1: Reranker Causal Path TokenizerWrapper [VERIFIED]
- **File:** `vmlx_engine/reranker.py:161`
- **Change:** `tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)` + use `tokenizer` for all calls
- **Callers:** `rerank()` → `_score_causal()` (line 93, when `self._backend == "causal"`)
- **All self._tokenizer refs:** L38 (init), L51 (encoder load), L61 (causal load), L121 (_score_encoder), L161 (unwrap), L217 (unload)
- **_score_encoder needs fix?** NO — `mlx_embeddings.load()` returns standard tokenizer, not TokenizerWrapper
- **Downstream:** Internal only, no leak

### FIX-A2: MLLM Scheduler GQA Head Normalization [VERIFIED]
- **File:** `vmlx_engine/mllm_scheduler.py:935-987`
- **New methods:** `_detect_n_kv_heads()`, `_normalize_gqa_state()`
- **Called from:** `_extract_cache_states()` at L1010 (n_kv), L1030 (CacheList sub), L1050 (regular KV)
- **_extract_cache_states callers:** `_store_cache_to_prefix()` at L1597 — ONLY caller
- **Matches scheduler.py?** YES — identical logic at `scheduler.py:458-494`
- **_n_kv_heads_cached reset:** NOT in reset()/deep_reset() — CORRECT (model config, not runtime state; new model = new scheduler instance)
- **Downstream:** prefix_cache.py L1073-1121 receives cache_list format, block_disk_store handles it

### FIX-A3: block_disk_store CacheList Serialize/Deserialize [VERIFIED]
- **File:** `vmlx_engine/block_disk_store.py:671-696` (serialize), `831-861` (deserialize), `882-901` (infer)
- **_serialize_block callers:** `put()` at L287
- **_deserialize_block callers:** `load()` at L224
- **prefix_cache.py format:** L1121 `("cache_list", sub_slices)` — exact match
- **sub_slices format:** `("kv", ks, vs)` or `("skip",)` or `("cumulative", ...)` — all handled
- **Roundtrip:** `sub_count` written at L696, read at L834 — correct
- **Naming:** `layer_{i}_sub_{j}_keys/values` — unique, no collision with other formats

### FIX-A4: L2 Block Disk Cache Default Disabled [VERIFIED]
- **File:** `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx:79`
- **Change:** `enableBlockDiskCache: true` → `false`
- **Read locations:**
  - `SessionConfigForm.tsx:27` (type), `:79` (default), `:434` (checkbox), `:435` (conditional)
  - `SessionSettings.tsx:110` (preview gating)
  - `sessions.ts:851` (RESTART_REQUIRED_KEYS), `:1739` (buildSessionArgs guard)
  - `server.ts:52` (optional type def)
- **DB sessions:** Keep old value — sessions.ts reads stored config, not default
- **Python side:** 3-condition gate in mllm_scheduler.py:422 (enable_prefix + use_paged + enable_block_disk)
- **When false:** block_disk_store = None, all `if self.block_disk_store is not None:` guards skip

### FIX-A5: Remove tool_call_generating from ChatCompletionChunk [VERIFIED]
- **File:** `vmlx_engine/api/models.py:757` — field removed
- **server.py:** Two `buf_chunk.tool_call_generating = True` assignments removed (L4465, L4537)
- **Remaining refs:** ZERO — grep confirms clean across entire codebase
- **Impact on Chat UI:** Minimal — UI handles tool call delays via streaming throttle

### FIX-A6: Mistral [ARGS] Token Stripping [VERIFIED LIVE]
- **File:** `vmlx_engine/tool_parsers/mistral_tool_parser.py:95`
- **Change:** `tool_name = tool_name.replace("[ARGS]", "").strip()`
- **Parser registration:** `@ToolParserManager.register_module("mistral")` at L37
- **Model config:** `model_configs.py:270` — mistral4 has `tool_parser="mistral"`
- **Also uses mistral parser:** mistral3 (L285), devstral (L230, L240, L251)
- **Router:** `server.py:674` → `ToolParserManager.get_tool_parser(_tool_call_parser)` → MistralToolParser
- **Verified live:** curl test returned `"name": "read_file"` (no [ARGS])

---

## Verification Matrix

### API Endpoints (OpenCode/Codex/ClaudeCode Compat)

| # | Endpoint | Method | Verified | Trace |
|---|----------|--------|----------|-------|
| 1 | `/v1/models` | GET | PASS | server.py:1842→list_models()→ModelsResponse |
| 2 | `/v1/chat/completions` | POST (non-stream) | PASS | server.py:3061→create_chat_completion() |
| 3 | `/v1/chat/completions` | POST (stream) | PASS | server.py:3362→stream_chat_completion() |
| 4 | `/v1/chat/completions` + tools | POST (stream) | PASS | server.py:3362→…→_parse_tool_calls_with_parser() |
| 5 | `/v1/chat/completions` + stream_options | POST | PASS | models.py:148→StreamOptions.include_usage |
| 6 | `/v1/completions` | POST | UNTESTED | server.py:3672 |
| 7 | `/v1/messages` (Anthropic) | POST | UNTESTED | server.py:1861→create_anthropic_message() |
| 8 | `/v1/responses` (OpenAI new) | POST | UNTESTED | server.py (if exists) |
| 9 | `/v1/embeddings` | POST | UNTESTED | server.py |
| 10 | `/v1/rerank` | POST | UNTESTED | server.py→reranker.py |
| 11 | `/v1/images/generations` | POST | UNTESTED | server.py |
| 12 | `/v1/images/edits` | POST | UNTESTED | server.py |
| 13 | `/health` | GET | UNTESTED | server.py |
| 14 | `/v1/cache/stats` | GET | UNTESTED | server.py |

### Streaming Format Compliance

| # | Item | Expected | Actual | Status |
|---|------|----------|--------|--------|
| 1 | First chunk has role | `delta: {role: "assistant"}` | Match | PASS |
| 2 | Content chunks | `delta: {content: "..."}` | Match | PASS |
| 3 | Tool call chunks | `delta: {tool_calls: [...]}` | Match | PASS |
| 4 | Tool call id format | alphanumeric string | `uSkwzSfp8` (9 chars) | PASS |
| 5 | Tool call type | `"function"` | Match | PASS |
| 6 | Tool call name | Clean function name | `read_file` (no [ARGS]) | PASS |
| 7 | Tool call arguments | JSON string | `{"path": "README.md"}` | PASS |
| 8 | finish_reason for tools | `"tool_calls"` | Match | PASS |
| 9 | finish_reason for text | `"stop"` or `"length"` | Match | PASS |
| 10 | Terminator | `data: [DONE]\n\n` | Match | PASS |
| 11 | No extra fields | No non-OpenAI fields | No tool_call_generating | PASS |
| 12 | Usage in chunks | When stream_options set | Present in every chunk | PASS |
| 13 | Empty choices final | `choices: []` with usage | Present | PASS |

### Tool Calling Flow

| # | Step | File:Line | Function | Status |
|---|------|-----------|----------|--------|
| 1 | Tools in request | models.py:226 | ChatCompletionRequest.tools | OK |
| 2 | Convert for template | tool_calling.py:383 | convert_tools_for_template() | OK |
| 3 | Pass to engine | server.py:3321 | chat_kwargs["tools"] = ... | OK |
| 4 | Template injection | engine/batched.py:474 | check_and_inject_fallback_tools() | OK |
| 5 | Marker detection | server.py:4426 | _TOOL_CALL_MARKERS list (10 markers) | OK |
| 6 | Buffering | server.py:4446 | tool_call_buffering = True | OK |
| 7 | Post-stream parse | server.py:4631 | _parse_tool_calls_with_parser() | OK |
| 8 | Parser dispatch | server.py:674 | ToolParserManager.get_tool_parser() | OK |
| 9 | Mistral parse | mistral_tool_parser.py:59 | extract_tool_calls() | OK |
| 10 | [ARGS] strip | mistral_tool_parser.py:95 | tool_name.replace("[ARGS]","") | OK |
| 11 | Emit tool chunk | server.py:4668 | ChatCompletionChunkDelta(tool_calls=...) | OK |
| 12 | Set finish_reason | server.py:4688 | finish_reason="tool_calls" | OK |
| 13 | No tools found | server.py:4713 | Flush content with engine finish_reason | OK |
| 14 | tool_choice=none | server.py:4360 | _suppress_tools=True | OK |
| 15 | tool_choice=required | server.py:4808 | SSE error if no tools emitted | OK |

### Cache System Trace

| # | Component | File | Init Guard | None-Safe | Status |
|---|-----------|------|-----------|-----------|--------|
| 1 | Prefix Cache | scheduler.py:352 | enable_prefix_cache | N/A (always init if flag) | OK |
| 2 | Paged Cache | scheduler.py:268 | use_paged_cache | N/A | OK |
| 3 | Block Disk L2 | scheduler.py:268 | enable_block_disk_cache | Yes (None checks) | OK |
| 4 | Disk Cache L2 | scheduler.py:352 | enable_disk_cache | Yes (None checks) | OK |
| 5 | Memory Cache | scheduler.py | no_memory_aware_cache | N/A | OK |
| 6 | MLLM Prefix | mllm_scheduler.py:514 | enable_prefix_cache | N/A | OK |
| 7 | MLLM Paged | mllm_scheduler.py:422 | use_paged_cache | N/A | OK |
| 8 | MLLM Block Disk | mllm_scheduler.py:422 | 3-condition gate | Yes (None checks) | OK |
| 9 | MLLM Disk L2 | mllm_scheduler.py:538 | enable_disk_cache | Yes (None checks) | OK |
| 10 | GQA normalize (LLM) | scheduler.py:1238 | n_kv > 0 | Returns unmodified if n_kv=0 | OK |
| 11 | GQA normalize (MLLM) | mllm_scheduler.py:1010 | n_kv > 0 | Returns unmodified if n_kv=0 | OK |
| 12 | CacheList serialize | block_disk_store.py:671 | tag == "cache_list" | Skips if no sub-slices | OK |
| 13 | CacheList deserialize | block_disk_store.py:831 | layer_type == "cache_list" | Falls back to skip | OK |
| 14 | CacheList infer | block_disk_store.py:882 | has_sub keys | Checked before other types | OK |

### Disk Streaming (SSD) Presence Check

| # | Component | File | Present in Bundled | Status |
|---|-----------|------|--------------------|--------|
| 1 | weight_index.py | vmlx_engine/utils/ | YES | OK |
| 2 | ssd_generate.py | vmlx_engine/utils/ | YES | OK |
| 3 | --stream-from-disk CLI | cli.py:941, :1274 | YES | OK |
| 4 | _stream_from_disk global | server.py:143 | YES | OK |
| 5 | Lazy loading setup | server.py:981-1007 | YES | OK |
| 6 | SessionManager flag | server.py:1457 | YES | OK |

---

## PENDING VERIFICATION (need live testing)

| # | Item | How to Test | Priority |
|---|------|-------------|----------|
| 1 | Disk streaming inference | Load model with --stream-from-disk, send chat | HIGH |
| 2 | L2 disk cache disabled | New session → verify no --enable-block-disk-cache in args | HIGH |
| 3 | L2 disk cache enabled (existing) | Existing session with it on → verify BlockDiskStore init | MEDIUM |
| 4 | OpenCode end-to-end | `opencode` with vMLX provider, /models → select → chat | HIGH |
| 5 | Codex CLI compat | `codex` with vMLX endpoint | MEDIUM |
| 6 | Claude Code compat | `claude` with vMLX endpoint | MEDIUM |
| 7 | Non-streaming tool calls | POST without stream:true + tools | LOW |
| 8 | Multiple concurrent requests | 2+ simultaneous tool-calling requests | LOW |
| 9 | Reranker causal path | POST /v1/rerank with causal model | LOW |
| 10 | CacheList disk roundtrip | MoE model + prefix cache + block disk L2 | LOW |
