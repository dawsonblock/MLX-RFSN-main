# Session 2026-03-23 — All Issues

## FIXED This Session

### 1. CRITICAL: `const` → `let` crash kills ALL streaming (chat.ts:1183)
- **Symptom:** No response shown in UI for ANY model. Server generates tokens, curl works, but UI shows nothing.
- **Root cause:** Line 1183 declares `const content = choice.content as string`, then line 1190 reassigns it: `content = content.replace(...)`. TypeError silently swallowed by try/catch at line 1019.
- **Introduced by:** Mistral 4 `[THINK]`/`[/THINK]` normalization code added this session.
- **Fix:** `const content` → `let content`

### 2. Non-VLM Mistral 4 quantized_matmul crash
- **Symptom:** `ValueError: [quantized_matmul] shapes incompatible w.shape() == (32,256,16) and scales.shape() == (32,256,1) with group_size=64 and bits=2`
- **Root cause:** `_fix_quantized_bits` didn't know about `QuantizedMultiLinear`. After sanitize loads 8-bit kv_b_proj split weights into QuantizedMultiLinear(bits=2), bits/group_size never get corrected.
- **Fix:** Added `QuantizedMultiLinear` to quant_types in `_fix_quantized_bits`

### 3. VLM Mistral 4 garbage output
- **Symptom:** Model loads, generates tokens, but output is garbage through server (works in direct Python).
- **Root cause:** VLM JANG loader never splits kv_b_proj into embed_q/unembed_out. Both MultiLinear modules keep random initialization values.
- **Fix:** Added kv_b_proj split logic in `_load_jang_v2_vlm` for mistral4 models.

### 4. Missing bfloat16 for MLA models
- **Symptom:** float16 overflow in MoE expert computations → garbage or NaN output.
- **Root cause:** Both loaders only enabled bfloat16 for `n_experts >= 512`. Mistral 4 has 128 experts with MLA — still needs bfloat16.
- **Fix:** Added `kv_lora_rank > 0` and `model_type == "mistral4"` checks to bfloat16 condition in both loaders.

### 5. VLM MoE gate dequant missing
- **Symptom:** MoEGate.weight loaded as uint32 (quantized) → garbage routing decisions.
- **Root cause:** Non-VLM loader had gate dequant logic but VLM loader didn't.
- **Fix:** Added gate dequant (uint32 → bfloat16) in VLM loader weight loop.

### 6. PR #18 — Block disk cache blind bfloat16 cast
- **Applied from:** Fail-Safe PR #18 (DRAFT)
- **Fix:** Store orig_dtype metadata in block disk cache. Deserialize restores original dtype instead of blindly casting float16→bfloat16.

### 7. PR #19 — Metal crash in prompt disk cache + truncation
- **Applied from:** Fail-Safe PR #19 (DRAFT)
- **Fix:** Convert MLX arrays to numpy on main thread before background disk cache write. Truncation slices go through numpy round-trip to avoid lazy MLX ops corrupting Metal command buffers.

## NEEDS INVESTIGATION

### 8. Nemotron prefix cache miss every turn
- **Symptom:** Multi-turn conversation with Nemotron shows "paged cache miss" on every request. No prefix reuse happening.
- **Possible causes:**
  - Could have been masked by Issue #1 (no content = empty messages = different prompt hash)
  - KV cache quantization (q8) might produce different cache keys
  - Tokenizer chat template changes between turns
- **Action:** Re-test after Issue #1 fix. If still happening, trace prefix cache key computation.

### 9. Nemotron responses getting cut off
- **Symptom:** Model generates reasoning (visible) but content is very short ("Hello", "Hi", "No stop —")
- **Possible causes:**
  - WAS caused by Issue #1: `processLine` crash meant `fullContent` never accumulated, `chat:complete` sent empty/minimal content
  - If still happening after fix: could be max_tokens too low, or stop token detection too aggressive
- **Action:** Re-test after Issue #1 fix.

### 10. Mistral 4 reasoning_effort support
- **Current state:** MistralReasoningParser uses `[THINK]`/`[/THINK]` tags. Model supports `reasoning_effort` parameter ("none" for fast, "high" for deep reasoning).
- **What works:** `reasoning_effort` field is forwarded to chat template kwargs (verified in previous session).
- **What needs verification:**
  - Does the chat template actually use `reasoning_effort` to control `[THINK]` injection?
  - Does `reasoning_effort="high"` produce `[THINK]...[/THINK]` in output?
  - Does `reasoning_effort="none"` skip reasoning entirely?
  - The `[THINK]`→`<think>` normalization in chat.ts (the `let content` fix) — does it properly parse Mistral 4 reasoning blocks?
- **Mistral docs say:** `reasoning_effort="none"` = fast, no reasoning. `reasoning_effort="high"` = step-by-step reasoning with [THINK] blocks.
- **Action:** Test with reasoning_effort="high" to see if [THINK] tags appear and are properly parsed.

### 11. Model config registry misses Mistral 4 for VLM wrappers
- **Symptom:** Mistral 4 JANG VLM has top-level `model_type: "mistral3"`, so registry returns DEFAULT config — no tool parser, no reasoning parser auto-detected.
- **Root cause:** Registry only checks top-level model_type, not `text_config.model_type`.
- **Fix:**
  1. Added `mistral3` model config entry (tool_parser=mistral, no reasoning)
  2. Registry now checks `text_config.model_type` for VLM wrappers — if inner model has higher-priority config, uses that
  3. Mistral 4 JANG (mistral3 wrapper, mistral4 inner) now correctly gets mistral4 config with reasoning_parser="mistral"

### 12. Mistral 4 reasoning_effort
- **Key finding:** Only "none" or "high" supported (not low/medium). Template raises exception for other values.
- **Current state:** reasoning_effort already plumbed through server.py → simple.py → chat template kwargs
- **Chat template emits:** `[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]`
- **Model produces:** `[THINK]...[/THINK]` blocks with reasoning_effort="high"
- **App UI:** Reasoning on/off toggle maps to enable_thinking. reasoning_effort is separate — needs dedicated UI control or mapping.
- **Action:** Verify the [THINK] normalization in chat.ts works with the `let` fix.

### 13. Mistral 4 VL image processing broken — FIXED
- **Symptom:** User sends image to Mistral 4 JANG VLM, it doesn't process it.
- **Root cause:** `is_mllm_model()` in `api/utils.py` returns `False` for JANG models. The function checks `jang_config.json` for `has_vision: false` and returns `False` early — never checks `config.json` which HAS `vision_config`.
- **Fix:**
  1. Removed early `return False` — now falls through to check config.json for `vision_config`
  2. Added `is_mllm=True` to the `mistral3` model config entry
- **What works now:** Vision encoder loads, images get processed through MLLM scheduler, `_Mistral4VLMBackbone` receives image embeddings
- **What user reports:** VL seems to be working, but model "says it's something it's not" — possibly the system prompt or model identity in the chat template

### 14. Nemotron prefix cache miss every turn — ROOT CAUSE FOUND
- **Symptom:** Every turn shows "paged cache miss" even in multi-turn conversation
- **Root cause:** The generation prompt tokens (e.g., `<|im_start|>assistant\n<think>\n`) are included in the block hash key. On the next turn, those same positions have the assistant's actual response (`<think></think>response`), so the hash never matches.
- **Affects:** ALL thinking models (Nemotron, Qwen3, DeepSeek-R1) — not just Nemotron
- **Fix needed:** Strip generation prompt tokens before storing cache key. Requires:
  1. Compute gen_prompt_len during template application in server.py
  2. Pass it through as `request._gen_prompt_len` to the scheduler
  3. In `_cleanup_finished`, strip last `gen_prompt_len` tokens from `prompt_tokens` before `store_cache()`
  4. The scheduler.py already has the code: `gen_prompt_len = getattr(request, '_gen_prompt_len', 0)` — just needs the value to be set upstream
- **Status:** Root cause confirmed, fix partially implemented (scheduler side ready), needs server.py gen_prompt_len computation. Complex multi-file change — defer to focused session.

### 15. Mistral 4 model identity / system prompt
- **Symptom:** User says "mistral keeps saying it's something it's not" — model may identify as wrong model or have stale system prompt
- **Possible causes:**
  - Default system prompt in the chat template identifies as "Mistral-Small-4-119B-2603" / "Le Chat"
  - The JANG model may have a modified system prompt or the template may inject one
  - Session system prompt override not reaching the template correctly
- **Action:** Check the tokenizer_config.json chat template for default_system_message

## WRITTEN

### 14. Cache Innovation Roadmap
- `docs/CACHE-INNOVATION-ROADMAP.md` — 6 approaches: MLA-native compressed KV, asymmetric paged blocks, SSM state checkpointing, expert activation caching, speculative prefill, unified memory topology.

## PR STATUS

| PR | Title | Status | Action |
|----|-------|--------|--------|
| #10 | Electron dev mode fixes | MERGED (in our main) | Already in codebase |
| #18 | Block disk cache orig_dtype | DRAFT → Applied locally | Code applied |
| #19 | Metal crash prevention | DRAFT → Applied locally | Code applied |
| #9 | QuantizedKVCache list/tuple | CLOSED | Already fixed independently |
