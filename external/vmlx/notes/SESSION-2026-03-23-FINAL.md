# Session 2026-03-23 — Final Summary

## 23 Fixes Applied

### Critical (broke ALL models)
1. **`const→let` in chat.ts:1183** — `[THINK]`→`<think>` normalization reassigned a `const`. TypeError silently swallowed by try/catch. Killed ALL streaming for ALL models.

### Mistral 4 MLA+MoE (7 fixes)
2. **Non-VLM quantized_matmul crash** — `_fix_quantized_bits` didn't handle `QuantizedMultiLinear`. Added to quant_types.
3. **VLM garbage output** — kv_b_proj→embed_q/unembed_out split missing in `_load_jang_v2_vlm`. Added with dequant + reshape + float16 store.
4. **Missing bfloat16** — Both loaders now check `kv_lora_rank > 0` not just `n_experts >= 512`.
5. **VLM MoE gate dequant** — Added to VLM loader (was only in text loader).
6. **Model config registry** — Added text_config.model_type lookup for VLM wrappers (mistral3→mistral4).
7. **mistral3 config entry** — Added with `is_mllm=True`, `tool_parser=mistral`.
8. **LLM scheduler MLA guard** — KV cache quant auto-disabled for MLA models (was only in MLLM scheduler).

### VLM Detection (2 fixes)
9. **`is_mllm_model()` early return** — Removed `return False` for JANG models. Now falls through to config.json vision_config check.
10. **jang_config.json** — Fixed `has_vision: false` → `true` for Mistral 4 JANG 2L.

### Reasoning (3 fixes)
11. **enable_thinking→reasoning_effort auto-map** — Mistral parser: True→"high", False→"none". Both Chat Completions + Responses API.
12. **Reasoning Effort UI** — Only Auto/High for mistral parser (not Low/Med). GPT-OSS keeps all 4.
13. **Pixtral VisionConfig** — Added `rope_parameters` extraction (same pattern as TextConfig).

### Prefix Cache (4 fixes)
14. **gen_prompt_len stripping** — Strips generation prompt tokens from cache key. Computed in batched.py, passed through engine_core.py, consumed in scheduler.py.
15. **Cache data re-truncation** — KV data truncated to match shortened key. Prevents `<unk>` flood on cache hit.
16. **MLA head validation** — `_get_n_kv_heads()` returns 1 for MLA models (was returning 32 from config → 100% cache miss).
17. **prefix_cache clear()** — Now resets `_n_kv_heads = None` on model switch.

### Cache Safety (PRs #18 + #19)
18. **Block disk cache orig_dtype** — Records dtype before bf16→f16 cast. Restores on deserialize. Backward compatible.
19. **Metal crash numpy round-trip** — disk_cache.py: MLX→numpy before background write. scheduler.py: numpy round-trip for truncation slices.

### MLLM Scheduler (1 fix)
20. **CacheList handling in `_truncate_hybrid_cache`** — Added recursive CacheList truncation (was passing through untruncated).

### UI (2 fixes)
21. **HF model size display** — Was showing parameter count as GB. Now computes actual bytes from per-dtype breakdown (F16×2 + U32×4 etc.).
22. **Attribution comments** — Added to all Mistral 4 and cache-related code sections. Credits waybarrios for base architecture, claims only specific vMLX additions.

### Release
23. **v1.3.9 released** — DMG notarized, uploaded to mlxstudio repo, latest.json updated.

## 8 Deep Audits Completed

| Audit | PASS | WARN | FAIL→Fixed |
|-------|------|------|------------|
| Code cross-check (14 changes) | 13 | 1 | 0 |
| 10 model families | 10 | 0 | 0 |
| Deep cache math | 9 | 1 | 1→fixed |
| Package deps | All | 0 | 0 |
| API compat (12 endpoints) | 12 | 0 | 0 |
| Hybrid SSM edge cases | 18 | 2 | 0 |
| Reasoning + quant interactions | 13 | 4 | 0 |
| MLA + MoE cache paths | 25 | 4 | 3→2 fixed |
| **TOTALS** | **100+** | **12** | **3 fixed** |

## 8 Open Bugs (for next session)

### B1: Reranker causal path `__call__` (LOW)
- **File:** `vmlx_engine/reranker.py:187`
- **Issue:** `self._tokenizer(prompt, ...)` in else branch of `_score_causal()` has same TokenizerWrapper bug as PR #21 fixed for encoder path
- **Fix:** Add `tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)` at top of `_score_causal()`

### B2: MLLM scheduler missing GQA head normalization (MEDIUM)
- **File:** `vmlx_engine/mllm_scheduler.py:917-994`
- **Issue:** `_extract_cache_states` doesn't normalize head inflation in CacheList sub-caches. LLM scheduler does this.
- **Fix:** Add `_detect_n_kv_heads()` to MLLMScheduler, apply normalization

### B3: block_disk_store CacheList tag (LOW)
- **File:** `vmlx_engine/block_disk_store.py:633-695`
- **Issue:** `_serialize_block` doesn't handle "cache_list" tag. MoE models get no disk L2 for CacheList layers.
- **Impact:** Low — numpy path also skips CacheList, so data rarely reaches this function.

### B4: Dead i18n file (LOW)
- **File:** `panel/src/renderer/src/i18n/index.tsx`
- **Issue:** Dead code. TypeScript resolves `index.ts` first.
- **Fix:** Delete the file.

### B5: mlx_vlm bundled audio=None bug (MEDIUM)
- **File:** `panel/bundled-python/.../mlx_vlm/utils.py:1068`
- **Issue:** `prepare_inputs` tries to load audio even when None → NameError on `sf`
- **Impact:** Blocks VLM CLI testing. App uses different code path (MLLM scheduler).

### B6: gen_prompt_len not in SimpleEngine path (MEDIUM)
- **File:** `vmlx_engine/engine/simple.py`
- **Issue:** SimpleEngine doesn't compute gen_prompt_len. Only BatchedEngine does.
- **Impact:** SimpleEngine sessions don't strip gen prompt from prefix cache key → cache misses on multi-turn for thinking models in simple mode.

### B7: Mistral 4 VLM image → empty response (HIGH — needs investigation)
- **Symptom:** Model loads VLM correctly, but `mlx_vlm.generate()` returns empty text (EOS immediately)
- **Root cause unknown:** Could be prompt formatting, image preprocessing, or 2-bit vision encoder quality
- **Cannot test via CLI** — system Python lacks `_Mistral4VLMBackbone`. Must test through app.
- **App testing needed:** Send image through Electron app, check server logs for "1 images" in MLLM path

### B8: Mistral 4 model identifies as "Le Chat" (LOW)
- **Issue:** Model says "I am Le Chat" — baked into training weights, not our code
- **Fix:** Set system prompt in session settings (e.g., "You are a helpful assistant")

## PR Status

| PR | Title | Status |
|----|-------|--------|
| #10 | Electron dev mode fixes | MERGED (already in main) |
| #18 | Block disk cache orig_dtype | MERGED this session |
| #19 | Metal crash prevention | MERGED this session |
| #21 | Reranker TokenizerWrapper | MERGED this session |
| #22 | Jina v3 late-interaction | OPEN — review comments posted, 4 items for contributor |

## GitHub Issues (open, not closing)

| # | Repo | Title | Priority |
|---|------|-------|----------|
| 20 | vmlx | Missing lib (macOS 15+) | LOW — replied |
| 23 | vmlx | Broader API support | MEDIUM — Ollama/LMStudio/Deepgram |
| 25 | mlxstudio | Single port serve all | HIGH — LM Studio-style UX |
| 26 | mlxstudio | Cluster support | LOW — long term |

## Master Checklist Stats

| Status | Count |
|--------|-------|
| [x] Verified PASS | 73 |
| [!] Known bug | 8 |
| [~] Known warn | 20 |
| [ ] Unchecked | 553 |
| **Total items** | **654** |

File: `notes/MASTER-CHECKLIST.md` — single source of truth for all future sessions.

## Files Changed This Session

### Python (vmlx_engine/)
- `utils/jang_loader.py` — QuantizedMultiLinear, kv_b_proj split, bfloat16, gate dequant
- `api/utils.py` — is_mllm_model fall-through
- `model_configs.py` — mistral3+mistral4 entries
- `model_config_registry.py` — text_config.model_type lookup
- `server.py` — enable_thinking→reasoning_effort auto-map
- `scheduler.py` — gen_prompt_len strip, MLA guard, numpy truncation, attribution
- `prefix_cache.py` — MLA H=1, clear() reset, attribution
- `paged_cache.py` — attribution
- `block_disk_store.py` — orig_dtype metadata, attribution
- `disk_cache.py` — numpy serialization
- `engine/batched.py` — _compute_gen_prompt_len
- `engine_core.py` — gen_prompt_len passthrough
- `mllm_scheduler.py` — CacheList truncation, attribution
- `mllm_batch_generator.py` — attribution
- `reasoning/mistral_parser.py` — new file, [THINK]/[/THINK] parser
- `reasoning/__init__.py` — register mistral parser

### TypeScript (panel/src/)
- `main/ipc/chat.ts` — const→let fix
- `main/ipc/models.ts` — HF size calculation (per-dtype bytes)
- `renderer/src/components/chat/ChatSettings.tsx` — reasoning effort UI

### Bundled Python patches (panel/bundled-python/)
- `mlx_vlm/models/mistral3/language.py` — _Mistral4VLMBackbone class
- `mlx_vlm/models/mistral3/config.py` — TextConfig rope_parameters extraction
- `mlx_vlm/models/pixtral/config.py` — VisionConfig rope_parameters extraction
- `mlx_lm/models/mistral4.py` — ModelArgs patches (super guard, rope, MLA, input_embeddings)

### Docs
- `notes/MASTER-CHECKLIST.md` — 654-item consolidated audit checklist
- `notes/AUDIT-CHECKLIST.md` — raw comprehensive checklist (1383 lines)
- `notes/todo-next-session.md` — next session TODOs
- `notes/session-2026-03-23-issues.md` — this session's issue tracker
- `notes/SESSION-2026-03-23-FINAL.md` — this file
- `docs/CACHE-INNOVATION-ROADMAP.md` — 6 revolutionary cache approaches
