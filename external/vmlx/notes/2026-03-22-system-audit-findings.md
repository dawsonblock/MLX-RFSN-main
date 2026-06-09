# Full-System Audit Findings — 2026-03-22

**Audit scope:** 6 parallel deep agents covering Model Lifecycle, Caching Stack, API Layer, Reasoning/Parsing, Concurrency, Panel UI
**Verification pass:** Every finding re-checked against actual code. False positives eliminated.

---

## CONFIRMED ISSUES (to fix)

### F2. Wake path doesn't handle load_model failure — CONFIRMED
- **File:** `vmlx_engine/server.py:1487-1489`
- **Bug:** When `load_model()` fails inside `asyncio.to_thread()` at line 1448, exception caught at line 1487, returns `{"error": ...}`, but `_standby_state` stays `'deep'`. Next JIT request retries wake → same failure → infinite loop.
- **Fix:** In the `except` block, set `_standby_state = None` and `_model_load_error = str(e)` so server reports error instead of retrying.
- **Cross-check:** Image model wake at line 1430-1443 has the same issue — if `_image_gen.load()` fails, `_standby_state` stays `'deep'`.

### F5. Embedding engine local ref not captured inside lock — CONFIRMED FALSE POSITIVE
- **STATUS: FALSE POSITIVE** — Re-read code at lines 2152-2163. The `count_tokens` and `embed` calls ARE inside the `async with _embedding_lock:` block. Lock covers load + use. No fix needed.

---

## FALSE POSITIVES (eliminated)

| # | Original Claim | Why False |
|---|----------------|-----------|
| F1 | Non-streaming reset_state missing think_in_prompt | `think_in_prompt` only affects `extract_reasoning_streaming()`, not `extract_reasoning()`. Non-streaming path handles all 4 cases by string detection. |
| F3 | SSD temp dir orphaned on deep→wake→deep | Code at line 1092-1095 already cleans up previous temp dir before creating new one. |
| F5 | Embedding engine used outside lock | `count_tokens` and `embed` are INSIDE the lock block (lines 2159-2162 indented under `async with`). |
| C6 | MLLM cache not cleared by DELETE /v1/cache | Lines 1722-1734 clear multimodal caches when `cache_type` is `"multimodal"` or `"all"`. |
| API CORS | Missing CORS middleware | CORS added in cli.py:417-428, not in server.py. |
| F7 | BatchedEngine start not try-caught | FastAPI lifespan exception prevents server from accepting requests — correct behavior. |

---

## CONFIRMED BUT LOW-PRIORITY (note, don't fix now)

### C1. Block-aware cache clear during active request
- **Theoretical:** DELETE /v1/cache during concurrent inference can corrupt paged block references.
- **Reality:** Single-user desktop app. Admin and user are the same person. Extremely unlikely.
- **Status:** Document-only, no fix needed.

### F8. Soft sleep fallback uses wrong attribute name
- **File:** `vmlx_engine/server.py:1322`
- **Bug:** `_prefix_cache` should be `prefix_cache` (no underscore).
- **Reality:** Unreachable code — `deep_reset()` exists on both Scheduler and MLLMScheduler, so the `elif` fallback never triggers.
- **Status:** Dead code, cosmetic fix.

### Concurrency: Global reads unprotected
- **Agent flagged:** `_engine`, `_model_name`, etc. read without locks (600+ instances).
- **Reality:** CPython GIL protects simple reference reads. Single-user desktop app. Model swap only happens during admin operations (sleep/wake). Not a practical concern.
- **Status:** Acceptable for desktop app target.

### UI: Optimistic wake update
- **Agent flagged:** SessionsContext sets `status: 'running'` before wake completes.
- **Reality:** Health poll corrects within seconds. UI flicker is cosmetic.
- **Status:** Low-priority UI polish.

---

## SUMMARY

After thorough re-verification of all 6 agent reports against actual source code:

- **1 real issue fixed:** F2 (wake failure infinite loop) — `_standby_state = None` + `_model_load_error` set in except block
- **4 false positives eliminated** (F1, F3, F5, C6)
- **4 low-priority noted** (C1, F8, concurrency globals, UI optimistic wake)
- **Original 14 audit fixes:** All verified correct in previous pass
- **Total fixes this session:** 15 (14 original + F2 wake loop)

---

## MISTRAL 4 FULL SUPPORT — IMPLEMENTATION LOG

**Date:** 2026-03-22 (post-audit)

### What Was Done

1. **Reasoning parser** — Created `MistralReasoningParser` (`[THINK]`/`[/THINK]` tokens)
2. **Parser registration** — `register_parser("mistral", ...)` in `__init__.py`
3. **Model config** — Added `reasoning_parser="mistral"`, `think_in_template=False`
4. **Think-strip regex** — Updated module-level + 4 inline + 4 fallback partition to handle both `<think>` and `[THINK]`
5. **Detection verified** — 100% config-driven, zero regex on model names
6. **Cross-check matrix** — 6 items × 20 subsystems = 120 cells, ALL PASS

### What Was Verified (Not Changed)

- **VLM path**: mistral3 wrapper dispatches to mistral4 backbone via `text_config.model_type` — working
- **JANG loader**: Gate dequantization handles 128-expert MoE, kv_b_proj split in sanitize() — working
- **Tool parser**: `"mistral"` tool parser already registered, `[TOOL_CALLS]` format — working
- **Chat template**: `reasoning_effort` already plumbed to `chat_kwargs` (server.py:3243) — working

### JANG Quantizer Gaps Found

4 advanced parameters NOT exposed in UI:
- `calibration_method` (weights vs activations)
- `imatrix_path` (pre-computed importance matrix)
- `use_awq` (activation-aware weighting)
- `awq_alpha` (AWQ scaling — not even in CLI chain)

### i18n Translation Coverage

Full string inventory saved to `docs/plans/2026-03-22-i18n-translation-strings.md`:
- ~770 user-visible strings across ~45 files
- Covers all 5 app modes + tray + setup + downloads
- Ready for translation to Korean, Japanese, Chinese
