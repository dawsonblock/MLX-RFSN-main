# Fix Considerations — 2026-03-22

**Purpose:** Supplementary list of every consideration before implementing the 14 confirmed fixes. Each fix must be coherent with ALL connected subsystems.

---

## H3+H4: Audio STT/TTS Lock Fix

### Global state
- Add `_stt_lock: asyncio.Lock | None = None` and `_tts_lock: asyncio.Lock | None = None` after line 2777
- Must use lazy init (same as `_embedding_lock`, `_reranker_lock`) — NOT `asyncio.Lock()` at import time (binds to wrong event loop)
- Declare `global _stt_lock` inside endpoint function body

### Lock scope
- Lock covers: engine check + unload + create + load + local ref capture
- Lock does NOT cover: file I/O, transcription/generation (would serialize all audio requests to ~30s each)
- Use `local_stt = _stt_engine` capture inside lock, use `local_stt.transcribe()` outside
- Same pattern as reranker (`local_reranker = _reranker` inside lock)

### Correct assignment order (critical)
- Do NOT assign global before load succeeds
- Pattern: `new_engine = STTEngine(name)` → `new_engine.load()` → `_stt_engine = new_engine`
- If `load()` raises, `_stt_engine` stays pointing to previous working engine (or None)
- This differs from reranker which assigns before load — reranker has a bug here too (pre-existing)

### Unload before reassignment
- Call `_stt_engine.unload()` before creating new engine
- `STTEngine.unload()` (stt.py:142) is idempotent — sets `self.model = None`, `self._loaded = False`
- `TTSEngine.unload()` (tts.py:295) is also idempotent
- Safe to call when nothing loaded (guard: `if _stt_engine is not None`)
- Consider adding `gc.collect(); mx.clear_cache()` between unload and new load to free Metal memory immediately (otherwise peak = 2x model size during swap)

### JIT middleware interaction — NO conflict
- `/v1/audio/*` paths are in `is_inference` list (line 427) — JIT wake fires before endpoint
- Middleware holds `_wake_lock`, audio endpoint holds `_stt_lock` — different locks, no deadlock
- Middleware completes wake before endpoint runs (sequential via `call_next`)

### Sleep/wake lifecycle
- `admin_deep_sleep()` does NOT touch `_stt_engine` or `_tts_engine`
- Audio engines survive deep sleep — their GPU memory is NOT freed
- This is pre-existing, not introduced by this fix — but worth noting
- After deep wake, audio engines are still loaded (no reload needed)

### Panel timeout consideration
- STT timeout: 60s (audio.ts:73). Lock wait + model load could approach this for concurrent different-model requests
- TTS timeout: 120s (audio.ts:109). Sufficient for typical use
- No panel code changes needed for single-user case

### `/v1/audio/voices` — NOT affected
- Returns static constants, does not use `_tts_engine`

### Tests needed
- Concurrent STT different-model race (assert no crash, unload called)
- Concurrent TTS same-model (assert lock serializes, both succeed)
- Lock lazy-init (fresh module state, no crash)
- Load failure state (assert global not corrupted)

---

## M7: Speculative GPU Leak Fix

### unload_draft_model() behavior
- Sets `_draft_model = None`, `_draft_tokenizer = None`
- Sets `_spec_config.enabled = False` (mutates in-place — object itself survives)
- Calls `gc.collect()` + `mx.metal.clear_cache()`
- Safe when nothing loaded (guard: `if _draft_model is not None`)
- `_spec_config` is preserved with `.model` and `.num_tokens` intact, only `.enabled` flipped

### _cli_args expansion
- Add `'speculative_model': str | None` and `'num_draft_tokens': int` to `_cli_args` dict (server.py ~840)
- Values come from CLI args — `args.speculative_model` and `args.num_draft_tokens`
- Must be passed through `load_model()` signature or stored separately
- Recommended: add params to `load_model()` signature, save in `_cli_args`

### admin_deep_sleep insertion
- Insert AFTER `_engine = None` (line 1375), BEFORE `gc.collect()` (line 1377)
- Wrap in try/except (non-fatal — deep sleep must succeed even if spec unload fails)
- `from .speculative import unload_draft_model` inside the try block
- Import inside function avoids circular import risk

### admin_wake reload
- Insert AFTER `_engine.start()` / JIT (line 1457), BEFORE `_standby_state = None` (line 1458)
- Reconstruct `SpeculativeConfig` from `_cli_args` values (can't reuse `_spec_config` directly — `.enabled` is False)
- New `SpeculativeConfig(model=spec_model, num_tokens=n)` — `__post_init__` sets `enabled=True`
- Must use `asyncio.to_thread(load_draft_model, spec_config)` — blocking call
- Wrap in try/except (non-fatal — main model should work without spec decoding)

### Soft sleep — do NOT unload draft model
- Soft sleep keeps main model loaded for instant wake
- Draft model is small — unloading gains little memory
- Reload on soft wake would require same `asyncio.to_thread` treatment (slow, defeats purpose)
- Recommendation: leave draft model in memory during soft sleep

### /health endpoint impact
- `get_spec_stats()` will return `enabled: False, draft_model_loaded: False` during deep sleep
- This is MORE accurate than current behavior (shows `enabled: True, loaded: True` while main model is unloaded)
- Health-monitoring clients see state flip — correct behavior

### SimpleEngine wiring — NO explicit wiring needed
- SimpleEngine calls `speculative.get_draft_model()` per-request (module-level global lookup)
- After wake reloads `_draft_model`, next request picks it up automatically
- No engine attribute to set

### Edge cases
- No draft model configured: `_cli_args.get('speculative_model')` returns None → skip reload (no-op)
- Draft model path deleted between sleep/wake: `load_draft_model()` raises → caught, logged as warning
- SSD mode: `speculative_model` already forced to None by cli.py:226 → stored as None in `_cli_args`
- Deep sleep → deep sleep: returns 409 before reaching unload code (line 1349–1350)
- Soft sleep → deep sleep: allowed, draft model exists, unload proceeds normally

### Tests needed
- `test_deep_sleep_unloads_draft`: patch `unload_draft_model`, assert called
- `test_deep_wake_reloads_draft`: set `_cli_args`, patch `load_draft_model`, assert called with correct config
- `test_no_spec_configured`: assert deep sleep + wake don't crash when no draft model

---

## M8: Cache Clear L2 Block Disk Store Fix

### CRITICAL DESIGN DECISION: Targeted fix in server.py, NOT in PagedCacheManager.clear()

Adding `_disk_store.clear()` to `PagedCacheManager.clear()` would affect ALL callers:
- `DELETE /v1/cache` — correct, should wipe disk
- Soft sleep (`scheduler.deep_reset()`) — WRONG, disk should survive for warm restart
- Deep sleep (`scheduler.deep_reset()`) — WRONG, same reason
- Model reload (`load_model()`) — WRONG, disk cache may be valid for same model

**Fix goes in server.py `clear_cache()` handler ONLY** — lines ~1688-1690.

### Exact code addition (after existing disk_cache.clear())
```python
paged_mgr = getattr(scheduler, 'paged_cache_manager', None)
if paged_mgr is not None:
    disk_store = getattr(paged_mgr, '_disk_store', None)
    if disk_store is not None:
        disk_store.clear()
```

### BlockDiskStore.clear() thread safety
- Has a TOCTOU race: drains write queue, but background writer may already have dequeued items
- Writer commits those items post-clear (file rename + SQLite INSERT after rmtree + DELETE)
- Race consequences: stale SQLite entries for non-existent files → self-healing on next read (cleanup path removes stale entries)
- NOT a crash risk — recoverable within one poll cycle (~200ms)

### SQLite concurrency
- clear() opens its own connection (WAL mode)
- Background writer has its own connection
- WAL serializes concurrent writes (timeout=5.0)
- `_read_conn` not reconnected but WAL ensures it sees committed deletes

### `_disk_store` may be None
- Conditionally set during init — only when `enable_block_disk_cache=True` and init succeeds
- ALL access must guard: `if disk_store is not None`

### Disk reclamation timing
- `shutil.rmtree(blocks_dir)` is synchronous — immediate reclamation
- `blocks_dir.mkdir(parents=True)` recreates empty directory
- Background writer's `shard_dir.mkdir(exist_ok=True)` will succeed on recreated directory

### Tests needed
- `test_block_disk_store_clear`: write blocks, clear, verify empty
- `test_delete_cache_endpoint_clears_disk`: integration test
- `test_soft_sleep_preserves_disk`: regression guard — deep_reset does NOT wipe disk

---

## M9+M10: Loading Progress Regex Fixes

### Pattern 9 (JANG loaded) — currently only matches v2 non-VLM
- Current: `/JANG v[12] loaded in/`
- v1 non-VLM: `"JANG v1 model loaded in ..."` — word "model" between v1 and loaded
- v2 VLM: `"JANG v2 VLM loaded in ..."` — word "VLM" between v2 and loaded
- v1 VLM: `"JANG v1 VLM loaded in ..."` — word "VLM" between v1 and loaded
- Fix: `/JANG v[12].{0,10}loaded in/` — allows up to 10 chars between version and "loaded"
- False-match risk: zero — only JANG completion messages contain this pattern

### Pattern 5 (Loading JANG VL) — dead pattern
- Current: `/Loading JANG VL model/`
- Actual v1 VLM log: `"Loading JANG v1 VLM: ..."` at jang_loader.py:615
- Fix: `/Loading JANG v1 VLM:/`

### Pattern 14 (model loaded mode) — two sub-issues
- Batched: uppercase `"Model loaded (batched mode)"` vs lowercase pattern
- Simple: prefix `"LLM model loaded (simple mode)"` — pattern doesn't account for prefix
- Fix: `/\bmodel loaded \((?:simple|batched) mode\)/i` — case-insensitive, word boundary allows prefix

### Dead patterns worth noting (not in fix scope)
- Pattern 6 `/Loading MLLM:/` — no engine log emits this string
- Pattern 10 `/Model loaded successfully/` — no engine log emits this string
- Pattern 16 `/Saved \d+\/\d+ layer weights to SSD/` — log string may have changed

### Progress percentage integrity after fixes
- Sequence: 5→10→15→20→30→50→55→60→70→72→73→74→75→80→85→90→95
- All strictly ascending, no gaps or overlaps introduced
- 100% only reached by health poll, not log scraping (by design)

### False-match prevention
- All regexes match unique strings that only appear in model loading context
- No debug-level messages share these strings
- Adding `/i` flag to pattern 14 cannot match unrelated lines

---

## M11: Speculative Draft UI Disabled for SSD

### Changes needed
- Draft model `<input>` at line 759: add `disabled={ssdActive}`
- numDraftTokens `<SliderField>` inside `{config.speculativeModel && ...}` block: add `disabled={ssdActive}`
- Keep existing `IncompatWarning` at line 754 — explains WHY field is disabled
- Keep `PerformanceHint` at line 755 gated by `{!ssdActive && ...}` (already correct)

### Consistency with other gated fields
- Continuous Batching checkbox: has `disabled={ssdActive}` ✓
- Prefix Cache checkbox: has `disabled={ssdActive}` ✓
- Speculative section: missing `disabled={ssdActive}` ✗ ← this fix

---

## L21: InfoNote SSD Guard

### Two InfoNotes need fix
- Line 281-283: "batching will auto-enable" — add `!ssdActive &&`
- Line 284-286: "Turning this off disables..." — add `!ssdActive &&`

### Why both: when SSD is active, continuousBatching checkbox is disabled (user can't toggle it). Both notes reference actions the user cannot take ("enable it"). The SSD IncompatWarning at line 229 already explains the SSD override.

### Other InfoNotes/Warnings checked — already correct
- Line 329: already has `{!ssdActive && ...}`
- Line 546: already has `{!ssdActive && ...}`
- Line 755-756: speculative section, already `{!ssdActive && ...}`

---

## L12: Think-Strip Regex Module-Level

### Exact change
- Add `_THINK_STRIP_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)` at module level (~line 155)
- Delete local `_think_strip_re = re.compile(...)` at lines 3128 and 3668
- Replace `_think_strip_re.sub(...)` with `_THINK_STRIP_RE.sub(...)` at lines 3133 and 3673
- Zero risk, zero behavioral change

---

## L13: Harmony Non-Streaming Fix

### Real behavioral bug — Harmony + non-streaming
- `GptOssReasoningParser.extract_reasoning()` checks `self._harmony_active` (gptoss_parser.py:122)
- Fresh clone via `__class__()` has `_harmony_active = False` — Harmony parsing branch never taken
- Fix: compute `_harmony_active` from `chat_kwargs["prompt_suffix"]` and call `reset_state()`
- For non-Harmony parsers (ThinkParser etc.), `reset_state()` is safe no-op or sets defaults

### What's available at line 3335
- `chat_kwargs` dict — contains `prompt_suffix` if Harmony prefix was injected
- Can detect Harmony via: `"prompt_suffix" in chat_kwargs and chat_kwargs["prompt_suffix"].startswith("<|start|>assistant<|channel|>")`
- Same detection logic as streaming path (server.py:4240)

---

## L17: Reranker model_path

### One-line fix
- Line 2256: `_reranker.model_path` → `local_reranker.model_path`
- `local_reranker` is in scope (captured at line 2232)
- Zero risk

---

## L18: MCP Execute Rate Limit

### One-line fix
- Line 2750: add `Depends(check_rate_limit)` to dependencies list
- FastAPI injects Request automatically
- No MCP-internal rate limiting exists that this would conflict with

---

## L19: Responses Usage Guard

### Pattern choice
- Use `getattr(output, 'attr', 0)` — consistent with `cached_tokens` on the same line
- Assign to local vars to avoid double-getattr: `_pt = getattr(output, 'prompt_tokens', 0)`
- Zero behavioral change for normal outputs
