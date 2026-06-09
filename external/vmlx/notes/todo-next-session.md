# TODO — Next Session(s)

## From This Session (2026-03-23)

### Bugs to Fix
1. **Reranker causal path TokenizerWrapper** — line 187 in `_score_causal()` has same `__call__` bug as PR #21 fixed for encoder path. Need unwrap on else branch.
2. **MLLM scheduler `_extract_cache_states` missing GQA head normalization** — LLM scheduler does this at lines 1238-1258. Add `_detect_n_kv_heads()` to MLLMScheduler.
3. **block_disk_store `_serialize_block` CacheList tag** — doesn't handle "cache_list" tag. Low practical impact (numpy path also skips) but should be correct. MoE models get no disk L2 for CacheList layers.
4. **Dead i18n file** — `panel/src/renderer/src/i18n/index.tsx` is dead code (TypeScript resolves `index.ts` first). Delete it.

### PR #22 (Jina Reranker v3) — Waiting on Contributor
Review comments posted. 4 items for Gattochoo to address:
- Weight shape interpretation (in_dim vs out_dim)
- Doc count vs embed token count assertion
- Context length guard for 100+ docs
- Zero-norm epsilon in cosine similarity

### Feature Requests (from GitHub Issues)
1. **Single port serve all models** (mlxstudio #25) — LM Studio-style UX. You said ~1 week. HIGH priority.
2. **Broader API support** (vmlx #23) — Ollama compat, LM Studio API, Deepgram, realtime. CobraSoftware offered to PR.
3. **Cluster support** (mlxstudio #26) — Multi-node MLX RDMA like exo. LONG TERM.

### Mistral 4 Remaining Checks
- [ ] Tool parser "mistral" works with native tool calling
- [ ] VLM image_token_index (10 = [IMG]) correct end-to-end
- [ ] Reasoning on/off toggle works (enable_thinking → reasoning_effort mapping)
- [ ] think_in_template=False doesn't break multi-turn with reasoning
- [ ] Appears in recommended/downloadable models list

### Mistral 4 VLM Testing & Status
- **Cannot test VLM via CLI** (`python3 -m vmlx_engine.server`) — system Python uses stock mlx_vlm without `_Mistral4VLMBackbone`
- **Must test via Electron app** which uses bundled Python (`-s` flag) with all our patches
- **Bundled Python verified to have ALL patches:**
  - `_Mistral4VLMBackbone`: True
  - VisionConfig `rope_parameters` extraction: True
  - TextConfig `rope_theta` extraction from rope_parameters: 10000.0
  - ModelArgs `rope_scaling` with `llama_4_scaling_beta`: True
  - Model loads: 36 Mistral4DecoderLayer, embed_q/unembed_out present, VisionModel present
- **`mlx_vlm.generate()` has bugs** in bundled version (audio None handling, image path splitting) — but the app doesn't use this function. The app uses MLLM scheduler's own image processing pipeline.
- **2-bit quantized vision encoder** produces poor image understanding (user confirmed with cat photo). This is a model quality limitation at 2-bit, not a code bug.
- **jang_config.json** fixed: `has_vision: true` (was `false`). Need to update jang-tools converter.

### JANG VL Fix
- `jang_config.json` had `has_vision: false` even though model has 218 vision weight tensors
- Fixed locally (set to `true`)
- Need to update jang-tools converter to auto-detect vision_config in config.json
- Our `is_mllm_model()` now falls through to config.json regardless (defense in depth)

### Audits Still Pending (from task list)
- Prefix caching after sleep/wake
- KV cache quantization after sleep/wake
- Continuous batching after sleep/wake
- Anthropic API streaming + stop button
- Coding tool integration (Claude Code, Codex, OpenCode)
- Download flow (pause, resume, no duplicates)
- Multiple sessions with JIT — no cross-contamination

### i18n Coverage
- Only ~5% of UI uses translations (TitleBar + About page)
- ~300+ hardcoded strings across 50+ components
- Locale files exist for all 5 languages with 176 keys each
- Many keys exist but aren't wired (`convert.*`, `tools.*`)
- LOW priority — cosmetic, not functional

## From Deep Audits

### 12 WARNs (all low-impact, documented)
1. bfloat16 for all MLA models (correct by design)
2. numpy block_slice skips CacheList (MLX path handles it)
3. `_ensure_batch_cache` checks ArraysCache not MambaCache directly
4. Reasoning ON but no think tags → fallback re-emit delay
5. q4 KV quant degrades long reasoning context on restore
6. Partial think tags across chunks → rare char leak
7. GPT-OSS emitted_reasoning shrink edge case
8. Stale `_n_kv_heads` on model switch (FIXED with clear() reset)
9. CacheList numpy path always "skip" (MLX path works)
10. MLA MLLM batch with head inflation (single-request OK)
11. `_resolve_model_path` dead code in PR #22
12. Thread safety in reranker `_load()` (no mutex)

### Community Bug Patterns (from closed issues)
- Metal kernel panic cluster: #5, #7, #11 — all caused by lazy MLX ops on background threads. Fixed with numpy round-trip.
- Cache reconstruction failures: #4, #8, #13 — hybrid SSM, QuantizedKVCache, block disk. All fixed.
- macOS version: #1, #20 — need macOS 15+ for Metal language v4. Add clearer README requirement.
