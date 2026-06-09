# MiMo V2.5 JANG_2L Runtime Status - 2026-06-06

## Scope

Active repo: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

Do not use `/Users/eric/vmlx`, ADLab, TB/RDMA workflows, or Swift-only paths for this work. This status is for the vMLX Python engine and MLXStudio-facing runtime behavior.

## Current verdict

MiMo V2.5 JANG_2L is not release-green.

Working pieces are real and live-proven, including repeated text cache,
multiturn recall, required-tool parsing after prior turns, tool-result
continuation, strict JSON, exact code/whitespace, and the 64-word long-prefix
cache row. Remaining red items are speed, broader auto-tool/loop-stop/raw-markup
matrix, media/VL/audio/video, UI parity, and release
packaging/signing/notarization.

## Live proofs completed

### Text and cache path

Latest full smoke artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep-native-tools-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Passing checks:

- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: `prompt_tokens_details.cached_tokens=67`.
- `text_cache_repeat_2`: `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- L2 block disk cache wrote 4 blocks / 141 tokens in the smoke.
- Native cache metadata reports `mixed_swa_kv_v1`, full plus rotating/sliding-window layout preserved.

### TurboQuant/KV cache split

Diagnostic artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-diagnostic-no-tq-kv-20260606`

Finding:

- Generic TurboQuant KV corrupted MiMo mixed full/SWA rotating-cache semantics.
- Disabling generic TQ-KV made the repeated cached text path pass.
- Runtime fix now skips flat generic TurboQuant KV when native cache contains `RotatingKVCache` / `BatchRotatingKVCache`.
- This is not a fake guard: native rotating cache remains active, and storage-boundary q4 cache quantization remains separate from generic TQ-KV.

### Pressure guard split

Finding:

- The old 98% Metal working-set guard rejected valid follow-up requests after cache growth.
- Runtime now clears reclaimable MLX cache before rejecting and evaluates non-cache pressure.
- Default guard threshold moved to 99%.
- This let MiMo progress past text cache and multiturn in live smoke.

## Remaining red blockers

### MIMO-TOOL-001: required tool calling corrupts output

Earlier failing request:

`Use the record_fact tool exactly once with value blue-cat. Do not answer in visible text.`

Expected tool:

`record_fact({"value":"blue-cat"})`

Observed with batched MLLM, paged cache, L2 block cache, native tools preserved:

- HTTP 400 because `tool_choice=required` produced no parsed tool calls.
- Generation emitted 96 tokens at about 1.6 tok/s.
- Raw preview: `<tool_call>` followed by punctuation/Chinese-comma garbage, not `<function=record_fact>`.

Observed with `--kv-cache-quantization none`:

- Same failure.
- Same raw preview shape.
- Rules out q4 KV quantization as the root cause.

Observed with SimpleEngine, `--no-continuous-batching`, `--disable-prefix-cache`, `--kv-cache-quantization none`:

- Same failure class.
- Raw preview is punctuation garbage.
- Rules out batched scheduler, paged prefix cache, L2 disk cache, and KV quantization as the sole cause.

Observed with thinking enabled in SimpleEngine:

- Still HTTP 400.
- Generated 97 tokens, but no visible parse text after reasoning extraction.
- Does not fix tool calling.

Current classification:

- This was red before the ChatML fallback and keep=0 rotating-cache fixes.
- The current narrow no-media source smoke now parses the required tool call
  correctly as `record_fact({"value":"blue-cat"})`.
- This does not clear the full tool matrix yet. Tool-result continuation,
  auto-tool behavior, loop-stop behavior, multi-turn tool memory, and raw XML
  leakage checks still need full E2E proof.

### MIMO-SPEED-001: speed far below target

Observed live smoke speeds:

- First short text request: about 0.1 tok/s due large first prefill/load path.
- Cached repeat: about 1.1 tok/s.
- Multiturn: about 1.2 tok/s.
- Required tool failure path: about 1.6 tok/s.

Target:

- 40+ tok/s expected by Eric.

Current classification:

- Red. Need separate profiling after correct artifact/runtime path is established.
- Do not claim release readiness or sign/notarize while this remains red.

### MIMO-VL-001: MiMo vision/audio/video runtime unwired

Observed capabilities report:

- Runtime modalities: `text`.
- Declared/preserved modalities: `vision`, `image`, `video`, `audio`.
- Status: preserved but unwired for image/video/audio.

Current classification:

- Red for VL/audio/video release.
- Text-only smoke cannot clear VL/audio/video.

## Engine changes made in this slice

Files changed:

- `vmlx_engine/mllm_batch_generator.py`
- `vmlx_engine/utils/tokenizer.py`
- `vmlx_engine/utils/jang_loader.py`
- `vmlx_engine/server.py`
- `vmlx_engine/prefix_cache.py`
- `vmlx_engine/api/tool_calling.py`
- `tests/test_engine_audit.py`

Implemented behavior:

- MiMo cached text fast path passes absolute position ids and seeds decode position state.
- Generic TurboQuant KV skips native rotating/full attention cache layouts.
- JANG TurboQuant KV patcher skips native rotating/full attention cache layouts.
- Metal working-set guard clears reclaimable cache and checks non-cache pressure before rejecting.
- Paged tensor-backed prefix cache no longer keeps full request cache references after L2 write-through.
- XML function fallback requires concrete `<function=name>` examples, not generic `example_function_name` only.
- XML function fallback injects into first user turn and preserves native `tools` during re-render.
- Required-tool failures now log a bounded raw preview for diagnosis.
- MiMo XML-function fallback instructions stay inside the rendered ChatML
  system turn when MiMo's template drops synthetic fallback messages; vMLX no
  longer prefixes those instructions before `<|im_start|>system`.

Focused source tests:

`tests/test_engine_audit.py -k "native_rotating_cache or metal_ws_pressure_guard_reclaims_cache_before_reject or paged_cache_tensor_store_drops_full_request_cache_reference or mimo_v2_cached_text_fast_path_uses_absolute_positions or xml_function_tool_fallback_requires_concrete_function_examples"`

Latest result:

- 6 passed, 515 deselected.

Additional focused parser/template proof after ChatML fallback fix:

`tests/test_tool_fallback_injection.py::test_mimo_xml_function_direct_fallback_stays_inside_chatml_system`

Latest result:

- 1 passed.

Full focused parser/fallback slice:

`tests/test_tool_fallback_injection.py::test_mimo_xml_function_fallback_matches_parser_dialect tests/test_tool_fallback_injection.py::test_mimo_xml_function_direct_fallback_stays_inside_chatml_system tests/test_xml_function_tool_parser.py`

Latest result:

- 12 passed.

## Next required checks

1. Obtain a corrected/current MiMo V2.5 JANG_2L artifact without using ADLab work as implementation guidance.
2. Prefer HTTP/local network transfer only for the model artifact itself; do not pull implementation guidance from ADLab/TB/RDMA notes.
3. Compare artifact metadata and hashes against the installed `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` copy.
4. Run the same direct XML continuation probe and required-tool SimpleEngine test on the replacement/current artifact.
5. If replacement works, classify the installed local model as corrupt/bad, enumerate exact old MiMo paths, then replace/delete the old MiMo copies.
6. If replacement fails identically, classify this as model/tool-template/decode compatibility and continue in vMLX Python engine only with legitimate fixes such as constrained/guided XML decoding.
7. Do not synthesize fake tool calls from `tool_choice=required` or user text as a release fix.
8. Only after required tools pass, run full smoke with paged cache, L2 block cache, native rotating cache, and no generic TQ-KV.
9. Only after text/tools pass, implement/prove MiMo image/video/audio runtime. Current MiMo media is preserved but unwired.
10. Only after all relevant families pass E2E should release/sign/notarize proceed.

## Release gate

Do not release, sign, notarize, or call MiMo/MLXStudio/vMLX green yet.

## 2026-06-07 decode speed gate row and partial-evidence crash proof

Source/harness changes:

- `tests/cross_matrix/run_decode_speed_gate.py` now includes a
  `mimo_v25_jang2l` row for the normal local artifact:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- The row uses the real MiMo route: `--is-mllm`, `xml_function` tool parser,
  `think_xml` reasoning parser, paged cache, block disk L2, and q4
  storage-boundary KV.
- The row enforces the intended release floor with `expected_min_tps=40.0`.
- The row uses a smaller decode batch (`--completion-batch-size 64`) and
  reduced PP targets (`512`, `1024`) so the speed harness is not accidentally
  testing the generic 512-token decode batch shape for this tight-memory model.
- The harness now preserves partial live evidence on later server death:
  `health_before`, warm usage, coherency result, completed PP rows, bundle row,
  greedy row, and log tail.

Focused validation:

- `.venv/bin/python -B -m py_compile tests/cross_matrix/run_decode_speed_gate.py tests/test_current_regression_suite.py`
- `.venv/bin/python -m pytest -q tests/test_current_regression_suite.py -k 'decode_speed_gate'`
- Result: `3 passed`.

Live artifact:

`build/current-decode-speed-live-mimo-v25-jang2l-source-partial-evidence-20260607.json`

Result:

- `status=error`
- Error: `RemoteDisconnected('Remote end closed connection without response')`
- Server log tail ends with Metal OOM during the first PP request:
  `[METAL] Command buffer execution failed: Insufficient Memory`
- Warm request completed: `4` completion tokens, server log `0.2 tok/s`.
- Deterministic coherency request completed:
  `READY\n17+28=45\nCERULEAN`
- Coherency speed: `16` completion tokens in `10.12s`, `1.58 tok/s`.
- No PP rows completed; the first PP probe crashed before returning.

Runtime/cache facts from `/health` before the crash:

- Native cache family: `mimo_v2`.
- Schema: `mixed_swa_kv_v1`.
- Cache subtype: `mimo_v2_asymmetric_swa`.
- Components: full-attention KV, sliding-window KV, rotating-window metadata.
- Generic flat TurboQuant KV: disabled, reason `not_active`.
- Storage quantization: enabled, q4, applies to full and sliding attention KV,
  with rotating-window metadata preserved.
- Prefix cache, paged cache, and block disk L2: enabled.
- Existing block disk cache totals showed `37` tokens on disk from previous MiMo
  cache activity, but this speed run did not reach a cache-reuse PP success.

Classification:

- This is a runtime/source blocker, not a release clearance.
- MiMo speed is still far below the `40 tok/s` target.
- MiMo largest-context/cache speed proof is red because the PP probe crashes the
  server under Metal OOM before any PP row completes.
- The current issue is not generic flat TurboQuant KV being enabled; the runtime
  correctly disables generic TQ-KV for MiMo mixed full/SWA rotating cache.
- Do not delete or replace MiMo artifacts based on this result alone. This proves
  current vMLX runtime still cannot make the installed artifact release-green.

## 2026-06-07 no-media source smoke after tight-memory tool prefill chunking

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-tight-tool-prefill-chunk-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Runtime change:

- Text-only MLLM prefill now uses a smaller chunked prefix path under detected
  tight Metal working-set margin, including shorter fallback-tool-schema prompts
  that were previously below the long-prompt chunking threshold.
- This is a real runtime prefill/chunking change, not a media guard, request
  rejection, or synthesized tool-call fallback.

Summary:

- Runner summary: `status=pass`, `row_count=1`, `failed=0`.
- Model path: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Served as MLLM through current source `vmlx_engine.cli serve`.
- Prior broad sequence failure was a native Metal OOM on the final
  `tool_required` request after text/cache/multiturn/reasoning rows.
- Current run completed all 5 requests and server health remained live.

Request results:

- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=67`,
  `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, no harness validation failure, but output quality
  remains poor/repetitive and does not clear reasoning quality.
- `tool_required`: HTTP 200, no visible text, parsed real tool call
  `record_fact({"value":"blue-cat"})`.

Cache and memory evidence:

- Server log shows tight-memory allocator drains before prefills and after batch
  finish instead of a native Metal process death.
- Server log shows mixed-SWA VLM cache layout detected from extracted
  `RotatingKVCache` layers and clean prompt-boundary store.
- Final cache stats: paged cache hit tokens `67`.
- Block-disk L2 wrote `12` blocks / `583` tokens.
- Native cache reports `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.
- Generic TurboQuant KV remains disabled; storage-boundary q4 KV remains active
  for full and sliding attention KV with rotating metadata preserved.

Remaining red items from this run:

- Speed remains far below target: aggregate generation `112` tokens in
  `63.85s`, about `1.75 tok/s`.
- `reasoning_on` output is low quality/repetitive.
- This run is no-media only and does not prove image/video/audio.
- This run does not prove tool-result continuation, auto-tool loop stopping,
  Anthropic/Ollama/Responses parity, UI settings parity, installed packaged app,
  or release notarization.

## 2026-06-07 tool-result continuation after template argument normalization

Artifacts:

- Failing pre-fix gate:
  `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-continuation-nomedia-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`
- Passing post-fix gate:
  `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-continuation-after-template-args-fix-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Runtime bug found:

- OpenAI Chat Completions assistant history stores
  `tool_calls[].function.arguments` as a JSON string.
- MiMo's text-only MLLM processor-template path bypassed the generic tokenizer
  normalization that converts those strings to dicts.
- The MiMo Jinja template then attempted `arguments|items` and failed before
  generation with:
  `TypeError: Can only get item pairs from a mapping.`

Runtime fix:

- `BatchedEngine._mimo_text_only_chat_template()` now normalizes a shallow copy
  of assistant tool-call arguments before processor template rendering.
- The request object is not mutated.
- This is a real API/tool-history compatibility fix, not prompt-only behavior
  forcing or synthetic tool-call fabrication.

Expanded harness proof:

- `bench/all_local_model_smoke.py --include-tools` now adds
  `tool_result_continuation` after `tool_required`.
- The continuation row supplies an assistant tool call, a matching `role=tool`
  result, then requires exact final visible text `STORED blue-cat`.
- Validation fails on a second tool call, raw tool markup, empty visible text,
  or non-exact final synthesis.

Post-fix live result:

- Runner summary: `status=pass`, `row_count=1`, `failed=0`.
- Six requests completed: text cache repeat, cache hit repeat, multiturn recall,
  reasoning-on, required tool, and tool-result continuation.
- `tool_required`: real parsed tool call
  `record_fact({"value":"blue-cat"})`, no visible text.
- `tool_result_continuation`: HTTP 200, visible `STORED blue-cat`,
  `tool_calls=[]`, no reasoning text, no raw markup.
- Paged cache hit remains `cached_tokens=67`.
- Block-disk L2 wrote `14` blocks / `690` tokens.
- Native cache remains `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`; generic TurboQuant KV remains disabled.

Remaining red items from this run:

- Speed remains far below target: aggregate generation `117` tokens in
  `66.33s`, about `1.76 tok/s`.
- Reasoning-on output remains low quality/repetitive despite no validation
  failure in this narrow row.
- No-media only: image/video/audio are still unwired.
- This does not prove auto-tool behavior, adversarial loop-stop, Responses /
  Anthropic / Ollama parity, streaming, installed-app UI, or release packaging.

## 2026-06-07 strict JSON and exact code/whitespace harness rows

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-json-code-tools-nomedia-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Harness change:

- `bench/all_local_model_smoke.py` now always includes:
  - `structured_json_exact`: output must parse with `json.loads` and equal
    `{"status":"ok","value":"blue-cat","count":3}`.
  - `exact_code_whitespace`: output must exactly equal:
    `def add(a, b):\n    return a + b\nprint(add(2, 3))`.
- These checks fail on markdown fences, trailing prose, malformed JSON,
  type/value drift, lost indentation, changed punctuation, or extra whitespace.

Live MiMo source result:

- Runner summary: `status=pass`, `row_count=1`, `failed=0`.
- Eight requests completed: text cache repeat, cache hit repeat, multiturn
  recall, reasoning-on, required tool, tool-result continuation, strict JSON,
  and exact code/whitespace.
- `structured_json_exact`: HTTP 200, visible
  `{"status":"ok","value":"blue-cat","count":3}`, parsed exactly.
- `exact_code_whitespace`: HTTP 200, visible code matched exactly with
  indentation and newlines preserved.
- Paged cache hit remains `cached_tokens=67`.
- Block-disk L2 wrote `16` blocks / `797` tokens.
- Native cache remains `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.

Remaining red items from this run:

- Speed remains far below target: aggregate generation `153` tokens in
  `87.09s`, about `1.76 tok/s`.
- Reasoning-on output remains low quality/repetitive despite no validation
  failure in this narrow row.
- No-media only: image/video/audio are still unwired.
- This does not prove JSON schema constrained decoding, JSON repair, XML
  structured output, Responses / Anthropic / Ollama parity, streaming,
  installed-app UI, or release packaging.

## 2026-06-07 no-media source smoke after keep=0 cache fix

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep0-cache-fix-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Summary:

- Runner summary: `status=pass`, `row_count=1`, `failed=0`.
- Model path: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Served as MLLM through current source `vmlx_engine.cli serve`.
- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Server log confirms `Installed vMLX MiMo-V2 keep=0 rotating-cache patch`.
- Runtime cache family reports `mimo_v2` / `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`.
- Generic TurboQuant KV remains disabled for MiMo; native storage-boundary q4 KV is active for full and sliding attention KV while preserving rotating-window metadata.
- Prefix cache, paged cache, and block-disk L2 are enabled.

Request results:

- `text_cache_repeat_1`: HTTP 200, visible `ACK`, no validation failures.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `prompt_tokens_details.cached_tokens=67`, `cache_detail=paged`, no validation failures.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`, no validation failures.
- `reasoning_on`: HTTP 200, no runner validation failure, but visible output is messy/repetitive and is not a release-quality reasoning proof.
- `tool_required`: HTTP 200, parsed OpenAI tool call `record_fact` with arguments `{"value": "blue-cat"}`, empty visible content, no validation failures.

## 2026-06-07 long-prefix MiMo MLLM cache crash fix

Source changes:

- `vmlx_engine/mllm_batch_generator.py` now performs a tight-memory MLLM
  allocator drain around prefills and after batch finish when startup detects
  that model residency leaves only a small Metal working-set margin.
- `vmlx_engine/mllm_scheduler.py` now detects mixed-SWA VLM cache layout from
  the live extracted cache when wrappers hide `cache_subtype`; seeing
  `RotatingKVCache` forces the clean prompt-boundary store path.
- `tests/test_mllm_scheduler_cache.py` pins both behaviors.

Why this was a runtime issue:

- The previous 64-word MiMo long-context cache row crashed the server on the
  second request with native Metal OOM before a Python error could be returned.
- The model could answer the prompt; the failure was allocator/cache lifecycle
  plus runtime cache-layout detection.
- The runtime was treating MiMo's extracted `RotatingKVCache` layout like a
  generic cache in cleanup, so no paged/L2 store was produced for repeat hits.

Live proof:

`build/current-local-long-context-cache-mimo-v25-installed-64w-after-tight-memory-rotating-store-20260607`

Result:

- `status=pass`.
- First turn: HTTP 200, visible `LONGCTX-OK`.
- Runtime detected mixed-SWA from extracted `RotatingKVCache`.
- Clean prompt-boundary store wrote 7 block-disk blocks / 435 tokens.
- Second turn: HTTP 200, visible `LONGCTX-OK`.
- Second turn reported `prompt_tokens_details.cached_tokens=435`.
- Second turn reported `cache_detail=paged`.
- No native Metal OOM or server disconnect.

Smaller confirmation:

`build/current-local-long-context-cache-mimo-v25-installed-16w-rotating-detect-store-20260607`

- `status=pass`.
- `second_cached_tokens=147`.
- `cache_detail=paged`.
- Block-disk L2 wrote 3 blocks / 147 tokens.

Remaining MiMo blockers after this fix:

- Speed remains far below the 40+ tok/s target. The 64-word proof still shows
  about `0.2 tok/s` first turn and about `1.4 tok/s` cached second turn.
- Full required/auto/tool-result/multiturn tool matrix is not yet cleared.
- VL/audio/video runtime remains preserved but unwired.
- UI settings/process parity and installed-app release proof remain open.

Cache/L2 proof:

- Final scheduler cache hit tokens: `67`.
- Final cache hits: `4`.
- Block-disk L2 writes: `5`.
- L2 block tokens on disk: `173`.
- Last cache execution: `cache_detail=paged`, `cached_tokens=67`, `reconstructed=true`, `dequantized=true`.

Speed proof:

- Prompt throughput in final health: about `24.76 prompt tok/s`.
- Generation throughput in final health: about `1.76 generation tok/s`.
- Server request logs show `0.1 tok/s`, `1.1 tok/s`, `1.2 tok/s`, `1.7 tok/s`, and `1.5 tok/s` across the smoke.
- This remains far below the `40+ tok/s` target and is a release blocker.

Current classification:

- The previous narrow MiMo required-tool corruption is fixed for this exact no-media current-source smoke.
- This is not a full tool-system clearance: tool-result continuation, auto-tool behavior, loop-stop, long-context tool prompts, streaming tool parity, Anthropic/Ollama parity, and UI-launched tool settings remain unproven.
- This is not a media clearance: image, audio, video, media-expanded cache keys, and post-media recovery remain unbuilt/unproven for MiMo.
- This is not a release clearance: speed, reasoning quality, full E2E UI/settings parity, largest-context cache, restart/L2 restore, cancellation cleanup, and packaged-app parity remain open.

## 2026-06-07 rejected MiMo compiled-MoE speed attempt

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-compiled-moe-decode-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

What was tried:

- A vMLX registration-layer patch compiled the whole single-token MiMo MoE decode block, not only the router.
- It kept the same routing, expert weights, cache behavior, and tool parsing semantics.
- Server log confirmed the attempted patch activated: `Installed vMLX MiMo-V2 compiled single-token MoE decode patch`.

Result:

- Correctness stayed green: runner `status=pass`, `failed=0`, and the required tool call still parsed as `record_fact({"value": "blue-cat"})`.
- Speed did not improve: generation throughput was `1.7457 tok/s`, versus `1.7586 tok/s` before the attempt.
- Request timings were effectively unchanged.

Decision:

- The compiled-MoE patch was removed and must not be treated as a speed fix.
- MiMo's speed blocker likely requires a real optimized routed-expert/gather-qmm path or a different artifact/runtime class, not simply wrapping the generic `SwitchGLU` call in `mx.compile`.

## 2026-06-07 restart/L2 restore proof

Runner:

`bench/local_restart_l2_gate.py`

Artifact:

`build/current-local-restart-l2-mimo-v25-installed-cache-status-weighted-only-20260607/summary.json`

Result:

- `MiMo-V2.5-JANG_2L`: cache restore pass.
- Output exactness remains review/red.
- Auxiliary `audio_tokenizer` was incorrectly discovered by an earlier runner
  version; the runner now skips auxiliary weighted component directories.

Cache evidence:

- First process wrote block-disk L2 for the fixed prompt.
- Second fresh server process used the same block-disk cache directory.
- Second process reported `cached_tokens=43`.
- Second process reported `cache_detail=paged+disk`.
- Second process reported block-disk hits `1`.
- `last_cache_execution.disk_hit=true`.
- Native cache reports `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.
- Generic TurboQuant KV remains disabled; storage-boundary q4 KV is active for
  full and sliding attention KV with rotating metadata preserved.

Runtime evidence:

- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Server log confirms `Installed vMLX MiMo-V2 keep=0 rotating-cache patch`.

Remaining blockers:

- L2 restore reconstruction took about `19.43s`; this is functionally correct
  but not performance-cleared.
- The first process produced Chinese explanatory text instead of exact `ACK`.
- The second process produced punctuation instead of exact `ACK`.
- Generation remains far below the target.
- Largest-context, UI-launched restore, media-salted restore, cancellation,
  sleep/wake, unload/reload, and full API parity remain unproven.

## 2026-06-07 long-context cache probe

Runner:

`bench/local_long_context_cache_gate.py`

Artifact:

`build/current-local-long-context-cache-mimo-v25-installed-64w-after-mixed-swa-prefill-20260607/summary.json`

Result:

- MiMo selected as one model row after auxiliary sidecar filtering.
- First request completed with HTTP 200.
- First request took `28.17s` for `5` completion tokens, about `0.2 tok/s`.
- Second request crashed the server before returning a response.

Failure:

```text
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Insufficient Memory
```

Classification:

- MiMo long-context/cache reuse remains red.
- This failure is on the MLLM scheduler path and is separate from the text
  scheduler mixed-SWA clean-prefill fix that repaired Step long-prefix cache
  storage.
- Current MiMo can pass narrow no-media tools/cache/restart rows, but long
  prompt reuse still runs into speed, output, and memory failures.

Current release status: red.

## 2026-06-07 required-tool metadata propagation and decode proof

Source changes under test:

- `vmlx_engine/engine/batched.py` now preserves tool metadata from chat render
  into MLLM scheduler requests.
- `vmlx_engine/server.py` now forwards `tool_choice` into chat kwargs for Chat
  Completions and Responses routes.
- `vmlx_engine/mllm_batch_generator.py` now keeps small request-policy metadata
  after media prefill cleanup instead of clearing all `extra_kwargs`.
- `vmlx_engine/mllm_batch_generator.py` now has a MiMo-only
  `tool_choice=required` XML prefix logits processor for the structural prefix:
  `<tool_call>\n<function=record_fact>\n<parameter=value>`.
- The prefix processor does not infer user arguments or synthesize final tool
  calls. It only constrains the XML opening structure, then releases logits to
  the model.

Focused source proof:

- `py_compile` passed for `vmlx_engine/mllm_batch_generator.py`,
  `vmlx_engine/engine/batched.py`, `vmlx_engine/server.py`, and
  `tests/test_mllm_continuous_batching.py`.
- Focused tests passed: `3 passed, 39 deselected` for
  `mimo_required_tool_sampler`, `mimo_auto_tool_sampler`, and
  `prefill_keeps_extra_kwargs`.

Live artifact:

`build/current-mimo-v25-required-tool-prefix-toolonly-pending-token-20260607`

Live positives:

- The request reached the decode-time sampler with
  `_vmlx_template_tools`, `_vmlx_tool_choice`, and `_vmlx_tools_present`.
- The MiMo required XML prefix constraint activated.
- Raw output no longer stopped immediately after bare `<tool_call>`.
- Raw preview began with the intended structural prefix:
  `<tool_call>\n<function=record_fact>\n<parameter=value>`.

Live failure:

- HTTP 400 remained because no parsed tool call was produced.
- After the structural prefix, the model generated punctuation/fullwidth comma
  garbage instead of `blue-cat</parameter></function></tool_call>`.
- Measured server speed: 96 tokens in 82.43s, about `1.2 tok/s`.

Auto-tool flag check:

`build/current-mimo-v25-required-tool-prefix-toolonly-auto-tool-20260607`

- `--enable-auto-tool-choice` did not change the failure.
- Tool calling was reported enabled with parser `xml_function`.
- Raw preview was the same malformed XML prefix plus punctuation.
- Measured server speed: 96 tokens in 75.97s, about `1.3 tok/s`.

Plain XML classification check:

`build/current-mimo-v25-xml-classification-default-cache-20260607`

- With no Tool API and a plain exact-copy XML prompt, MiMo did not print the
  requested XML. It returned only `Print exactly this text and nothing else:`.
- A simple non-XML text prompt still returned coherent text: `The sky is blue.`
- Default cache mode still skipped generic TurboQuant KV because MiMo has a
  mixed full/SWA `RotatingKVCache` layout.
- Speeds remained red: exact XML 9 tokens in 26.90s (`0.3 tok/s`) and simple
  text 6 tokens in 4.18s (`1.4 tok/s`).

Updated classification:

- Runtime metadata propagation was a real bug and is now fixed at source-test
  level.
- The live model/runtime path still fails required tool E2E because the model
  does not generate the argument value or closing XML after the valid prefix.
- This is not solved by enabling auto tools, disabling/enabling generic KV
  quantization, or parser repair.
- Generic TurboQuant KV is still correctly skipped for MiMo's mixed
  full/sliding cache contract. The speed issue needs a real MiMo-specific
  routed-expert/cache/kernel optimization, not a flat TQ-KV substitution.
- MiMo remains release-red for tools, speed, long-prompt quality, and all
  VL/audio/video rows.

Source-vs-quant preflight:

`build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-prefix-preflight-20260607.json`

- Status: `missing_prerequisites`.
- Source model path exists on `erics-m5-max2.local`:
  `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`.
- Quant model path exists locally:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Source endpoint `http://erics-m5-max2.local:8126` is down.
- Quant endpoint `http://127.0.0.1:8897` is down.
- No source-vs-quant classification can be claimed until both endpoints are
  intentionally launched and the prompt rows execute.

## 2026-06-07 XML fallback system-scope prompt fix

Source change:

- `vmlx_engine/api/tool_calling.py` now tries MiMo/XML-function fallback
  injection in the ChatML system turn before falling back to user-turn
  injection.
- This keeps the concrete native XML example in instruction scope instead of
  prepending it to the user's request.
- This is not a tool-call synthesis fix and does not fabricate arguments.

No-heavy proof:

- `py_compile` passed for `vmlx_engine/api/tool_calling.py` and
  `tests/test_tool_fallback_injection.py`.
- Focused parser/fallback tests passed: `13 passed`.
- Real local MiMo tokenizer render proof:
  - `system_has_tool_prompt=True`
  - `user_has_tool_prompt=False`
  - `system_has_record_fact=True`
  - `system_has_blue_cat=True`

Live artifact:

`build/current-mimo-v25-required-tool-system-fallback-live-20260607`

Live result:

- HTTP 400 remained because no parsed tool call was produced.
- Raw preview:
  `<tool_call>\n<function=record_fact>\n<parameter=value>` followed by
  punctuation/fullwidth-comma garbage.
- Measured server speed: 96 tokens in 80.02s, about `1.2 tok/s`.

Updated classification:

- MiMo XML fallback prompt placement is now cleaner and proven with the real
  tokenizer.
- MiMo required tool E2E remains red.
- MiMo speed remains red.
- The remaining blocker is not just fallback placement; it still requires
  source-vs-quant classification and/or real guided XML decode/model-artifact
  work.

## 2026-06-07 required-tool logits/cache divergence proof

Artifacts:

- `build/current-mimo-v25-required-tool-value-logits-20260607.json`
- `build/current-mimo-v25-prefix-completions-continuation-20260607`
- `build/current-mimo-v25-required-tool-value-continuation-logits-20260607.json`
- `build/current-mimo-v25-required-tool-cache-vs-full-logits-20260607.json`
- `build/current-mimo-v25-cache-decode-mask-ab-20260607.json`
- `build/current-mimo-v25-required-tool-cache-vs-full-logits-after-attn-cache-patch-20260607.json`

Findings:

- Direct full-forward logits after the rendered prompt plus
  `<tool_call>\n<function=record_fact>\n<parameter=value>` rank `blue` first
  with logprob about `-0.0006`.
- `/v1/completions` continuation from the same rendered prompt plus prefix
  starts with `blue`, proving the first value token is not a parser artifact.
- Direct full-forward logits after `blue` rank `-cat` first with logprob about
  `-0.0006`.
- Direct full-forward logits after `blue-cat` rank `</` first with logprob
  about `-0.0009`.
- Cached incremental decode diverges after `blue`: the same model/cache state
  ranks newline first and `-cat` around `-10.84`.
- Cached incremental decode also diverges after forced `blue-cat`: newline/EOS
  outrank the expected closing XML.
- Rotating cache offsets after the 406-token prefill remain absolute (`406`),
  so this is not a simple RoPE offset reset.
- Disabling cached decode masks for the diagnostic row does not change the bad
  ranking.
- A diagnostic patch that forced MiMo attention to pass `cache=None` to SDPA
  after manual `update_and_fetch` also did not change the bad ranking; it was
  removed and is not claimed as a fix.

Updated classification:

- The MiMo required-tool failure is now narrowed to incremental cache/decode
  divergence after the first generated value token.
- Full-forward model logits know the correct `blue-cat</...` continuation.
- The current cached decode path corrupts the continuation into newline and
  punctuation.
- This is a runtime/cache/decode compatibility blocker unless source-vs-quant
  later proves the local artifact has different cache behavior than source.
- MiMo remains release-red.

## 2026-06-07 keep=0 rotating-cache fix and required-tool pass

Root cause artifact:

`build/current-mimo-v25-rotating-cache-keep-ab-20260607.json`

Finding:

- MiMo's stale text runtime created SWA caches with
  `RotatingKVCache(max_size=sliding_window, keep=4)`.
- `keep=4` made cached decode after `blue` rank newline/punctuation first and
  put `-cat` around logprob `-10.84`.
- `keep=0` made cached decode after `blue` rank `-cat` first with logprob about
  `-0.0004`, matching full-forward behavior.
- `keep=8` was also wrong, proving this is specifically the preserved-prefix
  SWA cache behavior, not generic cache use.

Source fix:

- `vmlx_engine/models/mllm.py` now patches stale MiMo text runtimes at
  registration time so `Model.make_cache()` constructs SWA
  `RotatingKVCache(..., keep=0)`.
- This is a cache-layout fix only. It does not synthesize tool calls, alter
  parser output, or force argument values.

Focused source proof:

- `py_compile` passed for `vmlx_engine/models/mllm.py` and
  `tests/test_mimo_v2_rotating_cache_patch.py`.
- Focused test passed: `tests/test_mimo_v2_rotating_cache_patch.py`.

Direct cache/logit proof after fix:

`build/current-mimo-v25-cache-vs-full-after-keep0-patch-20260607.json`

- First SWA cache rows report `keep=0`.
- Cached decode after `blue` ranks `-cat` first.
- Cached decode after `blue-cat` ranks `</` first.

Live required-tool proof after fix:

`build/current-mimo-v25-required-tool-live-after-keep0-patch-20260607`

- `/v1/chat/completions` returned HTTP 200.
- `finish_reason="tool_calls"`.
- Returned OpenAI tool call:
  `record_fact({"value":"blue-cat"})`.
- Server log confirms the keep=0 patch installed:
  `Installed vMLX MiMo-V2 keep=0 rotating-cache patch`.
- Speed remains red: 22 completion tokens in 35.02s, about `0.6 tok/s`.

Updated classification:

- `MIMO-CACHE-DECODE-001` is fixed for the required-tool continuation.
- `MIMO-TOOL-001` has a narrow required-tool pass for this `record_fact` row.
- MiMo is still not release-green: speed, long-prompt quality, tool-result
  continuation, full cache/L2 matrix, UI parity, and VL/audio/video remain
  open.

## 2026-06-07 ChatML tool-fallback framing fix and live result

Source change:

- `vmlx_engine/api/tool_calling.py` now keeps MiMo/XML-function fallback tool
  instructions inside the first rendered ChatML system turn if MiMo's template
  drops synthetic fallback messages during re-render.
- `tests/test_tool_fallback_injection.py` now covers this regression.

No-heavy proof:

- `py_compile` passed for `vmlx_engine/api/tool_calling.py` and
  `tests/test_tool_fallback_injection.py`.
- Focused parser/fallback tests passed: 12 passed.
- Render proof for the real local MiMo tokenizer showed
  `inside_system=True` and `prefix_outside_chatml=False`.

Live proof:

`build/current-all-local-model-smoke-mimo-v25-jang2l-after-chatml-tool-fallback-20260607`

Command boundary:

- Current source venv.
- Installed model:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Continuous batching, paged cache, block-disk L2, and native mixed full/SWA
  cache active.
- `--kv-cache-quantization none`.
- No media/video/reasoning; tools included.

Live positives:

- `text_cache_repeat_1`: HTTP 200, `ACK`.
- `text_cache_repeat_2`: HTTP 200, `ACK`, `cached_tokens=67`,
  `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, `blue cat`.
- Block-disk L2 wrote 4 blocks / 141 tokens.
- Native cache remained `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.
- Generic TurboQuant KV remained disabled for this diagnostic.

Live failure:

- `tool_required`: HTTP 400, no parsed tool calls.
- Server raw preview:
  `<tool_call>\n\n：，，，。：，，。。。。。。，。。。。，。。，。。，，...`
- The model still failed to emit `<function=record_fact>` or
  `<parameter=value>blue-cat</parameter>`.
- Generation speed for the tool failure was about `1.7 tok/s`.

Classification:

- The ChatML fallback-framing bug is fixed.
- MiMo required-tool generation remains red after the framing fix.
- The remaining blocker is deeper than parser aliasing or prompt placement:
  current decode/model/runtime path still cannot reliably continue the
  XML-function grammar for required tool calls.
- Do not clear MiMo until a corrected artifact, source-vs-quant divergence fix,
  or a real constrained/guided XML decoder is implemented and live-proven
  without synthetic tool-call fabrication.

## 2026-06-07 source-vs-quant harness tool row and stale-cache cleanup

Source change:

- `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py` now
  includes the current release-blocking XML tool row:
  `xml_tool_required_record_fact`.
- The runner now extracts OpenAI `tool_calls`, records source/quant tool-call
  signatures, and classifies tool divergence separately from visible-text
  divergence.
- Proof pointers in the MiMo current-audit runner and release-regression
  manifest now point at:
  `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`.

Focused validation:

- `py_compile` passed for the source-vs-quant runner, MiMo current audit,
  release manifest, and source-vs-quant tests.
- `tests/test_mimo_v2_source_vs_quant_probe.py`: 8 passed.

Current preflight:

`build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`

- `status=missing_prerequisites`.
- Source path exists on Max2:
  `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`.
- Quant path exists locally:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Source endpoint `http://erics-m5-max2.local:8126` is not healthy.
- Quant endpoint `http://127.0.0.1:8897` is not currently running.
- This is preflight evidence only; it cannot classify model-upload vs runtime
  until both endpoints run and rows execute.

Current audit:

`build/current-mimo-v2-jang2l-current-audit-after-tool-row-source-quant-preflight-stale-clean-20260607.json`

- `status=open`.
- `stale_local_state_absent=true` after deleting the stale HF remote-code cache:
  `/Users/eric/.cache/huggingface/modules/transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L`.
- Remaining blockers:
  `mimo_long_prompt_coherence_blocked`,
  `mimo_tool_protocol_blocked`,
  `mimo_decode_speed_below_release_target`,
  `mimo_cb_system_prompt_working_set_pressure_blocked`,
  `mimo_source_vs_quant_first_divergence_missing_or_failed`,
  `mimo_vl_audio_video_unwired`.

## Remote canonical artifact comparison

Checked `erics-m5-max2.local` on 2026-06-06.

Remote canonical path:

`/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`

Local canonical path:

`/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`

Comparison result:

- Local size: `106G`.
- Remote size: `106G`.
- Local shard count: `150`.
- Remote shard count: `150`.
- `config.json` SHA-256 matches: `c52d416c847d75c3c6cf5fb4c7eb984bbe868a552bd063659ffcc9fc3199ffe8`.
- `model.safetensors.index.json` SHA-256 matches: `5868a0220ff6610a962da52211ebb59b4a98b09a1c7785c20633a711dd9b006b`.
- `chat_template.jinja` SHA-256 matches: `3134ac101acd29d3ab41297707cc1a85699f5f0acb283fdeb0681e3750998403`.
- `model-00001-of-00150.safetensors` SHA-256 matches: `b5c8623063d66fea9fd6578dc354dc82fbf7a007ae6262397b9263a4ed2baccf`.
- `model-00150-of-00150.safetensors` SHA-256 matches: `4cdad7d9e973a8617851666ea1ab12bf67a4b8ca00ffd35ea5a0996c1f49308f`.

Conclusion:

- Re-downloading the canonical remote MiMo JANG_2L artifact is not expected to fix the current tool-call failure.
- The local installed artifact appears to match the remote canonical artifact for metadata and sampled shards.
- The remote contract also records expected text generation speed around `1.97-2.64 tok/s`, not `40+ tok/s`, for this 310B / 15B-active bundle on local M5 Max.
- Remote notes say vision/audio are preserved in the bundle but runtime support was deferred. This matches current vMLX capabilities: text runtime only, media preserved but unwired.

## Reproducible harness KV-none proof

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-kvnone-extra-args-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Command boundary:

- Runner used `VMLINUX_BENCH_EXTRA_SERVE_ARGS='--kv-cache-quantization none'`.
- Served command tail includes `--kv-cache-quantization none`.
- Health reports `kv_cache_quantization.enabled=false`.
- Health reports `turboquant_kv_cache.enabled=false`.
- Native cache remains `mimo_v2` / `mixed_swa_kv_v1` with prefix, paged cache, and block-disk L2 enabled.

Results:

- `text_cache_repeat_1`: HTTP 200, `ACK`.
- `text_cache_repeat_2`: HTTP 200, `ACK`, `cached_tokens=67`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, `blue cat`.
- `tool_required`: HTTP 400, no parsed tool calls.
- Raw preview: `<tool_call>` followed by punctuation/fullwidth-comma garbage instead of `<function=record_fact>`.

Conclusion:

- MiMo tool corruption is not caused by q4 KV cache storage or generic TurboQuant KV.
- The failure persists with cache storage quantization disabled while the intended prefix/paged/L2 cache stack remains active.

## Direct continuation probe

Artifact:

`build/current-mimo-v25-tool-continuation-probe-20260606`

Command boundary:

- Served installed `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Used SimpleEngine: `--no-continuous-batching --disable-prefix-cache --kv-cache-quantization none`.
- Used `/v1/completions`, not Chat Completions tool parsing.
- Prompt was rendered with the real MiMo chat template plus concrete `mimo_xml_function` fallback example.
- No OpenAI tool parser, no `tool_choice=required`, no paged prefix cache, and no generic TurboQuant KV participated in this probe.

Results:

- Base rendered prompt output:
- `<think><think><think>。，，，。。。，，，，，。。，，，，，，。。，，，。。，。，。。。。，，，。，。，，。。。`
- Assistant-prefilled prompt ending in `<tool_call>\n<function=record_fact>\n<parameter=value>\n` output:
- `blue\n\n\n\n\n，，\n\n。，，，，，。，，，，。，。，，，，，，。，，。。。。。。。。。。。。。。。`
- Base probe speed: 48 completion tokens in 50.42s, about `1.0 tok/s`.
- Prefilled-function probe speed: 48 completion tokens in 29.52s, about `1.6 tok/s`.

Conclusion:

- The current installed MiMo JANG_2L path cannot reliably continue the native XML tool grammar even when the runtime has already supplied the `<function=record_fact>` and `<parameter=value>` prefix.
- This rules out parser-only failure, `tool_choice=required` response wrapping, continuous batching, prefix cache, paged L2, q4 KV storage, and generic TurboQuant KV as sole causes.
- Remaining likely causes are model/artifact/tool-template training incompatibility or missing optimized MiMo/JANG_2L decode support for this bundle.
- A runtime-side fake tool-call synthesis would be incorrect and must not be used as a release fix.
- A legitimate future runtime fix would need constrained/guided XML tool decoding or a proven better MiMo artifact that naturally emits valid XML tools.

## Remote `~/jang` runtime comparison

Remote checked:

`erics-m5-max2.local:/Users/eric/jang`

Findings:

- Remote implementation doc: `research/MIMO-V2.5-IMPLEMENTATION.md`.
- Remote quant contract: `docs/runtime/2026-05-27-mimo-v2-jang-2l-quant-contract.md`.
- Remote runtime source: `jang-tools/jang_tools/mimo_v2/mlx_model.py`.
- Local bundled runtime SHA-256: `7b6a0a524b486907f5e2fd2c5a122c1dc58726be1d8d6c2c1304a9aafda57f7d`.
- Remote runtime SHA-256: `a33ec86fc409e83feccf049c74d4c19be813ec6f5d130a6a20bf19d713d9372d`.

Remote docs confirm:

- The promoted coherent canonical MiMo JANG_2L path was measured at about `1.974-2.640 generation tok/s` on the local M5 Max.
- The docs explicitly say the current affine MLX path is coherent but not a `30 tok/s` path on this local M5 Max.
- A faster candidate reached `5.626 generation tok/s` on cached France but failed arithmetic with output `45.` for `2 + 2`, so it was rejected.
- Current 40+ tok/s expectations therefore require a different optimized artifact/runtime class, not just a serving flag on this 106GB affine JANG_2L bundle.

Runtime integration delta:

- Local bundled `jang_tools` lacked the remote MiMo compiled single-token router.
- vMLX now installs a narrow fallback compiled router during MiMo MLLM registration if the bundled `jang_tools` runtime is stale.
- The patch preserves the existing vMLX `inputs_embeds` compatibility wrapper and does not override newer `jang_tools` runtimes that already expose `run_compiled_mimo_decode_router`.

Live fallback-router probe:

Artifact:

`build/current-mimo-v25-compiled-router-live-probe-20260606`

Result:

- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Prompt prefilled through `<tool_call>\n<function=record_fact>\n<parameter=value>\n` still generated `blue` plus punctuation/newline garbage.
- Short first request speed was 16 completion tokens in 30.67s, about `0.5 tok/s`; this does not prove a speed win and includes first compiled-router request overhead.

Conclusion:

- The stale-runtime compiled-router integration miss is real and now patched in vMLX.
- It is not sufficient to clear `MIMO-TOOL-001`.
- It is not sufficient to clear `MIMO-SPEED-001`.
- Do not call MiMo fixed until either a better artifact is proven or a real constrained/guided XML decoder plus optimized routed-expert decode path is implemented and E2E-proven.

## 2026-06-07 current normal-engine refresh after Qwen35 work

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-refresh-after-qwen35-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Bundle/runtime facts checked:

- Local path: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- `config.json` has `model_type=mimo_v2`, full plus SWA rotating attention, media sidecars, and native XML tool grammar in the chat template.
- No local `jang_config.json` exists in this bundle; vMLX must not rely on missing JANG metadata to classify tools/media/cache.
- `generation_config.json` has `do_sample=false`, `top_p=0.95`, and EOS ids `[151643, 151645, 151672]`.

Live positives in current Python engine:

- MLLM path loaded with continuous batching, paged prefix cache, block-disk L2, and native mixed full/SWA cache.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=67`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- Block-disk L2 wrote 4 blocks / 141 tokens.
- Native cache reports `mimo_v2` / `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`.

Still failing:

- `tool_required`: HTTP 400 after 96 generated tokens.
- Expected: `record_fact({"value":"blue-cat"})`.
- Actual: zero parsed tool calls; raw preview starts `<tool_call>\n\n：，，，。...`.
- Server log explicitly says `tool_choice='required' but model produced no tool calls`.
- Failing tool row speed is about `1.6 tok/s`; this remains far below the requested 40+ tok/s release target.

Classification update:

- Cache is not the sole cause: current cache/L2/multiturn rows pass while tools fail, and earlier KV-none/no-cache probes failed the same tool grammar.
- Schema placement is not the sole cause: ChatML fallback placement is fixed and the current template already has native XML tool instructions.
- Current blocker is unresolved `decode_loop` versus `model_artifact` for XML tool grammar and speed until source-vs-quant first divergence runs against a live source endpoint or a replacement artifact passes the same rows.
- Do not delete current/local MiMo copies until all old paths are enumerated and a replacement artifact is proven better by text/cache/tool/long-prompt/speed rows.
