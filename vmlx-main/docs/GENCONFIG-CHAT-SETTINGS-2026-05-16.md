# vMLX 1.5.37 Generation Defaults and Chat Settings Contract

Date: 2026-05-16

## Contract

Generation policy and cache reuse are separate.

- Prefix cache, paged cache, and block/L2 disk cache may reuse computation for identical model/token/media prefixes. That is cache acceleration only.
- Sampling parameters, reasoning mode, and max output tokens must come from the current request/chat or the model bundle defaults. They must not be inherited from another chat, another session, or stale per-model settings.

## Resolution Order

Engine sampling defaults resolve as:

1. Explicit API or chat request value.
2. Explicit CLI default from a CLI user.
3. `jang_config.chat.sampling_defaults`.
4. `generation_config.json`.
5. Engine family fallback.

The panel session UI may display bundle defaults, but normal app startup does not emit:

- `--default-temperature`
- `--default-top-p`
- `--default-top-k`
- `--default-min-p`
- `--default-repetition-penalty`
- `--default-enable-thinking`

## Max Tokens

There are separate token concepts:

- Max output tokens: response length cap (`max_tokens`, `max_output_tokens`, or bundle `max_new_tokens`).
- Prompt/context tokens: the amount of input context accepted before prefill.
- Model context window: model capability detected from the bundle; the UI can use it as a default/display value, but it cannot make a model support more context than its weights/config support.

The chat settings panel keeps a Max Tokens input for max output tokens:

- Blank means no chat override is saved or sent.
- The placeholder shows the model-declared `max_new_tokens` when present.
- A saved value applies only to that chat.
- API callers can still send `max_tokens` or `max_output_tokens` per request.

Hidden family floors were removed. For example, DSV4 explicit `max_tokens=128` stays `128`; it is not raised to `4096`.

The server/session settings panel keeps a separate Max Context Tokens input:

- Blank/Auto means the app does not emit a context flag and the engine uses its memory-safe prompt estimate.
- A saved server value emits `--max-prompt-tokens N`.
- The engine stores that in `_max_prompt_tokens`, not the old dead `_max_context_length` name.
- Over-limit prompts are rejected before prefill with `prompt_too_long`; vMLX does not silently trim or slide the chat history.
- This cap has two layers: route preflight rejects obviously oversized request bodies, then the engine/scheduler enforces the exact rendered/tokenized prompt after chat-template control tokens are added and before cache lookup or prefill.
- The setting is route-level and model-family agnostic: OpenAI Chat, Responses, Completions, Anthropic Messages, and Ollama chat/generate all forward the cap into engine generation.
- API requests may also send a request-local cap via `max_prompt_tokens`, `max_context_tokens`, or `max_context`. Ollama-compatible requests may use `options.num_ctx` / `options.num_context`. These values can lower the active cap for that one request, but they cannot raise above the server/session `--max-prompt-tokens` ceiling.
- VLM media parts are not counted as zero. The route guard counts text plus a conservative media placeholder floor, then the MLLM batch generator enforces the exact processor-produced `input_ids` length before pixel-cache writes, prefix/paged cache lookups, or prefill.

Max Context Tokens is the max prompt/context admission cap. It must be wired as
a request-admission setting across startup, API routes, prompt rendering,
tokenization, cache admission, and error reporting. It must not be confused with
output-token budgeting.

| Layer | Functions / files | Required behavior |
| --- | --- | --- |
| Model metadata detection | `panel/src/main/model-config-registry.ts`, `CreateSession.tsx`, `ServerSettingsDrawer.tsx`, `SessionConfigForm.tsx` | Read `max_position_embeddings` / model-declared context only as the Auto/default hint. Do not save that hint as an explicit override. |
| Panel session storage | `panel/src/main/sessions.ts`, `panel/src/main/server.ts`, `panel/src/env.d.ts`, `SessionSettings.tsx`, `SessionConfigForm.tsx` | Persist only a user-entered server/session value in `config.maxContextLength`. `0` or blank means Auto. |
| Panel launch args | `panel/src/main/sessions.ts::buildArgs`, `SessionSettings.tsx::buildCommandPreview`, `panel/tests/settings-flow.test.ts` | Emit `--max-prompt-tokens N` only when `config.maxContextLength > 0`. Startup must not emit `--default-*` sampling flags for bundle defaults. |
| CLI parser | `vmlx_engine/cli.py::serve_command`, `vmlx_engine/server.py` CLI parser | Parse `--max-prompt-tokens` as the max prompt/context cap accepted before prefill. This is not `--max-tokens`. |
| Server startup | `server.py::load_model`, `_estimate_max_prompt_tokens`, `_resolve_max_prompt_tokens` | Resolve explicit `--max-prompt-tokens` over the memory-safe auto estimate and store the result in `_max_prompt_tokens`, the real enforced variable. |
| Server state / wake | `server.py::_cli_args`, wake/reload load path, `/health` | Preserve and report the explicit cap across sleep/wake reload. Never rehydrate the removed `_max_context_length` name. |
| API request models | `ChatCompletionRequest`, `CompletionRequest`, `ResponsesRequest`, `AnthropicRequest`, `ollama_adapter.py`, app `api-gateway.ts` | Accept `max_prompt_tokens` / `max_context_tokens` / `max_context`; map Ollama `options.num_ctx` to `max_prompt_tokens`; normalize aliases before route logic. |
| API text estimate helpers | `_effective_max_prompt_tokens`, `_text_prompt_token_estimate`, `_prompt_content_token_estimate`, `_message_prompt_token_estimate`, `_reject_if_prompt_too_long_for_messages`, `_reject_if_prompt_too_long_for_prompts` | Compute the effective cap as `min(request cap, session cap)` when both are set, then preflight request bodies before generation so obviously oversized prompts fail with `prompt_too_long` before expensive prefill. |
| OpenAI Chat | `create_chat_completion`, `stream_chat_completion` | Preflight `messages` against the effective cap, forward `max_prompt_tokens` into chat kwargs, catch `PromptTooLongError` in both nonstream and SSE paths. |
| OpenAI Responses | `create_response`, `stream_responses_api` | Preflight normalized input messages against the effective cap, forward `max_prompt_tokens`, and return/stream `prompt_too_long` instead of a generic engine failure. |
| OpenAI Completions | `create_completion`, `stream_completions_multi` | Preflight raw prompts against the effective cap, forward `max_prompt_tokens`, and map exact-admission failures to `prompt_too_long`. |
| Anthropic Messages | `create_anthropic_message` | Preflight converted messages and forward the effective `max_prompt_tokens` into the same engine path. |
| Ollama Chat / Generate | `ollama_chat`, `ollama_generate` | Map `options.num_ctx` / vMLX max-context aliases, preflight chat or raw prompts against the effective cap, and forward `max_prompt_tokens` for both streaming and nonstream paths. |
| Batched text engine carry | `BatchedEngine.chat`, `BatchedEngine.stream_chat`, `BatchedEngine.generate`, `BatchedEngine.stream_generate` | Preserve the cap through template rendering, then pass it to text or MLLM dispatch. |
| Batched exact admission | `EngineCore.add_request`, `Scheduler.add_request` | Attach `_max_prompt_tokens` to the request, then enforce exact tokenized prompt length after chat-template control tokens and before prefix/paged/block cache lookup or prefill. Cache hits cannot bypass the cap. |
| Simple text engine | `SimpleEngine.generate`, `stream_generate`, `chat`, `stream_chat`, `_raise_if_prompt_over_limit` | Pop `max_prompt_tokens` so it never leaks to mlx-lm, and enforce exact rendered/tokenized prompt length before model calls. |
| MLLM text-only admission | `BatchedEngine` MLLM dispatch, `MLLMScheduler.add_request` | Enforce exact tokenizer length for text-only VLM/MLLM requests before admission. |
| MLLM media admission | route media preflight, `_configured_media_prompt_token_floor`, `MLLMBatchRequest.max_prompt_tokens`, `MLLMBatchGenerator._raise_if_prompt_over_limit` | Count text plus a conservative media placeholder floor before dispatch, then exact-check the VLM processor's `input_ids` length before pixel-cache writes and before prefix/paged/disk cache lookup. Cached pixel inputs are checked again before reuse. |
| Error type / mapping | `vmlx_engine/errors.py::PromptTooLongError`, `_prompt_too_long_response`, `_prompt_too_long_response_from_error` | Preserve token count, cap, source, and request id so all routes return the same `prompt_too_long` code. |

Functions that must stay separate from Max Context Tokens:

| Output-token surface | Role | Why it must not consume `max_prompt_tokens` |
| --- | --- | --- |
| `server.py::_resolve_max_tokens` | Resolves response length from request/CLI/bundle `max_new_tokens`. | It caps generation length, not prompt admission. |
| API request fields `max_tokens`, `max_completion_tokens`, `max_output_tokens` | Per-request output caps for OpenAI-compatible and Responses routes. | Users still need these for response length without changing server context admission. |
| `SamplingParams.max_tokens` and request `.max_tokens` | Scheduler/generator stop condition for generated tokens. | It controls `finish_reason="length"` / `max_tokens`, not prompt rejection. |
| `SingleBatchGenerator`, `MLLMBatchGenerator`, DSV4 batch generator output loops | Decode loop output stopping. | They should stop generation after the response budget, not reject input context. |
| Reasoning budget helpers (`reasoning_effort`, `thinking_budget`) | Output-side reasoning budget and template kwargs. | They may influence generated reasoning length but do not set prompt/context capacity. |
| Prefix/paged/block disk cache knobs | Reuse already-computed prefix state. | Cache reuse must be gated by prompt admission; cache settings must not carry generation or context policy across chats. |

Removed or forbidden names/behaviors:

- `_max_context_length` is intentionally absent from the engine; wiring to it is
  a no-op bug.
- `DEFAULT_BOUNDED_TOP_K` is gone; no hidden `top_k=40` fallback should be
  synthesized by the panel.
- Normal app startup must not synthesize `--default-temperature`,
  `--default-top-p`, `--default-top-k`, `--default-min-p`,
  `--default-repetition-penalty`, or `--default-enable-thinking` from bundle
  defaults. The engine reads the bundle.

## State Storage

New chats:

- Create no `chat_overrides` row by default.
- Do not seed from `model_settings`.
- Do not seed from sibling chats.
- Do not auto-apply starred/default chat profiles.
- Do not copy `generation_config` or `jang_config` values into SQLite.

Chat overrides:

- Are keyed by `chat_id`.
- Persist only after the user saves Chat Settings or loads/saves a profile into that chat.
- Are the only panel-side source for explicit sampling, reasoning, tools, system prompt, working directory, and max output token overrides.

Model settings:

- Are launch/session metadata only: alias, TTL, pinned, port, cache quantization, disk cache enabled.
- Sampling, max output tokens, and reasoning mode are cleared from old rows and no longer written back.

## Migration

The 1.5.37 database migration clears known historical generic values that were generated by old panel defaults:

- `temperature = 0.7`
- `top_p = 0.95`
- `top_k = 40`
- `max_tokens IN (4096, 12000, 12068)`

It also clears per-model `temperature`, `top_p`, `max_tokens`, and `reasoning_mode` back to null/auto so those rows cannot poison future chats.

Non-generic explicit chat settings are preserved.

## Reasoning

Session startup does not pass server-level `enable_thinking`. The engine resolves model defaults, and explicit chat/API requests carry `enable_thinking` only for that request.

ZAYA text explicit reasoning-on opens a qwen3 `<think>` rail when the registry declares qwen3 reasoning support. Current ZAYA1-VL uploaded bundles use a plain VLM template and live proof showed hidden-only output, so runtime and panel capability resolution suppress ZAYA1-VL reasoning until the model upload ships a real VLM thinking template and passes live visible-output proof.

## Parser Fix

DSV4 DSML parsing no longer accepts canonical parser output when required arguments are `{}` or contain raw DSML/HTML-ish markup. Those cases fall through to the repair parser, which can recover plain `<param name="...">...</param>` bodies.

## Verification So Far

Commands already run in this release worktree:

```sh
.venv/bin/python -m pytest tests/test_reasoning_modes.py tests/test_dsml_tool_parser.py tests/test_engine_audit.py -q -k 'zaya_vl or canonical_encoder_empty_required_args or panel_defaults_are_speed_oriented or session_command_preview'
# 5 passed, 410 deselected

cd panel && npm test -- settings-flow.test.ts generation-defaults.test.ts request-builder.test.ts
# 236 passed

.venv/bin/python -m pytest tests/test_reasoning_modes.py tests/test_dsml_tool_parser.py tests/test_engine_audit.py tests/test_cache_bypass.py tests/test_cache_hit_worker_dequant.py -q
# 473 passed, 2 skipped

cd panel && npm test
# 1804 passed

.venv/bin/python -m pytest tests/test_sampling.py tests/test_reasoning_modes.py tests/test_dsml_tool_parser.py tests/test_cache_bypass.py tests/test_cache_hit_worker_dequant.py -q
# 117 passed

.venv/bin/python -m pytest tests/test_server.py tests/test_batching.py::TestSchedulerBasic tests/test_simple_engine.py tests/test_reasoning_modes.py tests/test_dsml_tool_parser.py tests/test_cache_hit_worker_dequant.py -q
# 160 passed, 3 deselected
```

Live prompt/context checks after the VLM media admission fix:

- ZAYA1-VL HTTP server with `--max-prompt-tokens 1050` accepted a one-image prompt because that bundle's processor keeps the exact `input_ids` prompt at 4 tokens and carries image tensors separately. That verified no false 413 from the exact path.
- Direct ZAYA1-VL engine call with the same image and `max_prompt_tokens=3` raised `PromptTooLongError`: `tokenized VLM media prompt has 7 tokens, max prompt/context tokens is 3`, before cache lookup/store.

Before release, still run fresh:

```sh
git diff --check
cd panel && npm run typecheck
cd panel && npm test
.venv/bin/python -m pytest tests/test_sampling.py tests/test_reasoning_modes.py tests/test_dsml_tool_parser.py tests/test_engine_audit.py tests/test_cache_bypass.py tests/test_cache_hit_worker_dequant.py -q
```

Then build, sign, notarize, staple, validate, and run the packaged app release gate before publishing.
