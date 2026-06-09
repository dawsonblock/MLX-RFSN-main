# vMLX 2026-05-13 Engine/Cache/App Gate

This note records the release-gate boundary for the Python/Electron vMLX
engine/app work on 2026-05-13. Raw private artifacts are under
`docs/internal/release-gates/20260513_installed_app_full_model_gate/`.

## Runtime Fixes

- `SimpleEngine` now loads and runs direct/no-continuous-batching model work on
  a dedicated single worker thread. MLX streams are thread-local; loading on
  one thread and later generating on arbitrary default-executor threads caused
  direct JANG/JANGTQ/VLM/hybrid rows to fail with missing `Stream(gpu,N)`.
- Direct streaming usage accounting now falls back to formatted prompt token
  counts when mlx-lm or mlx-vlm stream chunks report `prompt_tokens=0`.
- Direct MLLM thinking-off prompts now consult the real bundle/template
  contract before appending a synthetic `<think></think>` sentinel. Reasoning
  capability and `think_in_template` are separate; ZAYA/ZAYA-VL can support
  reasoning without starting the default/off prompt inside a think rail.
- MLLM cache bypass now reaches the component that actually fetches cache.
  The API/server path already accepted `skip_prefix_cache` and `cache_salt`,
  and `MLLMScheduler.add_request()` attached `_bypass_prefix_cache` to the
  `MLLMRequest`, but the scheduler dropped that flag when it converted the
  request into `MLLMBatchRequest`. The batch generator owns paged/CCA
  prefix-cache fetch, so MLLM text requests could still report cached tokens
  even when bypass was requested. The scheduler now copies the bypass flag to
  the batch request before dispatch.
- The same bypass is now applied to the MLLM vision pixel cache. Pixel-cache
  fetch and store are gated by `_bypass_prefix_cache`, so `skip_prefix_cache`
  and salted requests isolate media preprocessing cache state as well as
  token-prefix state.
- MLLM generation streams are now reset on server-side model replacement and
  deep sleep teardown. The reproduced failure was specific and real: after
  deep sleep, ZAYA-VL JANGTQ4 could reload on a fresh MLLM worker thread and
  then hit a stale process-global MLX stream from the old worker during paged
  CCA block-L2 restore, raising `There is no Stream(gpu,N) in current thread`.
  The fix clears both MLLM stream globals at teardown so the next worker lazily
  creates streams on its own thread.

## Live Installed-App Evidence

The installed app was rebuilt and installed from the 1.5.33 source tree using
`bash panel/scripts/build-and-install.sh`. The installed bundle reports
`vmlx_engine 1.5.33`, imports `jang_tools.turboquant.mpp_nax_kernel`, contains
the MLLM stream-reset patch, installs as `/Applications/vMLX.app`, and its
gateway responds on `127.0.0.1:8080`.

Post-build verification:

- `codesign --verify --deep --strict --verbose=2 /Applications/vMLX.app`
  passed.
- Installed bundle inspection confirmed:
  `reset_generation_streams` exists, `load_model()` calls
  `_reset_mllm_generation_streams()`, and `admin_deep_sleep()` calls
  `_reset_mllm_generation_streams()`.
- Installed bundle inspection also confirmed the MLLM bypass patch was present:
  `mllm_scheduler.py` copies `_bypass_prefix_cache` onto the batch request, and
  `mllm_batch_generator.py` gates vision pixel-cache fetch/store with the same
  flag.
- The installed app bundled Python smoke passed for Qwen3.6-27B JANG_4M and
  ZAYA1-VL-8B JANGTQ4 after the clean rebuild.
- No model servers or image servers were left running after the final gates.

Representative all-cache-mode rows passed:

- Qwen3.6-27B JANG_4M: direct, prefix, paged L1, paged L2.
- Gemma-4-26B JANG_4M: direct, prefix, paged L1, paged L2.
- ZAYA1-VL-8B JANGTQ4: direct, prefix, paged L1, paged L2.
- Ling-2.6-flash JANGTQ2: direct, prefix, paged L1, paged L2.

Additional direct plus paged-L2 rows passed:

- ZAYA1-8B JANGTQ_K and MXFP4.
- ZAYA1-VL-8B JANGTQ_K.
- Qwen3.6-27B MXFP4.
- Qwen3.6-35B-A3B 4bit and JANGTQ.
- Nemotron-Omni-Nano JANGTQ, JANGTQ4, and MXFP4.
- MiniMax-M2.7 Small JANGTQ.
- Hy3-preview JANGTQ2.
- DeepSeek-V4-Flash JANGTQ_K.
- Laguna-XS.2 JANGTQ.
- Ling-2.6-flash JANGTQ and MXFP4.
- MiniMax-M2.7 JANGTQ and JANGTQ_K, both JANGQ and dealign package layouts.
- DeepSeek-V4-Flash JANGTQ_K upload-staging layouts for JANGQ and Osaurus.

The latest installed-app direct/L2 sweep produced 52 result files across the
below-budget local model matrix:

- 50/52 rows passed.
- 2/52 rows remain `REVIEW`; both are the same bundle:
  `ZAYA1-VL-8B-MXFP4` in text-only strict-recall mode.
- PP range across the sweep was 152.6 to 8321.5 prompt tokens/s.
- TG range across the sweep was 5.4 to 62.7 output tokens/s.
- No row hit a traceback, stream-affinity failure, OOM, or working-set
  rejection.
- The broad `/v1/responses` READY content check passed 50/52 rows. The same
  two `ZAYA1-VL-8B-MXFP4` rows returned image-placeholder-style text instead
  of READY and stay in review.

Named model acceleration/cache summary from that sweep:

| Model row | Status | PP tok/s | TG tok/s | Kernel reported | Cache detail |
| --- | --- | ---: | ---: | --- | --- |
| `MiniMax-M2.7-JANGTQ` / `JANGTQ_K` | Pass | 264.7-267.9 | 5.5-7.9 | `turboquant_codebook_mpp_nax` | `paged+tq` |
| `Hy3-preview-JANGTQ2` | Pass | 223.7-225.1 | 5.4-6.2 | `turboquant_codebook_mpp_nax` | `paged+tq` |
| `Ling-2.6-flash-JANGTQ` / `JANGTQ2` | Pass | 514.1-537.9 | 9.2-13.9 | `turboquant_codebook_mpp_nax` | `paged+ssm` |
| `Ling-2.6-flash-MXFP4` | Pass | 801.6-844.7 | 6.1-17.7 | `affine_quantized_matmul` | `paged+ssm` |
| `ZAYA1-8B-JANGTQ_K` | Pass | 3834.4-4126.9 | 30.2-35.0 | `turboquant_codebook_mpp_nax` | `paged+zaya_cca` |
| `ZAYA1-8B-MXFP4` | Pass | 7091.7-8321.5 | 39.3-52.3 | `affine_quantized_matmul` | `paged+zaya_cca` |
| `ZAYA1-VL-8B-JANGTQ4` / `JANGTQ_K` | Pass | 3245.5-3951.2 | 23.7-54.9 | `turboquant_codebook_mpp_nax` | `paged+zaya_cca` |
| `ZAYA1-VL-8B-MXFP4` | Review | 7310.9-7689.3 | 9.6-56.9 | `affine_quantized_matmul` | `paged+zaya_cca` |
| `Qwen3.6-27B-JANG_4M` | Pass | 803.7-856.1 | 10.3-14.9 | `affine_quantized_matmul` | `paged+ssm` |
| `Qwen3.6-27B-MXFP4` | Pass | 576.5-624.3 | 11.9-19.6 | `affine_quantized_matmul` | `paged+ssm` |
| `Qwen3.6-35B-A3B-4bit` | Pass | 2732.4-3988.9 | 19.5-62.7 | `affine_quantized_matmul` | `paged+ssm` |
| `Qwen3.6-35B-A3B-JANGTQ` | Pass | 1407.2-1856.3 | 22.5-42.6 | `turboquant_codebook_mpp_nax` | `paged+ssm` |
| `Nemotron-Omni-Nano-JANGTQ` / `JANGTQ4` | Pass | 853.0-976.6 | 17.3-27.8 | `turboquant_codebook_mpp_nax` | `paged+ssm` |
| `Nemotron-Omni-Nano-MXFP4` | Pass | 3379.6-3621.2 | 25.6-49.8 | `affine_quantized_matmul` | `paged+ssm` |
| `Gemma-4-26B-JANG_4M` | Pass | 2418.7-3684.0 | 13.3-49.4 | `affine_quantized_matmul` | `paged` |
| `DeepSeek-V4-Flash-JANGTQ_K` | Pass for short/cache gate | 152.6-242.5 | 6.1-8.3 | `turboquant_codebook_mpp_nax` | `paged+dsv4` |

This split is intentional. JANGTQ/MXTQ TurboQuant codebook rows use the custom
`turboquant_codebook_mpp_nax` lane when available; affine/MLX/MXFP4/JANG_4M
rows use MLX's native affine quantized-matmul path. Image-generation/edit rows
are mflux image-runtime rows, not JANGTQ MPP/NAX rows.

`~/models` coverage was cross-checked against the saved gate artifacts. Every
local model root at or below the 85 GB single-process budget has at least one
live installed-app result artifact. Most have direct plus paged-L2 coverage;
the core families also have all-cache-mode and/or full API-adapter parity
coverage. The only below-budget root intentionally left as `REVIEW` is
`ZAYA1-VL-8B-MXFP4` in text-only strict recall mode.

Installed-app API parity passed after the clean rebuild:

- Core families: Qwen3.6-27B JANG_4M, Gemma-4-26B JANG_4M,
  ZAYA1-VL-8B JANGTQ4, Ling-2.6-flash JANGTQ2, Laguna-XS.2 JANGTQ,
  Nemotron-Omni-Nano JANGTQ4, and MiniMax-M2.7 Small JANGTQ.
- Heavy rows: Hy3-preview JANGTQ2 and DeepSeek-V4-Flash JANGTQ_K.
- Surfaces covered for those rows: OpenAI Chat Completions stream and
  non-stream, OpenAI Responses stream and non-stream with
  `previous_response_id`, Anthropic Messages stream and non-stream with
  thinking explicitly disabled, Ollama `/api/chat` stream and non-stream, and
  Ollama `/api/generate`.
- Exact outputs included `CERULEAN 45` recall and `READY` one-turn responses
  with no flags. DSV4 reported `DSV4BatchGenerator`, `engine_path=dsv4`,
  `block_size=256`, and `paged+dsv4` cache behavior. Hy3 reported
  `SingleBatchGenerator`, paged cache, block-disk writes/hits, and active
  `turboquant_codebook_mpp_nax`.

Installed-app media and image gates were rerun after the clean rebuild:

- `ZAYA1-VL-8B-JANGTQ4`, `Qwen3.6-35B-A3B-JANGTQ-CRACK`, and
  `Nemotron-Omni-Nano-JANGTQ-CRACK` answered image color correctly through
  Chat Completions, Responses, Anthropic Messages, and Ollama.
- `Nemotron-Omni-Nano-JANGTQ-CRACK` also exercised the packaged Omni
  RADIO/Nemotron-H encoder view path.
- `flux-schnell-4bit` and `z-image-turbo-4bit` returned valid PNGs from
  `/v1/images/generations`.
- `qwen-image-edit` returned a valid PNG from `/v1/images/edits`.

Focused `ZAYA1-VL-8B-MXFP4` follow-up:

- All four startup modes were re-run for this one bundle:
  direct/no-continuous-batching, continuous prefix-only, paged L1, and paged
  L2.
- The direct/no-continuous-batching row contained both `CERULEAN` and `45` in
  the strict recall answer, but it did not obey the exact-output instruction.
- The continuous/paged rows consistently returned only `45` for the same
  strict text-only multi-turn recall prompt, even when the focused probe sent
  the same turn-2 prompt with `skip_prefix_cache=true`. That makes this a
  text-generation/API behavior review, not a proven L2 cache-restore failure.
- One-shot text exact recall on the same installed app returned
  `CERULEAN 45`.
- A real blue PNG payload through Chat Completions returned `Blue`.
- A focused text-memory probe after the bypass fix showed Chat Completions
  multi-turn recall works for the same bundle in normal cached, skip-cache, and
  salted forms. All three returned `CERULEAN` and `45`.
- A focused Responses follow-up showed the supported Responses forms work when
  the request uses `instructions` or message-list input: both
  `previous_response_id` chains returned `CERULEAN` and `45`, with cache hits
  recorded. The earlier plain-string Responses prompt without `instructions`
  remains a brittle harness row, not evidence of a cache/runtime failure.
- PP stayed high in the focused rows: about 7.6k to 7.9k prompt tokens/s.
- Cache mechanics were present in the continuous rows:
  `paged+zaya_cca`, typed `zaya_cca_v1` records, block-disk writes in L2, and
  no stream/cache reconstruction exceptions.
- During this follow-up, a real MLLM bypass bug was found: before the fix,
  `skip_prefix_cache=true` and a salted request could still report cached
  tokens on ZAYA-VL because the scheduler did not copy the bypass flag into the
  batch request that owns prefix-cache fetch.
- After the fix and rebuild, installed text bypass proof for
  `ZAYA1-VL-8B-MXFP4` showed one normal repeated request hit
  `paged+zaya_cca` with 8 cached tokens, while both `skip_prefix_cache=true`
  and `cache_salt` produced no `prompt_tokens_details`. Final stats showed
  exactly one cache-hit request and 8 saved tokens, matching the single normal
  repeat only.
- Installed media bypass proof for the same bundle sent the same blue PNG four
  times: first normal, second normal, third with `skip_prefix_cache=true`, and
  fourth with `cache_salt`. All four returned `Blue`. Pixel-cache stats showed
  one hit and one miss with one cached image, while the bypassed/salted image
  requests did not add cache hits or stores.

Installed-app lifecycle gates were rerun after the stream-reset patch:

- Core lifecycle rows passed for Qwen3.6-27B JANG_4M and ZAYA1-VL-8B JANGTQ4:
  warm generation, `/admin/soft-sleep`, `/admin/wake`, generation after soft
  wake, `/admin/deep-sleep`, inference-triggered deep wake, and final cache
  stats.
- Qwen3.6-27B JANG_4M restored paged cache plus SSM companion block-L2 after
  deep wake: `SingleBatchGenerator`, hybrid SSM, disk hit, 8 cached tokens,
  outputs `READY.` on warm/soft/deep rows.
- ZAYA1-VL-8B JANGTQ4 restored typed paged CCA block-L2 after deep wake:
  `zaya_cca_v1`, block-disk hit, 8 cached tokens, MPP/NAX active, no stale
  stream or cache reconstruct failure after the fix.
- Heavy lifecycle rows passed for Hy3-preview JANGTQ2 and
  DeepSeek-V4-Flash JANGTQ_K. Hy3 reported `SingleBatchGenerator`,
  80-layer `TurboQuantKVCache`, block-disk hit, and MPP/NAX active. DSV4
  reported `DSV4BatchGenerator`, `engine_path=dsv4`, block size 256,
  `DeepseekV4Cache` layers, block-disk hit, and MPP/NAX active.
- JIT-enabled lifecycle rows passed for Qwen3.6-27B JANG_4M and ZAYA1-VL-8B
  JANGTQ4. Qwen compiled and warmed successfully. ZAYA attempted JIT, hit the
  known MLX `CacheList` compile incompatibility, rolled back cleanly, and still
  passed cache reuse plus soft/deep wake. That is the intended safe fallback:
  no fake compile claim, no hidden cache bypass.

## Cache/Detection Notes

- DSV4 rows used the DSV4-specific path: `DSV4_LONG_CTX=1`,
  `DSV4_POOL_QUANT=0`, DSV4 chat-template shim, `DeepseekV4Cache` layers,
  block size 256, and `deepseek_v4_v7` nested-state block-L2 serialization.
- MiniMax JANGTQ_K rows logged the mixed projection bit map
  `gate_proj=2`, `up_proj=2`, `down_proj=4` rather than relying on the folder
  name.
- MiniMax/Ling rows retain their parser/tool contracts without family-wide
  repetition floors. Repetition penalty now comes only from request values,
  explicit session/CLI defaults, or bundle metadata.
- JANGTQ acceleration is env-only and defaults to `auto` in the engine. The
  app does not expose or persist a user-facing MPP/NAX switch; packaged app
  session spawn scrubs stale/debug parent env values before starting Python.
- The installed app's bundled Python 3.12 runtime imports
  `jang_tools.turboquant.mpp_nax_kernel`; `panel/scripts/verify-bundled-python.sh`
  hash-gates and import-gates that helper so future package drift blocks the
  release gate.
- Packaged Python release probes now route bytecode/cache writes to a mutable
  evidence directory, with a system-temp fallback when the gate directory is
  intentionally read-only. This keeps import/version probes from mutating a
  signed app bundle.

## Boundaries

- This gate does not claim Kimi-sized bundles or source-sized DSV4 are locally
  tested; they exceed the current 128 GB machine budget for this pass.
  Specifically skipped by budget: `Kimi-K2.6-Small-JANGTQ` (~142.6 GB),
  source `DeepSeek-V4-Flash` (~148.7 GB), and `Kimi-K2.6-JANGTQ_K`
  (~328.1 GB).
- `ZAYA1-VL-8B-MXFP4` remains a harness review row for the exact
  plain-string `READY` prompt. Cache mechanics, prompt-processing speed, image
  input, Chat Completions multi-turn text recall, and Responses
  `instructions`/message-list recall are working in the installed app.
- The matrix is installed-app API/server evidence. Computer Use could not attach
  to the Electron window because macOS denied the automation event, so visual
  click-through proof is still manual/user-side until that permission is fixed.
- Sleep/wake and JIT toggling have representative installed-app proof for the
  core cache families listed above. DSV4 long-context/max-reasoning
  qualification remains a separate lane. A follow-up DSV4 probe confirmed the
  native cache path (`deepseek_v4_v7`, block size 256, `paged+dsv4`, trained
  top-k 6, generic TurboQuant KV off, pool quant off). This gate proves short
  lifecycle, API, and cache behavior for DSV4 JANGTQ_K, not upload-quality
  long max reasoning.

## No-Hidden-Guard Follow-Up

The follow-up cleanup after this matrix removed hidden sampling and acceleration
surfaces rather than adding behavior guards:

- `--jangtq-mpp-nax` was removed from CLI/app UI. Missing env means `auto`;
  explicit `JANGTQ_MPP_NAX=off/on` remains diagnostic-only.
- MiniMax/Ling/DSV4 family sampling floors are gone. The generic fallback is
  neutral (`temperature=0.0`, `top_p=1.0`) when the request, session, and
  bundle declare nothing.
- DSV4 hard n-gram repetition blocking is gone. `VMLX_DSV4_FINALIZER_TOKENS`
  defaults to 0, so caller `max_tokens` is not secretly extended.
- `top_k` and `min_p` from `generation_config.json` or
  `jang_config.chat.sampling_defaults` are wired through app session launch and
  server API kwargs.
- Current repo-local build proof is under
  `panel/release/mac-arm64/vMLX.app`; it was packaged unsigned, then ad-hoc
  signed only so the bundled Python dylibs can run locally.
