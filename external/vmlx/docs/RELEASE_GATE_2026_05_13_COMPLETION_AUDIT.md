# vMLX 2026-05-13 Completion Audit

This is the tracked completion audit for the installed-app model/cache gate.
Raw local artifacts are intentionally kept under ignored
`docs/internal/release-gates/20260513_installed_app_full_model_gate/`.

## Objective

Prove the installed Python/Electron vMLX app is release-ready for as many local
`~/models` bundles as can fit on this M5 Max, covering:

- JANG, JANGTQ, MXFP4, affine/MLX, VL/media, and image-generation/edit
  detection and startup;
- coherent generation, multi-turn recall, and reasoning controls where
  relevant;
- prompt-processing speed and token-generation speed capture;
- direct, prefix/paged/L2, bypass, sleep/wake, and JIT fallback cache paths;
- Chat Completions, Responses, Anthropic, Ollama, image generation, and image
  edit API surfaces;
- source commit/push state, local installed-app sanity, and secret hygiene.

## Evidence Map

| Requirement | Evidence | Status |
| --- | --- | --- |
| Source pushed | Current working tree has follow-up fixes that are not committed/pushed yet | Pending |
| Repo-local app | `panel/release/mac-arm64/vMLX.app` packaged with `CSC_IDENTITY_AUTO_DISCOVERY=false`, then ad-hoc signed for local bundled-Python execution | Pass |
| Bundle verifier | `panel/scripts/verify-bundled-python.sh` passed after source/bundled parity sync | Pass |
| Local model matrix | `cache_mode_matrix_all85_direct_l2_after_stream_reset/SUMMARY.json`: 52 rows, 26 model IDs | Partial |
| Matrix result | 50 pass, 2 review | Partial |
| Speed capture | PP `152.6..8321.5` tok/s, TG `5.4..62.7` tok/s | Pass |
| Direct and paged/L2 cache | Direct plus paged-L2 rows for every selected model | Pass |
| Core API parity | 7/7 pass in `api_parity_core_post_rebuild_thinking_off` | Pass |
| Heavy API parity | Hy3 and DSV4 pass in `api_parity_heavy_post_rebuild_thinking_off` | Pass |
| VL/media API | ZAYA-VL, Qwen3.6 VL/MoE, and Nemotron Omni image rows pass across Chat, Responses, Anthropic, and Ollama | Pass |
| Image generation | Flux Schnell and Z-Image-Turbo return valid PNGs | Pass |
| Image edit | Qwen image edit returns valid PNG | Pass |
| Sleep/wake | Qwen, ZAYA-VL, Hy3, and DSV4 lifecycle rows pass | Pass |
| JIT toggle/fallback | Qwen JIT pass, ZAYA safe fallback pass | Pass |
| MLLM prefix bypass | Normal repeat hit `paged+zaya_cca`; `skip_prefix_cache` and `cache_salt` did not | Pass |
| MLLM pixel bypass | Normal repeated blue image hit pixel cache; bypass/salt did not add hits/stores | Pass |
| JANGTQ MPP/NAX bundled helper | `panel/scripts/verify-bundled-python.sh` imports `jang_tools.turboquant.mpp_nax_kernel`; installed app import path is Python 3.12 site-packages | Pass |
| JANGTQ MPP/NAX UI surface | User-facing MPP/NAX setting/CLI flag removed; tests assert the toggle is not exposed | Pass |
| JANGTQ MPP/NAX engine wiring | Env-only default `auto`, health/acceleration status, package helper import, and stale parent-env scrub covered by tests | Pass |
| ZAYA1-VL-MXFP4 Chat memory | Cached, skip-cache, and salted Chat Completions all returned `CERULEAN` and `45` | Pass |
| ZAYA1-VL-MXFP4 Responses memory | `instructions` and message-list `previous_response_id` forms returned `CERULEAN` and `45` | Pass |
| ZAYA exact plain-string harness | Exact `READY` plain-string row remains brittle | Review |
| DSV4 short gate | `DSV4BatchGenerator`, `engine_path=dsv4`, block size 256, `paged+dsv4` observed | Pass |
| DSV4 long-context/max-reasoning | `dsv4_jangtqk_long_reasoning_probe/result.json`: native cache passed, long high-effort output failed quality | Review |
| Kimi/source DSV4 | Skipped by RAM budget | Gap |
| GUI visual click-through | Blocked by macOS automation: Computer Use `procNotFound` / Apple event `-1743`; `screencapture` failed | Gap |
| Leftover servers | No `vmlx_engine.cli`, `mflux`, `uvicorn`, or image server left after probes | Pass |
| Secret hygiene | Touched/staged diff scans found no HF/GitHub/OpenAI/private-key patterns | Pass |

## Decision

Do **not** claim every model in `~/models` is utterly proven. The installed app
has strong release-gate evidence for below-budget rows and the current engine
fixes, but the remaining unproven areas are:

- GUI visual click-through until macOS automation permission is fixed or a
  manual screen-recorded pass is supplied;
- Kimi-sized/source DSV4 rows that exceed this 128 GB local pass;
- JANGTQ MPP/NAX applies to JANGTQ/MXTQ TurboQuant codebook kernels. It is not
  the image-generation runtime path; Flux/Z-Image/Qwen-image rows are mflux
  native image paths and were tested for valid PNG output, not JANGTQ MPP/NAX
  acceleration;
- DSV4 long-context/max-reasoning behavior. A follow-up probe showed the
  cache/runtime path is wired correctly (`deepseek_v4_v7`, `paged+dsv4`,
  block size 256, trained top-k 6, generic TurboQuant KV off, pool quant off),
  but upload-quality long max reasoning remains a separate qualification lane;
- exact plain-string `READY` behavior for ZAYA harness prompts.

The current source is suitable for another review/test pass with these
boundaries explicitly stated. It should not be described as pushed or fully
released until the pending working-tree changes are committed and the selected
release artifact is rebuilt through the final release lane.

## Follow-Up Proof After No-Hidden-Guard Cleanup

- Focused Python tests passed: `tests/test_reasoning_modes.py`,
  `tests/test_dsv4_thinking_finalizer.py`, `tests/test_dsv4_paged_cache.py`,
  `tests/test_dsv4_contract_hardening.py`, `tests/test_engine_audit.py`, and
  `tests/test_cross_matrix_audit_runner.py` (`488 passed`).
- Full panel test suite passed (`1806 passed`), plus `npm run typecheck`,
  `npx electron-vite build`, and repo-local `electron-builder --dir`.
- `panel/scripts/verify-bundled-python.sh` passed after syncing local source
  into bundled Python and rewriting console-script shebangs.
- A fresh packaged-app MiniMax JANGTQ row from the repo-local build returned
  coherent number-sequence output with no loop. Health reported
  `turboquant_codebook_mpp_nax`, `JANGTQ_MPP_NAX=auto`,
  `SingleBatchGenerator`, and `TurboQuantKVCache`. Wall-clock token/s was
  `38.33` with bundle sampling and `40.30` with greedy/top-k-off, so the row is
  recorded as `review` against the strict 40 tok/s bundle-sampling threshold,
  not relabeled as green.
