# JANGTQ MPP/NAX Runtime Integration

Date: 2026-05-12

## What Changed

vMLX now has an explicit startup, UI, and health surface for the JANGTQ/MXTQ
MPP/NAX TensorOps lane.

- CLI: `--jangtq-mpp-nax off|auto|on`
- Panel session config: `jangtqMppNax`, default `auto`
- Session launch: emits `--jangtq-mpp-nax <mode>`
- `/health`: reports `acceleration.jangtq_mpp_nax`
- Performance panel: shows `JANGTQ MPP/NAX`

This is not the same as MLX affine quantized matmul. JANGTQ still uses the
TurboQuant codebook path; the new lane is the custom JANGTQ kernel using MPP/NAX
directly. MXFP4/NVFP4/FP4-style bundles are kept on the native MLX affine path.

## Startup Semantics

`auto` is the app default. The CLI sets `JANGTQ_MPP_NAX=auto` before model load.
The jang-tools kernels decide per dispatch shape whether the MPP/NAX path is
faster.

`off` sets `JANGTQ_MPP_NAX=off` and preserves the legacy custom kernels.

`on` sets `JANGTQ_MPP_NAX=1` and forces the path for diagnostics.

## Health Semantics

`quantization.codec` remains the weight codec:

- `turboquant_codebook`: JANGTQ/MXTQ routed codebook tensors
- `affine_quantized_matmul`: MLX affine/MXFP-style tensors
- `full_precision_or_unknown`: no quantized matmul evidence

`acceleration.kernel_type` reports:

- `turboquant_codebook_mpp_nax` when JANGTQ MPP/NAX is active
- `turboquant_codebook` when JANGTQ is present but MPP/NAX is off/unavailable
- `affine_quantized_matmul` for MLX affine/MXFP-style bundles

`acceleration.jangtq_mpp_nax` reports:

- `mode`
- `requested`
- `available`
- `active`
- `uses_mlx_quantized_matmul=false`
- `reason`

## Cache Boundary

This change does not alter prefix cache, paged cache, block-L2, typed ZAYA CCA,
hybrid SSM companion state, DSV4 composite cache, or TurboQuant KV cache policy.
It only controls the JANGTQ weight-matmul dispatch lane.

Cache correctness must still be validated per family. Do not use an MPP/NAX
health win as proof of cache reuse or multi-turn coherency.

## Verification

Focused checks run after the integration patch:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest -q \
  tests/test_engine_audit.py::TestJangTqMppNaxCliPolicy \
  tests/test_engine_audit.py::TestTurboQuantKVTelemetry

cd panel
npx vitest run tests/settings-flow.test.ts
npm run typecheck

cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -B -m py_compile vmlx_engine/server.py vmlx_engine/cli.py
.venv/bin/python -m vmlx_engine.cli serve --help | rg -n "jangtq-mpp-nax|JANGTQ/MXTQ"
JANGTQ_MPP_NAX=auto .venv/bin/python - <<'PY'
from vmlx_engine.server import _jangtq_mpp_nax_runtime_status
print(_jangtq_mpp_nax_runtime_status({"supported": True, "brand": "Apple M5 Max"}))
PY
```

Observed runtime status on this machine:

```text
{'mode': 'auto', 'requested': True, 'available': True, 'active': True,
 'uses_mlx_quantized_matmul': False, 'reason': None}
```

## Bundled Live Gate

After syncing the local `jang-tools` prototype into `panel/bundled-python`, the
bundled server path was tested directly with a real Qwen JANGTQ model:

```sh
cd /Users/eric/mlx/vllm-mlx
panel/bundled-python/python/bin/python3 -B -s -m vmlx_engine.cli serve \
  /Volumes/EricsLLMDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4 \
  --host 127.0.0.1 --port 8137 \
  --max-num-seqs 1 \
  --prefill-batch-size 512 \
  --prefill-step-size 2048 \
  --completion-batch-size 512 \
  --continuous-batching \
  --tool-call-parser qwen \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --use-paged-cache \
  --paged-cache-block-size 64 \
  --max-cache-blocks 1000 \
  --enable-block-disk-cache \
  --block-disk-cache-max-gb 10 \
  --stream-interval 1 \
  --max-tokens 512 \
  --default-temperature 0.0 \
  --default-top-p 1.0 \
  --default-repetition-penalty 1.0 \
  --jangtq-mpp-nax auto
```

Observed `/health` fields:

```json
{
  "acceleration": {
    "kernel_type": "turboquant_codebook_mpp_nax",
    "metal_na_capable": true,
    "metal_na_active_on_host": true,
    "jangtq_mpp_nax": {
      "mode": "auto",
      "requested": true,
      "available": true,
      "active": true,
      "uses_mlx_quantized_matmul": false,
      "reason": null
    }
  },
  "quantization": {
    "codec": "turboquant_codebook",
    "weight_format": "mxtq",
    "mxtq_bits": 4,
    "routed_expert_bits": 4,
    "target_bits": 4
  }
}
```

Real `/v1/chat/completions` output with `enable_thinking=false`:

```text
READY
45
CERULEAN
```

The server log also confirmed Qwen3.6 JANGTQ4 loaded through the JANGTQ VLM fast
path, patched `SwitchGLU`, started paged cache plus block disk cache plus SSM
companion L2, and stored a paged prefix cache entry after generation.

## Packaging Requirement

The vMLX integration assumes the packaged `jang_tools` contains
`jang_tools.turboquant.mpp_nax_kernel` plus the `JANGTQ_MPP_NAX` call sites in
the TurboQuant kernels. If a DMG bundles an older jang-tools wheel/source tree,
health will report `available=false` and fall back to the legacy custom kernels.

The live bundled gate above proves the local development bundle after a manual
`pip install --force-reinstall --no-deps /Users/eric/jang/jang-tools`. It is not
by itself proof that a clean checkout or future DMG contains the same
`jang-tools` files. The release lane must still make the jang-tools helper
module and call-site changes part of the packaged source/wheel.

## 2026-05-12 Follow-Up Packaging Audit

Codex rechecked the JANG source/package boundary after the initial live proof.

Problem found:

- The parent `/Users/eric/jang/.gitignore` ignored `jang-tools/jang_tools/turboquant/`
  and broad `test_*` files, so `mpp_nax_kernel.py`, `mpp_dense_kernel.py`, and
  their tests were hidden from normal Git status.
- A clean checkout would therefore miss the helper modules unless they were
  force-added. That would make vMLX's bundled verifier and runtime health
  surface regress to `available=false` or import failure.

Fix applied locally:

- `.gitignore` now keeps the private TurboQuant scratch area ignored while
  explicitly surfacing:
  - `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py`
  - `jang-tools/jang_tools/turboquant/mpp_dense_kernel.py`
  - `jang-tools/tests/test_turboquant_mpp_nax_kernel.py`
  - `jang-tools/tests/test_turboquant_mpp_dense_kernel.py`

Fresh verification:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest -q \
  tests/test_turboquant_mpp_nax_kernel.py \
  tests/test_turboquant_mpp_dense_kernel.py \
  tests/test_turboquant_kernel_source_contract.py \
  tests/test_turboquant_cache.py
# 47 passed

/Users/eric/mlx/vllm-mlx/.venv/bin/python -B -m py_compile \
  jang_tools/turboquant/mpp_nax_kernel.py \
  jang_tools/turboquant/mpp_dense_kernel.py \
  jang_tools/turboquant/tq_kernel.py \
  jang_tools/turboquant/gather_tq_kernel.py \
  jang_tools/turboquant/fused_gate_up_kernel.py

cd /Users/eric/jang
git diff --check
```

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest -q \
  tests/test_engine_audit.py::TestJangTqMppNaxCliPolicy \
  tests/test_engine_audit.py::TestTurboQuantKVTelemetry
# 75 passed

cd panel
npx vitest run tests/settings-flow.test.ts
# 197 passed
npm run typecheck
```

Bundled boundary:

```sh
cd /Users/eric/mlx/vllm-mlx
VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang/jang-tools \
  panel/scripts/verify-bundled-python.sh
# all critical imports ok, including jang_tools.turboquant.mpp_nax_kernel
```

Release-build guard:

```sh
cd /Users/eric/mlx/vllm-mlx
panel/scripts/bundle-python.sh
# exits 1 before packaging because /Users/eric/jang/jang-tools is tracked-dirty
```

That failure is intentional. It prevents a DMG from silently bundling local
uncommitted JANG runtime edits. The fix is to commit/stash/drop the JANG lane or
point `VMLINUX_JANG_TOOLS_SOURCE` at a clean checkout.

Wheel/sdist package check:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m build \
  --wheel --sdist --outdir /tmp/jang-mpp-package-check
```

Direct archive inspection:

- wheel contains `jang_tools/turboquant/mpp_nax_kernel.py`
- wheel contains `jang_tools/turboquant/mpp_dense_kernel.py`
- sdist contains both runtime modules and both new tests
- wheel intentionally does not install top-level `tests/`

Installed-wheel import check:

```sh
rm -rf /tmp/jang-wheel-import-check
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pip install --no-deps \
  --target /tmp/jang-wheel-import-check \
  /tmp/jang-mpp-package-check/jang-2.5.28-py3-none-any.whl
cd /tmp
PYTHONPATH=/tmp/jang-wheel-import-check \
  /Users/eric/mlx/vllm-mlx/.venv/bin/python - <<'PY'
import sys
sys.path = [p for p in sys.path if not p.endswith('/Users/eric/jang/jang-tools') and p not in ('', '.')]
import jang_tools.turboquant.mpp_nax_kernel as nax
print(nax.__file__)
print(nax.mpp_nax_tensorops_available())
PY
# /tmp/jang-wheel-import-check/jang_tools/turboquant/mpp_nax_kernel.py
# True
```

Live speed/output parity gates:

| Model | Prompt | Mode | Approx PP tok/s | Decode tok/s after first | Output parity |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B-JANGTQ4 | 2084 tokens | off | 1075.0 | 106.8 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 2084 tokens | auto | 1902.3 | 106.7 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 8244 tokens | off | 1056.6 | 101.9 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 8244 tokens | auto | 1858.5 | 101.8 | same prefix |
| ZAYA1-8B-JANGTQ_K | 2092 tokens | off | 2394.9 | 71.9 | same prefix |
| ZAYA1-8B-JANGTQ_K | 2092 tokens | auto | 4292.3 | 72.4 | same prefix |
| MiniMax-M2.7-Small-JANGTQ | 2126 tokens | off | 161.8 | 40.5 | same prefix |
| MiniMax-M2.7-Small-JANGTQ | 2126 tokens | auto | 233.9 | 34.2 | same prefix |
| Hy3-preview-JANGTQ2 | 2099 tokens | off | 111.0 | 17.1 | same visible prefix |
| Hy3-preview-JANGTQ2 | 2099 tokens | auto | 202.9 | 15.8 | same visible prefix |

Notes:

- Qwen and ZAYA show prefill improvement with decode unchanged.
- MiniMax Small shows prefill improvement, but decode in this short run was
  lower under `auto`; treat MiniMax as needing deeper decode profiling before
  calling it an all-around speed win.
- Hy3-preview-JANGTQ2 shows prefill improvement and the same correct visible
  answer prefix, `READY CERULEAN 45`. The standalone probe continued sampling
  after the model emitted EOS, so post-EOS continuation is not counted as a
  runtime loop; app/API generation must stop at EOS.
- `ZAYA1-VL-8B-JANGTQ4` could not be loaded by `jangtq_live_nax_probe.py`
  because that harness uses `load_jangtq_model`, not the VLM loader. This is a
  harness limitation, not evidence of a VLM runtime failure.

Live vMLX API gate:

- Server command used source vMLX plus `PYTHONPATH=/Users/eric/jang/jang-tools`
  and `--jangtq-mpp-nax auto` on Qwen3.6-35B-A3B-JANGTQ4.
- `/health` reported:
  - `acceleration.kernel_type=turboquant_codebook_mpp_nax`
  - `jangtq_mpp_nax.mode=auto`
  - `requested=true`, `available=true`, `active=true`
  - quantization `codec=turboquant_codebook`, `mxtq_bits=4`,
    `routed_expert_bits=4`
  - routing `trained_active_experts=8`, source `config.text_config.num_experts_per_tok`
- `/v1/responses` first turn returned `READY\n45` with no reasoning leak.
- `/v1/responses` second turn with `previous_response_id` returned `CERULEAN`
  and usage reported `cached_tokens=24`, `cache_detail=paged+ssm`.
- `/v1/chat/completions` returned `READY 42` with no `reasoning_content`.
- `/api/chat` returned `OLLAMA_OK`.
- `/v1/messages` returned `ANTHROPIC_OK`.
- Server was stopped after the gate; port 8137 had no listener.

Live vMLX ZAYA1-VL gate:

- Server command used source vMLX plus `PYTHONPATH=/Users/eric/jang/jang-tools`,
  `--is-mllm`, and `--jangtq-mpp-nax auto` on
  `/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4`.
- `/health` reported:
  - `acceleration.kernel_type=turboquant_codebook_mpp_nax`
  - `jangtq_mpp_nax.active=true`
  - model family `zaya1_vl`, `is_mllm=true`
  - `cache_subtype=zaya_cca`
- Single-turn `/v1/responses` returned `17 + 28 = 45`; scheduler telemetry
  showed `prompt_tps=180.8` and `generation_tps=77.7` for that short text-only
  MLLM request.
- Two-turn `/v1/responses` with `previous_response_id` returned `CERULEAN` on
  the second turn and usage reported `cached_tokens=25`,
  `cache_detail=paged+zaya_cca`.
- Server log confirmed JANGTQ VLM fast path, 40 fused `SwitchGLU` instances,
  paged cache, block disk write-through, and typed ZAYA CCA cache records.

JANG test hygiene fixed during this audit:

- `tests/test_inference.py` now inserts its optional fake `mlx_vlm` module via
  `monkeypatch.setitem`, so it cannot poison later real ZAYA-VL imports.
- `pyproject.toml` now constrains pytest discovery to `tests` and
  `jang_tools/dsv4/tests`, so manual smoke script
  `jang_tools/scripts/test_qwen36_python.py` is not executed as a pytest file.
- Fresh full JANG test roots result: `505 passed, 8 skipped`.

Repo-local packaged app gate:

- Built `panel/release/mac-arm64/vMLX.app` from vMLX `300dbe8b` plus JANG
  `2be5c5f`.
- Direct `codesign --deep --sign - vMLX.app` was insufficient: bundled
  `libpython3.12.dylib` remained unsigned at load time even though app-level
  strict verify passed.
- Correct local signing sequence is the one used by `build-and-install.sh`:
  sign bundled Python Mach-O files first, then seal the `.app`.
- After signing 499 bundled Python native files and resealing the app,
  `panel/scripts/release-gate-python-app.py --skip-sleep-wake --skip-gui`
  passed.
- The release gate now hashes `turboquant/mpp_nax_kernel.py` in the packaged
  `jang_tools` tree, so future packages cannot pass with the UI/CLI toggle but
  without the critical JANG runtime module.

Release boundary:

- Do not commit the vMLX `--jangtq-mpp-nax` UI/CLI/health wiring without the
  matching JANG kernel modules and call-site changes.
- Do not claim this as DSV4 correctness work. It only changes JANGTQ weight
  matmul dispatch. DSV4 cache/attention/coherency gates remain separate.

## App-Wide Toggle Follow-Up

The first UI pass exposed `jangtqMppNax` as a per-session setting. That was not
enough for production because users could change the tray/menu expectation while
old session JSON still launched with a stale mode.

The source of truth is now the app setting `jangtq_mpp_nax_mode`, normalized by
`panel/src/shared/jangtqMppNax.ts`.

Wiring:

- Main settings IPC normalizes `auto|off|on`, persists the normalized value,
  broadcasts `runtime:jangtqMppNaxChanged`, and rebuilds the tray menu.
- Tray exposes Auto / Off / On (diagnostic) radio items and writes the same
  app setting.
- Create Session, Server Settings, and legacy Session Settings load and
  subscribe to the same setting so the selector reflects tray/main changes.
- Session startup resolves `--jangtq-mpp-nax` from the app setting, not stale
  per-session JSON.
- Locale keys exist for English, Chinese, Japanese, Korean, and Spanish.

Verification after this follow-up:

```sh
cd /Users/eric/mlx/vllm-mlx/panel
npx vitest run
# 1805 passed

npm run typecheck
npm run build

cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest -q \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards \
  tests/test_engine_audit.py::TestJangTqMppNaxCliPolicy \
  tests/test_engine_audit.py::TestTurboQuantKVTelemetry::test_acceleration_status_does_not_claim_metal_na_for_jangtq \
  tests/test_engine_audit.py::TestTurboQuantKVTelemetry::test_acceleration_status_reports_jangtq_mpp_nax_when_enabled \
  tests/test_cross_matrix_audit_runner.py
# 54 passed

git diff --check
```

Live bundled-source row after the app-wide toggle patch:

```sh
JANGTQ_MPP_NAX=auto VMLX_AUDIT_KILL_EXISTING=1 \
  /Users/eric/mlx/vllm-mlx/panel/bundled-python/python/bin/python3 -B -s -P \
  tests/cross_matrix/run_production_family_audit.py \
  --live \
  --rows zaya_vl_jangtq4 \
  --py /Users/eric/mlx/vllm-mlx/panel/bundled-python/python/bin/python3 \
  --port 8142 \
  --load-timeout 900 \
  --out docs/internal/release-gates/20260512_post_mpp_full_matrix/live_zaya_vl_jangtq4_after_global_toggle_source_build.json
```

Result:

- `PASS`, `14/14` checks.
- Chat Completions, Responses, Anthropic Messages, Ollama chat, streaming, and
  cache stats all returned successfully.
- Repeat prompt reported `cached_tokens=32`,
  `cache_detail=paged+zaya_cca`.
- No blocking runtime log findings and no listener remained on port `8142`.

Signing/notarization were intentionally not run for this source-runtime pass.

## Ling Follow-Up Boundary

The first bundled-source Ling row after the app-wide toggle patch failed
`ling_multilingual_loop_trigger` with a false loop score. The response was a
coherent Cyrillic list, but the old loop scorer treated normal high-frequency
Cyrillic characters as dominant-character repetition.

The loop scorer was hardened so dominant-character evidence now requires a real
same-character run, and periodic evidence only counts when the tail closely
matches an actual repeated pattern. Focused regression now covers both sides:

- exact no-space CJK phrase repetition, emoji runs, and repeated English words
  still score above the loop threshold;
- a coherent Cyrillic five-item list scores below the loop threshold.

After that fix, the bundled-source Ling row no longer failed from loop score
(`loop_score=0.09375`) but it still did not clear the quality row: the Russian
prompt produced mixed Russian/English/Chinese text. Cache and API plumbing
remained healthy in the same run:

- OpenAI Chat Completions, Responses, Anthropic Messages, Ollama chat,
  streaming disconnect, and cache-repeat checks passed.
- Hybrid SSM cache was detected as `paged+ssm`; repeat requests reported
  `cached_tokens=26`.
- Server logs showed SSM companion deferred re-derive and clean companion-store
  for 28 SSM layers.

This means the current Ling finding is a multilingual output-quality boundary,
not evidence that MPP/NAX, prefix/paged cache, block-L2, or the hybrid SSM
companion path regressed.

## Post-MPP Family And Media Gate Notes

Tracked source commit `0f3d3222` plus JANG commit `2be5c5f` were used for the
post-MPP bundled-source matrix. The private raw artifacts live under
`docs/internal/release-gates/20260512_post_mpp_full_matrix/` and
`/tmp/vmlx_family_audit/`.

Rows that passed the standard API/cache family gate:

- ZAYA1-VL JANGTQ4: Chat Completions, Responses, Anthropic, Ollama, streaming
  disconnect, typed `zaya_cca`, and `paged+zaya_cca` repeat hits.
- Qwen3.6 35B-A3B JANGTQ4: hybrid SSM `paged+ssm`, trained active `top_k=8`,
  and `turboquant_codebook_mpp_nax`.
- MiniMax M2.7 Small JANGTQ: trained active `top_k=8`, reasoning extraction,
  tool continuation, auto tool choice, and cache repeat.
- Hy3-preview JANGTQ2: Hunyuan tool parser, Qwen3 reasoning parser,
  Low/High effort contract with no Medium, and `paged+tq` cache repeat.
- Gemma 4 JANG_4M: Gemma tool/reasoning parser, mixed SWA/KV native cache,
  affine dispatch telemetry, and cache repeat.
- Nemotron Omni JANGTQ4: RADIO/Parakeet capability flags, DeepSeek-R1
  reasoning parser, `hybrid_ssm_v1`, SSM companion async re-derive, and
  `paged+ssm` cache repeat.

Direct media probes:

- ZAYA1-VL JANGTQ4 real image Chat/Responses returned `blue`. Media-bearing
  requests skipped token-prefix CCA fetch/store; text-only repeats still hit
  `paged+zaya_cca`.
- Qwen3.6 JANGTQ4 real image, real MP4 video, and Responses image returned
  `blue`; text-only repeats hit `paged+ssm`.

Image API probe:

- Z-Image Turbo mflux generated one 64x64 PNG through
  `/v1/images/generations`.
- `/v1/images/edits` correctly rejected a generation-only model with a clear
  model/endpoint mismatch error.
- The reusable live image matrix runner is now
  `tests/cross_matrix/run_image_model_audit.py`. It starts a real bundled
  Python image server per row, waits for `/health`, calls the OpenAI-compatible
  image endpoint, decodes and verifies the returned PNG, checks for runtime /
  resource-tracker warnings, and then terminates the server before the next row.
- Full local image matrix artifact:
  `/tmp/vmlx_family_audit/image_model_audit_full_local.json`.

Current local image matrix:

| Row | Path | Endpoint | Result |
| --- | --- | --- | --- |
| Flux Schnell 4-bit | `~/.mlxstudio/models/image/FLUX.1-schnell-mflux-4bit` | `/v1/images/generations` | PASS, 64x64 PNG, 3.5 KB decoded, 0.797s request |
| Z-Image Turbo 4-bit | `~/.mlxstudio/models/image/Z-Image-Turbo-mflux-4bit` | `/v1/images/generations` | PASS, 64x64 PNG, 5.6 KB decoded, 0.541s request |
| FLUX.2 Klein 4B 4-bit | `~/.mlxstudio/models/image/FLUX.2-klein-4B-mflux-4bit` | `/v1/images/generations` | PASS, 64x64 PNG, 2.6 KB decoded, 2.747s request |
| Qwen Image 4-bit | `~/.mlxstudio/models/image/qwen-image-mflux-4bit` | `/v1/images/generations` | PASS, 64x64 PNG, 6.7 KB decoded, 2.844s request |
| FLUX.2 Klein 9B full | `~/.mlxstudio/models/image/FLUX.2-klein-9B` | `/v1/images/generations` | PASS, 64x64 PNG, 2.4 KB decoded, 8.970s request |
| Qwen Image Edit full | `~/.mlxstudio/models/image/Qwen-Image-Edit` | `/v1/images/edits` | PASS, instruction edit produced 64x64 PNG, 5.6 KB decoded, 24.502s request |
| Flux Fill full | `~/.mlxstudio/models/image/FLUX.1-Fill-dev` | `/v1/images/edits` | SKIP, local directory is incomplete and only contains license/readme/cache files |

Image acceleration applicability:

- The local image directories do not contain `jang_config.json` or JANGTQ
  `*.tq_packed` weights, so the JANGTQ MPP/NAX TurboQuant lane does not apply
  to these mflux image rows.
- `/health` for the hardened Z-Image probe reported `engine_type=mflux`,
  `kernel_type=full_precision_or_unknown`, and
  `reason=not_affine_quantized_matmul`. That is expected for the mflux image
  path and should not be counted as a JANGTQ speed failure.
- Image speed work belongs to the mflux/native image runtime unless future
  image bundles use JANGTQ/TurboQuant weights.

Follow-up fix after the image probe:

- The first Z-Image run showed a benign but user-facing
  `multiprocessing.resource_tracker` leaked-semaphore warning at shutdown.
- Root cause: the old image-mode warning suppression only filtered the parent
  process, but Python emits this warning from the resource-tracker helper
  process.
- `vmlx_engine.cli._suppress_image_resource_tracker_warning()` now also appends
  `ignore:resource_tracker:UserWarning` to `PYTHONWARNINGS` before mflux/loky
  can start the helper process, while preserving any existing warning filters.
- Regression coverage:
  - focused shutdown tests: 3 passed;
  - image API/engine regression slice: 146 passed;
  - bundled-Python Z-Image generation returned one PNG and the shutdown log had
    no `resource_tracker` / leaked-semaphore warning.

Current boundaries:

- Ling JANGTQ is API/cache healthy but still not multilingual-quality clean on
  the Russian stress row; the visible output mixed Russian, English, and
  Chinese terms after the loop false-positive was removed.
- DSV4 K is tracked separately in `docs/DSV4_RUNTIME_PROGRESS_LOG.md`.
- GUI operation and model download UI still require app-level manual
  verification after the next unsigned build. Live image generation/edit API
  output is now covered for the local models listed above.

## Unsigned Build And Product-Path Boundary After `7cf50544`

After pushing `7cf50544`, the unsigned build lane was rerun:

```sh
cd /Users/eric/mlx/vllm-mlx/panel
npm run build
CSC_IDENTITY_AUTO_DISCOVERY=false npm run package
```

Results:

- `npm run build` passed.
- Bundled Python was rebuilt from source and reported `vmlx_engine 1.5.32`.
- The bundled verifier passed and checked source-matching critical
  `vmlx_engine` files, source-matching critical `jang_tools` files, mflux,
  mlx-vlm Qwen/Gemma modules, Kimi patches, `hadamard_kernel`,
  `fused_gate_up_kernel`, `gather_tq_kernel`, and
  `jang_tools.turboquant.mpp_nax_kernel`.
- `npm run package` passed with `CSC_IDENTITY_AUTO_DISCOVERY=false`, producing
  `panel/release/mac-arm64/vMLX.app`.
- Focused panel source tests passed:
  - settings/chat/image/download slice: 295 passed;
  - API gateway/chat UI/image system/secret startup slice: 279 passed.
- Rebuilt `panel/bundled-python` image runtime probe generated one Z-Image
  Turbo PNG and the shutdown log had no `resource_tracker` / leaked semaphore
  warning:
  `/tmp/vmlx_family_audit/z_image_turbo_rebuilt_bundle_after_resource_tracker_fix.log`.

Intentional boundary:

- `panel/scripts/release-gate-python-app.py --skip-sleep-wake --skip-gui`
  fails on the packaged app because packaged Python cannot load unsigned
  `libpython3.12.dylib`. This is the expected macOS library-validation failure
  for an unsigned packaged bundle, not an engine/runtime import failure.
- Signing/notarization were intentionally skipped in this pass. Actual GUI
  launch of the packaged app requires the signing/ad-hoc-sign lane before
  packaged Python can execute.

## Dev-App GUI Gate CSP Note After `d7e4d83c`

The repo-local Electron dev app initially failed before real UI validation
because the production Content Security Policy blocked the Vite React dev
preamble:

- browser console/log symptom: inline script refused by `script-src 'self'`;
- renderer symptom: `@vitejs/plugin-react can't detect preamble`.

The CSP builder now keeps the production renderer strict while allowing
`script-src 'self' 'unsafe-inline'` only when Electron is running in dev mode
with `ELECTRON_RENDERER_URL` set. The existing Chinese mirror allowances for
README images and download/network access are preserved in the shared builder.

Regression coverage:

- production CSP does not contain `script-src 'self' 'unsafe-inline'`;
- dev renderer CSP contains `script-src 'self' 'unsafe-inline'`;
- `hf-mirror.com` and `modelscope.cn` remain present in image/connect policy.

Live dev-app check:

- launched with isolated user data and remote debugging;
- no Vite preamble/CSP errors in the Electron log;
- DOM rendered the main tabs and Image tab;
- Image tab showed generation rows for Flux Schnell, Z-Image Turbo, Flux Dev,
  FLUX.2 Klein 4B, FLUX.2 Klein 9B, and Qwen Image;
- edit rows for Qwen Image Edit, Flux Kontext, and Flux Fill were present;
- Z-Image Turbo opened with downloaded state, quantization controls, server
  settings, and Start Server action visible.

## MLLM Shutdown Stream Fix After `5d561829`

The live source-row rerun for ZAYA1-VL JANGTQ4 found a shutdown-only warning:

```text
Engine shutdown error (continuing): There is no Stream(gpu, 4) in current thread.
```

Root cause:

- `MLLMBatchGenerator.close()` synchronized the generation stream handle from
  the FastAPI shutdown thread.
- MLX stream handles are thread-local; resolving that handle from the shutdown
  thread can raise before the global wired/cache limits are restored.

Fix:

- `MLLMBatchGenerator.close()` now tries `mx.synchronize(stream)` first, then
  falls back to bare `mx.synchronize()` only for the thread-local
  `There is no Stream(...)` failure before restoring the wired limit.
- `tests/cross_matrix/run_production_family_audit.py` now has
  `VMLINUX_AUDIT_USE_SOURCE_VMLX=1` so live gates can explicitly test the
  source tree while keeping installed-app isolation as the default.

Verification:

- regression tests:
  - `test_batch_generator_close_falls_back_when_stream_is_thread_local`;
  - `test_live_audit_can_opt_into_source_vmlx_imports`;
- source live gate:
  `/tmp/vmlx_family_audit/live_zaya_vl_jangtq4_after_mllm_close_fix.json`;
- ZAYA1-VL JANGTQ4 source row passed 14/14 checks;
- new log `/tmp/vmlx_family_audit/zaya_vl_jangtq4_1778627915.log` has no
  `Engine shutdown error` and no `There is no Stream`;
- typed ZAYA CCA cache still stored and hit paged prefixes;
- first chat generation in that row logged `39 tokens in 0.92s (42.3 tok/s)`.
