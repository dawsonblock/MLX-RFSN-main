# JANG Runtime Detection and M5 Dispatch Notes

Date: 2026-05-15

This note records the production rules for vMLX Python/Electron JANG-family
metadata detection. It exists to keep future plain JANG, MXFP, and JANGTQ
bundles from being routed through the wrong weight-matmul path.

## Codec Rules

vMLX separates weight codec from KV/cache codec:

- Plain JANG / JJQF / MXQ (`format` or `weight_format` set to `jang`, `jjqf`,
  or `mxq`) uses `affine_quantized_matmul`.
- MXFP/NVFP/FP4/FP8 (`weight_format` set to `mxfp4`, `mxfp8`, `nvfp4`,
  `fp4`, or `fp8`) uses `affine_quantized_matmul`.
- JANGTQ/MXTQ (`weight_format` set to `mxtq`/`jangtq`, or a
  `jangtq_runtime.safetensors` sidecar) uses `turboquant_codebook`.

Plain JANG and MXFP are therefore eligible for the native MLX affine
quantized-matmul lane and M5 Metal Neural Accelerator symbols. JANGTQ uses
custom TurboQuant codebook kernels; its M5 acceleration is the internal
JANGTQ MPP/NAX lane, not MLX affine matmul.

## Bit Metadata Rules

The engine reads:

- `profile` from `jang_config.quantization.profile`, then
  `jang_config.profile`, then `config.profile`.
- `target_bits` from JANG quantization metadata, JANGTQ routed bits/profile, or
  `config.quantization.bits`.
- `actual_bits` from `actual_bits`, `actual_bits_per_weight`, or top-level
  `jang_config.actual_bits`.
- `block_size`/`group_size` from JANG quantization metadata for panel and
  health reporting.
- JANGTQ role bits from `mxtq_bits`, including mixed routed projection maps
  such as `gate_proj=2`, `up_proj=2`, `down_proj=4`.
- DSV4 and other layer-specific JANGTQ plans from
  `quantization.routed_experts.bit_plan.routed_layer_bits` when present.

## Dispatch Defaults

- M5 JANGTQ acceleration is internal and env-only. The default is `auto`.
- The app scrubs stale parent environment values for JANGTQ MPP/NAX and DSV4
  diagnostic toggles before spawning the engine.
- The panel does not expose a JANGTQ acceleration user toggle. Users should not
  have to pick a kernel lane.
- JANGTQ top-k override is not emitted by the app. Runtime routing uses the
  model's trained `num_experts_per_tok`/equivalent config value.

## Sampling Defaults

The source of truth for omitted sampling parameters is the bundle metadata,
not hidden family guards:

- Request values win when explicitly provided by the API/UI caller.
- Explicit CLI/session defaults win next.
- Bundle defaults are read from `jang_config.chat.sampling_defaults` and then
  `generation_config.json`.
- Generic fallback is only used when no request, CLI/session, or bundle default
  exists.

All live API routes must pass the request model into the resolver helpers for
temperature, top-p, top-k, min-p, repetition penalty, and max tokens. This keeps
`/v1/messages`, `/api/chat`, `/v1/completions`, `/v1/chat/completions`, and
`/v1/responses` aligned with the same model-scoped bundle defaults. Do not add
sampling floors, repetition blockers, forced EOS, or hidden reasoning demotions
as compensating controls for runtime bugs.

## Verification

Fresh source gates after the detection fix:

- `pytest -q tests/test_engine_audit.py -k 'quantization_status or acceleration_status or routing_status'`
  -> `20 passed`.
- `npm test -- --run tests/jang-quantization-label.test.ts tests/settings-flow.test.ts`
  -> `202 passed`.
- Full focused Python release suite -> `1220 passed, 45 skipped`.
- Full panel test suite -> `1801 passed`.
- Panel typecheck -> passed.
- Packaged bundled Python verifier -> passed with vMLX `1.5.33`,
  JANG `2.5.30`, and MLX `0.31.2`.
- Model-scoped sampling resolver regression:
  `pytest -q tests/test_engine_audit.py::TestServerSamplingResolution tests/test_reasoning_modes.py`
  -> `45 passed`.
- Cache bypass propagation after the sampling resolver fix:
  `pytest -q tests/test_cache_bypass.py` -> `58 passed`.

Packaged live gate:

- Model: `/Users/eric/.cache/huggingface/hub/dealignai/MiniMax-M2.7-JANG_2L-CRACK`
- Codec: `affine_quantized_matmul`
- Kernel: `affine_quantized_matmul`
- M5 Metal NA active: `true`
- Trained active experts: `8`
- Decode: about `50 tok/s`
- Prompt processing rows: about `1229`, `1408`, and `994 tok/s`
- Output: coherent, no loop.
