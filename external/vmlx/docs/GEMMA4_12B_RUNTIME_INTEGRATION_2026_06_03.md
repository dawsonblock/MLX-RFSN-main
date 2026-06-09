# Gemma 4 12B runtime integration notes

This is the vMLX-facing handoff for the local Gemma 4 12B bundles copied from
`erics-m5-max2.local`.

## Local artifacts

| Bundle | Local path | Format contract |
| --- | --- | --- |
| MXFP4 | `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4` | `weight_format=mxfp4`, `bits=4`, `group_size=32` |
| MXFP8 | `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP8` | `weight_format=mxfp8`, `bits=8`, `group_size=32` |
| JANG_4M | `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` | `weight_format=jang_affine`, attention 8-bit, MLP 4-bit, embed 16-bit |

Each bundle must carry `config.json`, `jang_config.json`,
`generation_config.json`, `tokenizer_config.json`, `processor_config.json`,
`chat_template.jinja`, and `model.safetensors.index.json`.

The source runtime spec copied from the build host is:

`/Users/eric/jang/docs/runtime/2026-06-03-gemma4-12b-unified-runtime-spec.md`

## Executable contract

Run the no-heavy artifact contract before interpreting a Gemma 4 12B runtime
failure:

```bash
.venv/bin/python tests/cross_matrix/run_gemma4_12b_artifact_contract.py
```

Expected artifact:

`build/current-gemma4-12b-artifact-contract-local.json`

This does not load weights. It verifies sidecars, model identity, attention
shape metadata, cache-sensitive fields, generation defaults, chat template
semantics, processor metadata, quantization metadata, and native-MTP absence.

## Architecture contract

Gemma 4 12B is `Gemma4UnifiedForConditionalGeneration` with top-level
`model_type=gemma4_unified`. The text config is `gemma4_unified_text`, vision is
`gemma4_unified_vision`, and audio is `gemma4_unified_audio`.

The model is dense. Do not route it through MoE expert code. On 12B,
`enable_moe_block=false`, `num_experts=null`, `num_kv_shared_layers=0`,
`hidden_size_per_layer_input=0`, and `use_double_wide_mlp=false`.

Use the config context limit: `text_config.max_position_embeddings=131072`.
The model card has conflicting 256K language; the shipped config is the runtime
source of truth until upstream publishes different metadata.

## Attention and cache details that matter

Text backbone:

- 48 decoder layers.
- 5 sliding-attention layers then 1 full-attention layer, repeated 8 times.
- Full attention layers are `[5, 11, 17, 23, 29, 35, 41, 47]`.
- Sliding layers use 16 Q heads, 8 KV heads, `head_dim=256`, and `sliding_window=1024`.
- Full layers use 16 Q heads, 1 global KV head, `global_head_dim=512`.
- `attention_k_eq_v=true` on full layers: there is no `v_proj` tensor there; derive V from raw K before `k_norm`.
- SDPA scale is `1.0`, not `1/sqrt(head_dim)`.
- Gemma 4 RMSNorm is zero-shift: use `x * w`, not Gemma 1/2/3 `x * (1 + w)`.
- Full RoPE is proportional p-RoPE, theta `1e6`, partial rotary factor `0.25`.
- Sliding RoPE is default, theta `10000`.

Cache topology:

- Full layers use unbounded `KVCache`.
- Sliding layers use `RotatingKVCache(max_size=1024, keep=0)`.
- Cache widths are heterogeneous: full layers are 1 x 512 KV, sliding layers are 8 x 256 KV.
- Do not allocate native MTP; the artifact contract requires `mtp_policy=none`.

## Chat, tools, and generation defaults

Generation defaults come from `generation_config.json`:

- `do_sample=true`
- `temperature=1.0`
- `top_p=0.95`
- `top_k=64`
- `eos_token_id=[1, 106, 50]`
- `suppress_tokens=[258883, 258882]`

The parser pair is `reasoning_parser=gemma4` and `tool_parser=gemma4`.

Gemma 4 tools are not JSON-native in the model text. Tool calls use
`<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>` style syntax with
unquoted keys and the `<|"|>` string delimiter. Generic JSON parsing is not
enough.

No-thinking generation should not pre-seed an empty thought block. Thinking is
explicitly represented through `<|channel>thought ... <channel|>` only when the
thinking path is active, and prior model turns should strip thinking from
history.

## Multimodal runtime nuance

The 12B unified checkpoint is encoder-free / early-fusion. It does not ship a
full ViT or audio conformer. Runtime support for image/audio/video must use the
thin `vision_embedder`, `embed_vision`, and `embed_audio` path.

Important image path traps:

- Image patches are rescaled by `1/255`, then the model path applies `2 * (x - 0.5)`.
- Images are not mean/std normalized; video frames are normalized.
- Soft-token counts are dynamic. Do not hardcode 280 image tokens.
- The config declares `use_bidirectional_attention="vision"`; vision-span mask parity still needs live/HF-source proof before claiming image/video quality fixed.

## Conservative serving boundary

The copied artifacts are metadata-validatable. The first live vMLX integration
pass also proves text-only serving, parser behavior, prefix/paged cache, block
L2, and q8 storage-boundary KV cache for all three bundles.

Live proof artifacts:

| Artifact | Scope | Status |
| --- | --- | --- |
| `build/current-gemma4-12b-live-mxfp4-conservative.json` | MXFP4 conservative, no continuous batching, no prefix cache, no KV quant, no native MTP | pass |
| `build/current-gemma4-12b-live-mxfp4-prefix-paged-l2.json` | MXFP4 continuous batching, prefix cache, paged cache, block L2, no KV quant, no native MTP | pass |
| `build/current-gemma4-12b-live-mxfp4-prefix-paged-tq-q8.json` | MXFP4 continuous batching, prefix cache, paged cache, block L2, q8 storage-boundary KV quant, no native MTP | pass |
| `build/current-gemma4-12b-live-mxfp8-jang4m-conservative.json` | MXFP8 conservative pass; superseded JANG_4M conservative failure before `jang_affine` detection fix | open |
| `build/current-gemma4-12b-live-jang4m-conservative.json` | JANG_4M conservative after `jang_affine` detection fix | pass |
| `build/current-gemma4-12b-live-mxfp8-jang4m-prefix-paged-l2.json` | MXFP8 + JANG_4M continuous batching, prefix cache, paged cache, block L2, no KV quant, no native MTP | pass |
| `build/current-gemma4-12b-live-mxfp8-jang4m-prefix-paged-tq-q8.json` | MXFP8 + JANG_4M continuous batching, prefix cache, paged cache, block L2, q8 storage-boundary KV quant, no native MTP | pass |

The live audit probes:

- `/health`
- `/v1/models`
- `/v1/models/{id}/capabilities`
- `/v1/cache/stats`
- `/v1/cache/warm`
- basic OpenAI chat
- multi-turn recall
- Gemma4 reasoning-parser visible answer
- Gemma4 required tool-call parser
- repeated prompt cache behavior

The q8 rows prove `kv_cache_quantization.enabled=true` with native Gemma4
`storage_quantization.enabled=true` for `full_and_sliding_attention_kv`.
They do **not** prove generic live `TurboQuantKVCache`; the artifacts report
`turboquant_kv_cache.enabled=false` and
`native_cache.generic_turboquant_kv.enabled=false`.

## 2026-06-04 remote note reconciliation

I re-read the Gemma 4 12B notes on `erics-m5-max2.local`:

- `/Users/eric/jang/docs/runtime/2026-06-03-gemma4-12b-unified-runtime-spec.md`
- `/Users/eric/wiki/notes/2026-06-03-gemma4-unified-architecture-invariants.md`
- `/Users/eric/.claude/projects/-Users-eric-jang/memory/project_gemma4_12b_unified.md`

The local contract still matches the remote source-of-truth:

- Runtime context limit is the shipped config value `131072`, not the unverified README 256K claim.
- Full-attention layers are `[5, 11, 17, 23, 29, 35, 41, 47]`; those layers have no `v_proj` and derive V from raw K before `k_norm`.
- Gemma 4 RMSNorm is zero-shift `x * w`; do not use Gemma 1/2/3 `x * (1 + w)`.
- Attention scale is `1.0`; full layers use proportional p-RoPE over 128 of 512 dims with exponent denominator 512.
- Native cache is heterogeneous: full layers use unbounded KV and sliding layers use `RotatingKVCache(max_size=1024, keep=0)`.
- `generation_config.json` is authoritative for EOS and suppression: `eos_token_id=[1,106,50]`, `suppress_tokens=[258883,258882]`.
- 12B has no native MTP; speculative decoding requires an external draft model.
- MXFP4, MXFP8, and JANG_4M preserve the 11 multimodal early-fusion tensors as fp16 passthrough with byte-identical tokenizer, processor, generation config, and chat template sidecars.

Remote open items still apply locally and must not be overclaimed:

- Bidirectional vision-span mask exact boundaries are not proven against HF source.
- Vision embedder ordering/pooler details are still runtime-sensitive even though the red-image smoke now passes.
- End-to-end multimodal numerical parity has not been proven broadly; current release evidence is functional live-smoke evidence, not exhaustive HF parity.

Use this conservative recipe when debugging text generation without cache
variables:

```bash
vmlx-engine serve /Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4 \
  --no-continuous-batching \
  --disable-prefix-cache \
  --kv-cache-quantization none \
  --disable-native-mtp \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4
```

Repeat the same flags for MXFP8 and JANG_4M when isolating text-only behavior.

For cache-path validation, use:

```bash
vmlx-engine serve /Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4 \
  --continuous-batching \
  --max-num-seqs 1 \
  --enable-prefix-cache \
  --use-paged-cache \
  --paged-cache-block-size 64 \
  --enable-block-disk-cache \
  --kv-cache-quantization q8 \
  --kv-cache-group-size 64 \
  --disable-native-mtp \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4
```

Repeat for MXFP8 and JANG_4M.

Remaining boundary: image/audio/video quality is not universally production-cleared.
The source-owned `gemma4_unified` runtime now exists locally, and red-image
smokes pass for the default release bundles. JANG_4M audio has a passing speech
fixture row. Uniform MXFP4/MXFP8 audio remains gated because live speech output
is semantically wrong; those bundles must not advertise audio until that row is
fixed or the release scope explicitly excludes uniform-MXFP audio.

## 2026-06-03 cache-warm contract correction

The live audit harness now treats `/v1/cache/warm` as a real endpoint contract, not a status-code-only probe. The request body is `{"prompts": ["..."]}` and the row fails unless the response body has no `error` and reports `warmed >= 1`.

Focused proof artifact: `build/current-gemma4-12b-live-mxfp4-prefix-paged-l2-cachewarm.json`.

Observed result for MXFP4 `prefix_paged_l2`:

- `cache_warm.body.warmed = 1`
- `cache_warm.body.token_counts = [15]`
- `scheduler_cache.hits = 7`
- `scheduler_cache.tokens_saved = 210`
- `scheduler_stats.cache_hit_tokens_by_detail = {"paged+disk": 161, "paged": 49}`
- `block_disk_cache.disk_hits = 12`
- `block_disk_cache.disk_writes = 2`

This is still text-runtime proof only. Image/audio/video remain not production-cleared until a real `gemma4_unified` early-fusion VLM/audio runtime is integrated and live media rows pass.

## 2026-06-03 source Gemma4 Unified runtime integration

vMLX now vendors and registers the source-owned `mlx_vlm.models.gemma4_unified` runtime from the Gemma 4 12B conversion environment. This removes the permanent text-only fallback when the runtime is available.

Changed runtime boundaries:

- `vmlx_engine.models.gemma4_unified_register` registers the vendored package under `mlx_vlm.models.gemma4_unified` and defers to upstream if mlx-vlm later ships it.
- `gemma4_unified` / `gemma4_unified_text` are added to the deferred mlx-vlm prompt registry patch.
- Gemma 4 Unified routes text-only only when the runtime is absent.
- The model registry advertises `runtime_scope=source_gemma4_unified_vlm`, `vl_runtime_available=true`, and `audio_runtime_available=true` when the vendored runtime is present.
- `/v1/cache/warm` now supports the MLLM scheduler path by resolving `processor.tokenizer` and delegating prompt-only prefill through the MLLM batch generator clean-prefix path.

Focused verification:

- `tests/test_gemma4_12b_live_runtime_audit.py` plus focused Gemma4 Unified engine tests: `7 passed`.
- py_compile passed for the touched Gemma4 Unified runtime, registry, server, scheduler, loader, and harness files.
- Conservative text/tool/reasoning row: `build/current-gemma4-12b-live-mxfp4-conservative-unified-runtime.json`, status pass.
- Prefix/paged/L2 row: `build/current-gemma4-12b-live-mxfp4-prefix-paged-l2-unified-runtime.json`, status pass, `cache_warm.warmed=1`, `token_counts=[15]`, cache-hit detail `paged+mixed_swa+disk=230`, `paged+mixed_swa=26`.
- q8 storage-boundary KV row: `build/current-gemma4-12b-live-mxfp4-prefix-paged-tq-q8-unified-runtime.json`, status pass, `kv_cache_quantization.enabled=true`, `bits=8`, `group_size=64`, native storage quantization enabled for full+sliding Gemma4 KV while preserving rotating metadata.
- Image smoke: `build/current-gemma4-12b-live-mxfp4-image-smoke-unified-runtime.json`, status pass, 64x64 red PNG data URL returned visible `Red`, capabilities modalities `text`, `vision`, `video`.

Boundary after this pass: image smoke is now live-proven for MXFP4 only. Audio and video still need dedicated live rows. MXFP8 and JANG_4M still need reruns through the source unified VLM path before broad Gemma 4 12B media production clearance.
