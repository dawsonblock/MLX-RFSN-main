# Gemma 4 12B max2 runtime notes reconciliation

Date: 2026-06-04
Remote source read: `erics-m5-max2.local:/Users/eric/CRACK_abliteration/gemma4-12b-crack/BUILD_LOG.md`

This note reconciles the max2 Gemma 4 12B build/surgery notes with the current
vMLX release-gate artifacts. It is not a replacement for live vMLX proof.

## Bundles covered

- `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4`
- `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP8`
- `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`
- CRACK-side controls on max2:
  - `/Users/eric/models/dealign.ai/Gemma-4-12B-it-MXFP4-CRACK`
  - `/Users/eric/models/dealign.ai/Gemma-4-12B-it-MXFP8-CRACK`
  - `/Users/eric/models/dealign.ai/Gemma-4-12B-it-JANG_4M-CRACK`

## Architecture facts from max2 notes

- Model family: `gemma4_unified`.
- Text config type: `gemma4_unified_text`.
- Modalities in metadata: text, image, audio, video.
- Runtime shape: encoder-free early-fusion unified model.
- Layers: 48 transformer layers.
- Hidden size: 3840.
- Intermediate size: 15360.
- Hybrid attention ratio: 5 sliding layers to 1 full-attention layer.
- Full-attention layers: `5, 11, 17, 23, 29, 35, 41, 47`.
- Sliding-window layers: all other layers, with `sliding_window=1024`.
- Full-attention layers use K-equals-V sharing: `attention_k_eq_v=true`.
- Sliding head dimension: 256.
- Full/global head dimension: 512.
- Proportional RoPE is used on full-attention layers.
- Final logit softcap: 30.0.
- BOS token id: `2` (`<bos>`).

These facts are pinned locally by
`tests/cross_matrix/run_gemma4_12b_artifact_contract.py`.

## Template, reasoning, and tool-format facts

- Gemma 4 12B uses channel-marker reasoning, not Qwen-style `<think>` tags.
- Reasoning markers include `<|channel>thought`, `<channel|>`, and `<turn|>`.
- The chat template must not pre-open an empty thought channel for normal
  no-thinking generation.
- Tool calls use Gemma 4 native markers such as `<|tool_call>call:`.
- Tool responses use `<|tool_response>`.
- The current template must preserve `tool_choice=required` behavior and
  strip prior assistant thinking when re-rendering conversation history.

The local artifact contract checks these template fragments and rejects a
template that pre-seeds `<|turn>model\n<|channel>thought`.

## Quantization and surgery notes from max2

MXFP4 and MXFP8:

- Uniform quantization path.
- Surgery recipe used dual pathway:
  - `self_attn.o_proj` layers `22..35`, MPOA strength `5.0`.
  - `mlp.down_proj` layers `22..35`, MPOA strength `3.0`.
- Vectors came from projected 2000 EN+ZH structurally mirrored prompts.

JANG_4M:

- Mixed quantization path.
- Attention is 8-bit, MLP is 4-bit, embeddings are fp16 passthrough.
- Surgery recipe was attention-only:
  - `self_attn.o_proj` layers `22..35`, MPOA strength `5.0`.
- `mlp.down_proj` was intentionally not modified because the mixed MLP path
  needs per-tensor quant detection before safe binary patching.

Max2 notes report final local surgery validation:

- MXFP8: base MMLU 69.7%, CRACK MMLU 70.2%, HB-20 20/20.
- MXFP4: base MMLU 64.9%, CRACK MMLU 64.9%, HB-20 20/20.
- JANG_4M: base MMLU 67.1%, CRACK MMLU 69.3%, HB-20 20/20.

The max2 note also corrects the Osaurus validation port: use port `4242`, not
the stale `1337` daemon.

## Current vMLX proof state

Current local artifacts in this worktree:

- `build/current-gemma4-12b-live-all-unified-runtime-fullmatrix-postfix-20260604.json`
  - Status: `pass`.
  - Coverage: Gemma 4 12B MXFP4, MXFP8, and JANG_4M across conservative
    runtime flags, prefix+paged+L2, and prefix+paged+q8 stored-KV modes.
- `build/current-gemma4-12b-media-smoke-all-unified-runtime-postfix-20260604.json`
  - Status: `pass`.
  - Coverage: red-image vision smoke for all three 12B quants.
- `build/current-gemma4-12b-audio-smoke-release-default-gated-mxfp-jang-pass-20260604.json`
  - Status: `pass`.
  - Coverage: release-default audio capability boundary. MXFP4 and MXFP8 do
    not advertise audio and are gated honestly; JANG_4M advertises audio and
    passed the speech fixture plus no-audio contrast row.
- `build/current-gemma4-12b-speed-gate-jang4m-20260604.json`
  - Status: `pass`.
  - JANG_4M default median speed: `46.665 tok/s`.
- `docs/internal/agent-notes/current-gemma4-lfm-step-ling-live-matrix-20260604-proof.json`
  - Records Gemma 4 12B, Gemma 4 26B, Gemma 4 31B, LFM2.5, Step3.7, and Ling
    continuation proof coverage.

Current release boundary:

- Text/runtime/cache/speed proof for Gemma 4 12B is green for the three named
  quants above.
- Image smoke is green.
- Audio capability is green for the release-default boundary: MXFP4/MXFP8 are
  intentionally non-audio/default-gated, and JANG_4M passed live speech audio.
- Video is metadata/runtime-contract covered but not fully production-cleared
  by a dedicated video live media matrix in the current artifacts.
- The broader release objective still remains open on non-Gemma rows,
  especially DSV4 exact-code quality and the full real Electron UI
  cross-family proof.

## Repro scripts from max2 notes

Projected refusal probe:

```bash
~/jang/jang-tools/.venv/bin/python ~/probe_gemma4_12b_v2.py \
  --model ~/models/OsaurusAI/gemma-4-12B-it-MXFP4 \
  --harmful ~/harmful_2k.txt \
  --harmless ~/harmless_2k.txt \
  -o ~/gemma4_12b_mxfp4_projected_2k.safetensors \
  --max-prompts 2000
```

Surgery pattern:

```bash
~/jang/jang-tools/.venv/bin/python ~/surgery_gemma4_12b.py \
  --model <BASE> \
  --vectors ~/gemma4_12b_<quant>_projected_2k.safetensors \
  --layers 22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --strength 5.0 \
  --mode mpoa \
  --target self_attn.o_proj \
  -o <DEST>
```

For MXFP4/MXFP8, add the second MLP pass:

```bash
~/jang/jang-tools/.venv/bin/python ~/surgery_gemma4_12b.py \
  --model <BASE_AFTER_O_PROJ> \
  --vectors ~/gemma4_12b_<quant>_projected_2k.safetensors \
  --layers 22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --strength 3.0 \
  --mode mpoa \
  --target mlp.down_proj \
  -o <DEST>
```

Do not apply the MLP pass to JANG_4M until the surgery script can prove
per-tensor mixed-quant safety for 4-bit MLP tensors.

## Integration requirements for future agents

- Do not downgrade `gemma4_unified` to plain `gemma4` to make old `mlx_vlm`
  import paths work. That was a historical failure mode that produced tensor
  mismatch or garbage output.
- Keep the runtime metadata model-owned: `generation_config.json`,
  `jang_config.json`, tokenizer config, and `chat_template.jinja` must drive
  defaults and parser choice.
- Keep Gemma 4 12B on `gemma4` reasoning and tool parsers, with channel-marker
  stripping, not `<think>` parsing.
- Keep mixed-SWA/full KV cache status visible in `/health`, capabilities, and
  cache stats.
- For JANG_4M speed claims, use the speed-gate artifact and preserve the
  current 45 tok/s floor.
- Do not claim video production clearance from image smoke alone.
