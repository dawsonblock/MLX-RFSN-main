# Session J Status — 2026-03-23

## CURRENT BLOCKER: Mistral 4 RoPE Error

**Error:** `[rope] Neither base nor freqs has a value` during prefill
**Model:** Mistral-Small-4-119B-JANG_2L (VLM, model_type=mistral4)
**Root cause investigation:**
- The VLM wrapper (`mlx_vlm/models/mistral3/language.py`) creates `YarnRoPE` with correct `base=10000.0` from `config.rope_parameters["rope_theta"]`
- `YarnRoPE.__call__` passes `base=None, freqs=self._freqs` to `mx.fast.rope` — this should work
- Both base and freqs work in isolation when tested directly
- **Likely cause:** JANG v2 VLM loader (`_load_jang_v2_vlm()` in jang_loader.py) may construct the model with a config that's missing `rope_parameters`, causing `Attention.__init__` to fail during weight repacking
- Need to trace: what config does `_load_jang_v2_vlm()` pass to the model constructor? Does it preserve `text_config.rope_parameters`?

## Session Changes Summary

### Fixes Applied (all in source, bundled, and built app):
1. i18n infrastructure rebuilt (I18nProvider, locale files, TitleBar language picker)
2. All 15 audit fixes intact (server.py: STT/TTS locks, cache clear, speculative sleep, etc.)
3. Mistral 4 reasoning parser added to UI dropdown
4. `[THINK]`/`[/THINK]` client-side fallback parser added
5. `[/THINK]` implicit strip in tool parser fixed
6. `--stream-from-disk` added to buildArgs + RESTART_REQUIRED_KEYS
7. 3 SSD progress patterns added
8. Role alternation fix for Mistral chat template
9. All inline re.sub calls converted to _THINK_STRIP_RE
10. env.d.ts onLoadProgress type added
11. Matrix sections 26-29 added (comprehensive parser/detection tracking)

### All 7 Deep Trace Agents Completed:
- UI components: OK (2 cosmetic)
- JANG+VLM+configs: OK (1 fixed)
- Panel IPC 108 channels: OK (1 fixed)
- Caching stack 8 paths: OK
- server.py 25 endpoints: OK (3 low/known)
- SSD+spec+sleep: OK (2 fixed)
- Reasoning on/off full stack: OK (1 fixed)

### Next Steps:
1. Fix Mistral 4 RoPE error (JANG v2 VLM loader config issue)
2. Run full Python test suite
3. Build, deploy to Mac Studio
4. Push to GitHub (clean history, no tests/notes)
