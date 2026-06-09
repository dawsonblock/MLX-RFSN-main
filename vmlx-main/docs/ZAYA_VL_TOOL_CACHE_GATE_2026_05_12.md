# ZAYA-VL Tool And Cache Gate

Date: 2026-05-12

## What Failed

The packaged `zaya_vl_jangtq4` production row initially failed the live gate:

- ZAYA CCA cache was active, but the live gate expected an old text-scheduler log string.
- MLLM scheduler did not emit the same `Runtime cache layout:` diagnostic as the text scheduler.
- `/v1/responses` auto tool choice did not produce a structured call.
- The first chat turn was coherent but did not reply with the exact literal `noted`.

The tool failure was real. ZAYA-VL uses the `zaya_xml` parser, whose native format is
`<zyphra_tool_call>...</zyphra_tool_call>`. The fallback tool injector only selected
that format when the rendered prompt already contained Zyphra tags. Current ZAYA-VL
tokenizer templates are plain `user:` / `assistant:` templates and ignore `system`
messages, so fallback instructions inserted as a system message were dropped. The live
symptom was a three-token prompt and fabricated prose instead of a function call.

## Fix

- Engine prompt formatting now passes the registry tool parser id into fallback tool
  injection.
- Tool fallback treats `tool_parser_id=zaya_xml` as authoritative and emits Zyphra XML
  even when the tokenizer template itself is plain.
- If a template drops fallback system instructions, the injector retries by prepending
  the parser-native tool instructions to the first user turn.
- MLLM scheduler logs per-layer runtime cache layout and an explicit ZAYA CCA typed
  paged-cache line.
- The production family audit row now validates the ZAYA-VL first turn by the product
  concern: no reasoning leak, clean stop, no repetition, and the seeded blue/cat facts
  are present. The second turn still verifies recall with a `paged+zaya_cca` hit.

## Evidence

Source live gate:

```text
docs/internal/release-gates/20260512_post_mpp_full_matrix/live_zaya_vl_jangtq4_source_green.json
```

Result:

- `zaya_vl_jangtq4`: PASS, 0 failures.
- `responses_auto_tool_choice_structured`: emitted `list_directory` with `{"path": "."}`.
- `chat_turn2_recall_thinking_on`: returned `Blue` with `cached_tokens=33`,
  `cache_detail=paged+zaya_cca`.
- `/health.acceleration.jangtq_mpp_nax.active=true`.
- Runtime log includes `Runtime cache layout:` and `ZAYA/CCA typed paged prefix cache enabled`.

Focused verification:

```bash
.venv/bin/python -m pytest -q tests/test_tool_format.py
.venv/bin/python -m pytest -q tests/test_mllm_scheduler_cache.py tests/test_mllm_cache.py
.venv/bin/python -m pytest -q tests/test_api_surface_parity.py tests/test_responses_history.py tests/test_reasoning_tool_interaction.py tests/test_tool_fallback_injection.py
.venv/bin/python -m pytest -q tests/test_cross_matrix_audit_runner.py
```
