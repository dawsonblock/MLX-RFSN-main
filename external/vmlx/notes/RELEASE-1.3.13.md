## vMLX v1.3.13

### TurboQuant KV Cache Compression
- Auto-enabled for ALL models (JANG and standard MLX) — ~5x KV cache reduction during generation
- 3-bit compressed KV cache with zero-overhead fill phase
- Hybrid SSM models fully supported (attention layers compressed, SSM layers unchanged)

### Community Contributions
- **@EliasOenal** — HuggingFace repo ID resolution (PR #28), assistant image placeholders (PR #29), `--chat-template-kwargs` CLI flag (PR #30)
- **@ludovicc** — Dependency fixes (PR #32)

### New Features
- Chat settings profiles — save, load, star, and delete named presets
- Custom Jinja chat template override via UI and CLI
- Reasoning Effort control for Mistral 4 in chat settings
- Post-generation Metal cache cleanup (prevents GPU memory hoarding)
- Prompt length admission control (rejects before OOM with clear error)

### Bug Fixes
- Anthropic adapter now merges server-wide `--chat-template-kwargs` defaults
- Anthropic adapter Mistral 4 reasoning_effort auto-mapping
- Anthropic multi-turn think-strip (strips `<think>` blocks when thinking disabled)
- Streaming timeout now uses per-request value (was hardcoded 300s)
- "No limit" timeout maps to 86400s instead of client-side abort
- `bool("false") == True` coercion fix in chat_template_kwargs
- Assistant-role image placeholder tokens (prevents UI crash)
- Post-template image count guard in batch path
- L2 disk cache hybrid guard removed (hybrid SSM models can use disk cache)
- TQ-to-KVCache remap on disk cache fetch
- 17 audit fixes from 11-agent full-system review (203 checks)

### Closed Issues
- #31 — Generation interrupted error (streaming timeout fix)

### Closed PRs
- #28 — HF repo ID resolution (@EliasOenal) — integrated with VLM detection + JANG metadata fixes
- #29 — Assistant image placeholders (@EliasOenal) — integrated with image count guard
- #30 — `--chat-template-kwargs` (@EliasOenal) — integrated with bool coercion + Anthropic forwarding
- #32 — Dependency fixes (@ludovicc) — integrated with Python version corrections

### Feature Requests Implemented
- mlxstudio#30 — Chat session settings profiles
- mlxstudio#29 — Custom Jinja chat templates
