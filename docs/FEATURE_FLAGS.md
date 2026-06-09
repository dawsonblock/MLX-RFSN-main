# Feature Flags

All experimental features are **disabled by default**. Enable them only if you understand the tradeoffs.

## Stable Features

| Flag | Default | Description |
|------|---------|-------------|
| `RFSN_BACKEND` | `mlx` | Backend: `mlx` or `numpy` |
| `RFSN_MODEL_ID` | — | Required: model ID or local path |
| `RFSN_ENABLE_KV_COMPRESSION` | `false` | v10 KV compression — enable after benchmarking on your model/context |
| `RFSN_ENABLE_QUANTIZED_KV` | `false` | **Deprecated alias** for `RFSN_ENABLE_KV_COMPRESSION` (emits warning) |
| `RFSN_ENABLE_SPARSE_DECODE` | `false` | Sparse decode (**not benchmark-proven**) |

## Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `RFSN_HOST` | `127.0.0.1` | Bind host (`0.0.0.0` for LAN) |
| `RFSN_PORT` | `8000` | Bind port |
| `RFSN_REQUIRE_API_KEY` | `false` | Require `Authorization: Bearer <key>` |
| `RFSN_API_KEY` | — | API key (required if REQUIRE_API_KEY=true) |
| `RFSN_MAX_PROMPT_CHARS` | `24000` | Max prompt length in characters |
| `RFSN_MAX_TOKENS_LIMIT` | `4096` | Max allowed max_tokens per request |
| `RFSN_REQUEST_TIMEOUT_SECONDS` | `120` | Per-request timeout |

## Experimental Feature Flags

Enabling any of these emits a warning. They are not validated for production use.

| Flag | Default | Description |
|------|---------|-------------|
| `RFSN_EXPERIMENTAL_QJL` | `false` | QJL score correction (disabled — quality not proven) |
| `RFSN_EXPERIMENTAL_POLAR` | `false` | PolarQuant (disabled — slower on head_dim=64) |
| `RFSN_EXPERIMENTAL_ADAPTIVE` | `false` | Adaptive sparse controller (disabled) |
| `RFSN_ALLOW_EXPERIMENTAL` | `false` | Master gate for all experimental paths |

## Promotion Criteria

A feature moves from experimental to stable only when it passes the shootout gate.
See `docs/CANDIDATE_PROMOTION.md`.
