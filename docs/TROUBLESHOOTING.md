# Troubleshooting

Common problems and fixes for RFSN v10.

---

## Server won't start

### `Configuration error: LAN mode (host=0.0.0.0) requires API key enforcement`

You tried to bind to all interfaces without an API key. Fix:

```bash
export RFSN_REQUIRE_API_KEY=true
export RFSN_API_KEY="your-secret"
rfsn-server --host 0.0.0.0
```

### `Configuration error: ServerConfig.api_key must be set when require_api_key=True`

`RFSN_REQUIRE_API_KEY=true` but no key was provided:

```bash
export RFSN_API_KEY="your-secret"
```

### `Model is required`

No model specified:

```bash
rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
# or
export RFSN_MODEL_ID="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

### `uvicorn is not installed`

```bash
pip install 'mlx-rfsn[production]'
# or
pip install uvicorn
```

---

## HTTP errors

### 401 Unauthorized

The request is missing or has an incorrect `Authorization` header:

```bash
curl ... -H "Authorization: Bearer your-api-key-here"
```

### 413 Prompt Too Large

Your prompt exceeds `RFSN_MAX_PROMPT_CHARS` (default 32768). Either shorten
the prompt or increase the limit:

```bash
export RFSN_MAX_PROMPT_CHARS=65536
```

### 429 Too Many Requests

All generation slots are busy (`RFSN_MAX_CONCURRENT_REQUESTS`, default 1).
Wait and retry, or increase the limit:

```bash
export RFSN_MAX_CONCURRENT_REQUESTS=2
```

> Note: each concurrent generation shares Mac unified memory. Setting this
> above 2 on a 16 GB device is not recommended.

### 504 Gateway Timeout

Generation exceeded `RFSN_REQUEST_TIMEOUT_SECONDS` (default 120s):

```bash
export RFSN_REQUEST_TIMEOUT_SECONDS=300
```

Or reduce `max_tokens` in the request.

---

## Model loading

### `mlx` not found / no module named `mlx`

MLX is only available on Apple Silicon Macs running macOS 13.3+. See
`docs/INSTALL_MAC_MLX.md` for the full install guide.

### Model download is slow / fails

The model is downloaded from HuggingFace on first use. Ensure you have:

1. A stable internet connection
2. Enough disk space (models range from 300 MB to several GB)
3. HuggingFace CLI auth if the model is gated:

   ```bash
   huggingface-cli login
   ```

### Wrong model architecture

RFSN v10 currently supports causal LM architectures (Qwen2, Llama, Mistral,
Gemma). Encoder models and multimodal models are not supported.

---

## Performance

### Generation is slow

1. Make sure the backend is `mlx` (default), not `numpy`:

   ```bash
   echo $RFSN_BACKEND  # should be empty or "mlx"
   ```

2. Make sure no other GPU/ML workloads are competing for memory.

3. On first generation, MLX JIT-compiles the compute graph. Subsequent
   requests are faster.

### KV compression makes things slower

This is expected on short prompts. KV compression benefits appear at longer
context lengths. Run the shootout to measure on your model:

```bash
rfsn-bench --categories long_context
```

---

## Dashboard / browser

### Dashboard shows "Unreachable"

The server is not running or the port is wrong. Check:

```bash
curl http://127.0.0.1:8000/health
```

### `/metrics` shows all `null`

No requests have been processed since server start. Send at least one
`/v1/chat/completions` request first.

---

## Deprecation warnings

### `RFSN_ENABLE_QUANTIZED_KV is deprecated`

Rename to the canonical flag:

```bash
# old
RFSN_ENABLE_QUANTIZED_KV=true

# new
RFSN_ENABLE_KV_COMPRESSION=true
```

The old name still works but prints a `DeprecationWarning`.

---

## Getting more help

1. Run `rfsn-config-check` to dump the resolved config and spot misconfiguration.
2. Check server logs (stdout) for tracebacks.
3. Open an issue referencing the exact error message and your OS + Python version.
