# Running the Server

## Quickstart

```bash
# Activate your venv first
source .venv/bin/activate

# Start with a small model
rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Server starts at `http://127.0.0.1:8000`.

## Useful URLs

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000/dashboard` | Local monitoring dashboard |
| `http://127.0.0.1:8000/health` | Raw health JSON |
| `http://127.0.0.1:8000/docs` | Swagger API docs |
| `http://127.0.0.1:8000/v1/models` | List loaded models |

## Configuration options

All options can be set as CLI flags or environment variables.

```bash
rfsn-server \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --backend mlx
```

## Sending requests

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 100,
    "stream": false
  }'
```

## Streaming

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
```

## Feature flags

```bash
# Disable KV compression (baseline)
RFSN_ENABLE_KV_COMPRESSION=false rfsn-server --model ...

# Enable sparse decode (experimental, not benchmark-proven)
RFSN_ENABLE_SPARSE_DECODE=true rfsn-server --model ...
```

See `docs/FEATURE_FLAGS.md` for all options.

## Checking config before starting

```bash
rfsn-config-check
```

This validates Python version, env vars, and config without starting the server.

## Checking a running server

```bash
rfsn-health
rfsn-health --url http://192.168.1.10:8000
```
