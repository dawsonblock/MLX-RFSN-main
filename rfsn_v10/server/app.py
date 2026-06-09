"""RFSN v10 FastAPI inference server.

Provides an OpenAI-compatible ``/v1/chat/completions`` endpoint with
Server-Sent Events (SSE) streaming.  The server lazily loads the model
on first request and keeps it in memory for the process lifetime.

Run locally::

    uvicorn rfsn_v10.server.app:app --host 127.0.0.1 --port 8000

Or via the CLI entry-point::

    rfsn-server --model <model-id>

Environment variables
---------------------
RFSN_MODEL_ID
    HuggingFace model ID or local path (required).
RFSN_BACKEND
    ``mlx`` or ``numpy`` (default: ``mlx``).
RFSN_ENABLE_SPARSE_DECODE
    ``true`` or ``false`` (default: ``false``).
RFSN_ENABLE_QUANTIZED_KV
    ``true`` or ``false`` (default: ``true``).
RFSN_MAX_NEW_TOKENS
    Default ``256``.
RFSN_HOST
    Bind host.  Default ``127.0.0.1`` (local-only).  Set ``0.0.0.0`` for LAN.
RFSN_PORT
    Bind port.  Default ``8000``.
RFSN_REQUIRE_API_KEY
    ``true`` or ``false`` (default: ``false``).
RFSN_API_KEY
    Bearer token required when RFSN_REQUIRE_API_KEY=true.
RFSN_MAX_PROMPT_CHARS
    Maximum prompt length in characters.  Default ``24000``.
RFSN_MAX_TOKENS_LIMIT
    Maximum allowed max_tokens per request.  Default ``4096``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from threading import Thread
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .._version import __version__
from ..model_loader import load_model_auto
from ..runtime.generation import GenerationConfig, RFSNGenerator


# ---------------------------------------------------------------------------
# Server settings (read once at startup)
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() == "true"


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


_REQUIRE_API_KEY: bool = _env_bool("RFSN_REQUIRE_API_KEY", False)
_API_KEY: str = os.environ.get("RFSN_API_KEY", "")
_MAX_PROMPT_CHARS: int = _env_int("RFSN_MAX_PROMPT_CHARS", 24000)
_MAX_TOKENS_LIMIT: int = _env_int("RFSN_MAX_TOKENS_LIMIT", 4096)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="", description="Model identifier (informational only)")
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = Field(default=True)
    stop: list[str] | None = Field(default=None)
    repetition_penalty: float = Field(default=1.0, ge=1.0)


class ChatCompletionChoice(BaseModel):
    """Single choice in a chat completion response."""

    index: int = 0
    message: ChatMessage | None = None
    delta: ChatMessage | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response (non-streaming)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RFSN v10 Inference Server",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Lazy-loaded singletons
_model: object | None = None
_tokenizer: object | None = None
_generator: RFSNGenerator | None = None
_model_id_loaded: str = ""
_kv_compression_enabled: bool = False
_sparse_decode_enabled: bool = False


def _get_model_id() -> str:
    model_id = os.environ.get("RFSN_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError(
            "RFSN_MODEL_ID is not set.  "
            "Set it to a HuggingFace model ID, e.g.:\n"
            "  export RFSN_MODEL_ID=mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        )
    return model_id


def _load_generator() -> RFSNGenerator:
    global _model, _tokenizer, _generator
    global _model_id_loaded, _kv_compression_enabled, _sparse_decode_enabled
    if _generator is not None:
        return _generator

    model_id = _get_model_id()
    backend = os.environ.get("RFSN_BACKEND", "mlx").lower()
    _sparse_decode_enabled = _env_bool("RFSN_ENABLE_SPARSE_DECODE", False)
    _kv_compression_enabled = _env_bool("RFSN_ENABLE_QUANTIZED_KV", True)

    _model, _tokenizer = load_model_auto(model_id, backend=backend)
    _generator = RFSNGenerator(
        model=_model,
        tokenizer=_tokenizer,
        enable_sparse_decode=_sparse_decode_enabled,
        enable_quantized_kv=_kv_compression_enabled,
    )
    _model_id_loaded = model_id
    return _generator


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


async def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    if not _REQUIRE_API_KEY:
        return
    if not _API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: RFSN_API_KEY not set but RFSN_REQUIRE_API_KEY=true",
        )
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Health + models endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness/readiness probe.  Returns feature flag status."""
    return {
        "status": "ok",
        "version": __version__,
        "backend": os.environ.get("RFSN_BACKEND", "mlx"),
        "model_loaded": _generator is not None,
        "model_id": _model_id_loaded or None,
        "kv_compression": _kv_compression_enabled,
        "sparse_decode": _sparse_decode_enabled,
        "telemetry": False,
        "host": os.environ.get("RFSN_HOST", "127.0.0.1"),
    }


@app.get("/v1/models")
async def list_models(_auth=Depends(_require_auth)) -> dict:
    """List available models (OpenAI-compatible)."""
    models = []
    if _model_id_loaded:
        models.append({
            "id": _model_id_loaded,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rfsn-v10",
        })
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RFSN v10 Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 640px; margin: 40px auto; padding: 0 20px;
         background: #f5f5f7; color: #1d1d1f; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: #6e6e73; font-size: 0.85rem; margin-bottom: 28px; }
  .card { background: white; border-radius: 12px; padding: 20px 24px;
          margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card h2 { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: .05em; color: #6e6e73; margin: 0 0 12px; }
  .row { display: flex; justify-content: space-between; align-items: center;
         padding: 5px 0; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }
  .row:last-child { border-bottom: none; }
  .label { color: #6e6e73; }
  .val { font-weight: 500; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
           font-size: 0.78rem; font-weight: 600; }
  .badge-ok  { background: #d1fae5; color: #065f46; }
  .badge-off { background: #f3f4f6; color: #6b7280; }
  .badge-on  { background: #dbeafe; color: #1e40af; }
  .badge-warn { background: #fef9c3; color: #92400e; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%;
                display: inline-block; margin-right: 6px; }
  .dot-ok { background: #10b981; }
  .dot-err { background: #ef4444; }
  .footer { text-align: center; color: #9ca3af; font-size: 0.78rem; margin-top: 24px; }
  #last-update { color: #9ca3af; font-size: 0.78rem; text-align: right; }
</style>
</head>
<body>
<h1>RFSN v10</h1>
<p class="subtitle">Local inference dashboard &mdash; refreshes every 3s</p>
<div id="last-update">Loading...</div>
<div class="card" id="card-status">
  <h2>Server</h2>
  <div class="row"><span class="label">Status</span>
    <span id="status-val" class="val">...</span></div>
  <div class="row"><span class="label">Version</span>
    <span id="version-val" class="val">...</span></div>
  <div class="row"><span class="label">Backend</span>
    <span id="backend-val" class="val">...</span></div>
  <div class="row"><span class="label">Host</span>
    <span id="host-val" class="val">...</span></div>
</div>
<div class="card" id="card-model">
  <h2>Model</h2>
  <div class="row"><span class="label">Loaded</span>
    <span id="model-loaded-val" class="val">...</span></div>
  <div class="row"><span class="label">Model ID</span>
    <span id="model-id-val" class="val">...</span></div>
</div>
<div class="card" id="card-features">
  <h2>Features</h2>
  <div class="row"><span class="label">KV Compression</span>
    <span id="kv-val" class="val">...</span></div>
  <div class="row"><span class="label">Sparse Decode</span>
    <span id="sparse-val" class="val">...</span></div>
  <div class="row"><span class="label">Telemetry</span>
    <span id="telemetry-val" class="val">...</span></div>
</div>
<p class="footer">
  <a href="/docs">API Docs</a> &middot;
  <a href="/health">Raw Health JSON</a> &middot;
  <a href="/v1/models">Models</a>
</p>
<script>
function badge(val, trueLabel, trueClass, falseLabel, falseClass) {
  const on = val === true || val === 'true' || val === 'ok';
  return '<span class="badge ' + (on ? trueClass : falseClass) + '">'
       + (on ? trueLabel : falseLabel) + '</span>';
}
async function refresh() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('status-val').innerHTML =
      '<span class="status-dot ' + (d.status==='ok'?'dot-ok':'dot-err') + '"></span>'
      + (d.status || 'unknown');
    document.getElementById('version-val').textContent = d.version || '?';
    document.getElementById('backend-val').textContent = d.backend || '?';
    document.getElementById('host-val').textContent = d.host || '?';
    document.getElementById('model-loaded-val').innerHTML =
      badge(d.model_loaded, 'Yes', 'badge-ok', 'No', 'badge-warn');
    document.getElementById('model-id-val').textContent = d.model_id || '(none)';
    document.getElementById('kv-val').innerHTML =
      badge(d.kv_compression, 'On', 'badge-on', 'Off', 'badge-off');
    document.getElementById('sparse-val').innerHTML =
      badge(d.sparse_decode, 'On (experimental)', 'badge-warn', 'Off', 'badge-off');
    document.getElementById('telemetry-val').innerHTML =
      badge(d.telemetry, 'On', 'badge-on', 'Off', 'badge-off');
    document.getElementById('last-update').textContent =
      'Last updated: ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('status-val').innerHTML =
      '<span class="status-dot dot-err"></span>Unreachable';
    document.getElementById('last-update').textContent = 'Error: ' + e.message;
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> str:
    """Local monitoring dashboard.  Polls /health every 3 seconds."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    _auth=Depends(_require_auth),
) -> StreamingResponse | ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    # Load generator (may raise on bad config)
    try:
        generator = _load_generator()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Build the prompt
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt: str = _tokenizer.apply_chat_template(  # type: ignore[union-attr]
        messages, tokenize=False, add_generation_prompt=True
    )

    # Request limit checks
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Prompt too large ({len(prompt)} chars). "
                f"Limit: {_MAX_PROMPT_CHARS} chars."
            ),
        )
    if request.max_tokens > _MAX_TOKENS_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_tokens ({request.max_tokens}) exceeds configured limit "
                f"({_MAX_TOKENS_LIMIT})."
            ),
        )

    cfg = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop_sequences=request.stop or [],
        stream=request.stream,
    )

    if request.stream:
        return StreamingResponse(
            _sse_stream(generator, prompt, cfg),
            media_type="text/event-stream",
        )

    # Non-streaming: run generation in thread to avoid blocking the event loop
    result = await asyncio.to_thread(
        generator.chat,
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
    )
    return ChatCompletionResponse(
        id=f"rfsn-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model or _model_id_loaded or "rfsn-v10",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason="stop",
            )
        ],
    )


async def _sse_stream(
    generator: RFSNGenerator,
    prompt: str,
    cfg: GenerationConfig,
) -> AsyncIterator[str]:
    """Yield SSE events from a background thread via a queue bridge.

    Running synchronous token generation directly on the event loop would
    block all other requests.  We push tokens from a daemon thread through
    an asyncio.Queue so the event loop stays free between tokens.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    created = int(time.time())
    id_prefix = f"rfsn-{created}"

    def _worker() -> None:
        try:
            for idx, token in enumerate(
                generator.generate(
                    prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty,
                    stop_sequences=cfg.stop_sequences,
                )
            ):
                payload = {
                    "id": f"{id_prefix}-{idx}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_id_loaded or "rfsn-v10",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                loop.call_soon_threadsafe(
                    queue.put_nowait, f"data: {json.dumps(payload)}\n\n"
                )
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    Thread(target=_worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module entry-point (python -m rfsn_v10.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RFSN_HOST", "127.0.0.1")
    port = int(os.environ.get("RFSN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
