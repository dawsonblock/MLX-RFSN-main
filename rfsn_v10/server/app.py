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
RFSN_ENABLE_KV_COMPRESSION
    ``true`` or ``false`` (default: ``false``). Deprecated alias:
    ``RFSN_ENABLE_QUANTIZED_KV`` still accepted but emits a warning.
    Even when ``true``, the adapter is disabled unless
    ``RFSN_EXPERIMENTAL_KV_UNSAFE=1`` is also set.
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
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .._version import __version__
from ..config import RFSNConfig
from ..model_loader import load_model_auto
from ..runtime.generation import GenerationConfig, RFSNGenerator

# ---------------------------------------------------------------------------
# ServerState: all mutable per-app state in one object
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    """Encapsulates all mutable server state for a single app instance."""

    cfg: RFSNConfig

    # Lazy-loaded singletons
    model: object | None = None
    tokenizer: object | None = None
    generator: RFSNGenerator | None = None
    model_id_loaded: str = ""
    kv_compression_enabled: bool = False
    sparse_decode_enabled: bool = False

    # Live metrics
    metrics: dict = field(default_factory=lambda: {
        "requests_total": 0,
        "last_latency_ms": None,
        "last_decode_tps": None,
        "last_error": None,
        "model_loaded": False,
        "kv_compression": False,
    })

    # Concurrency gate (created in __post_init__)
    semaphore: asyncio.Semaphore = field(init=False)
    _semaphore_slots: int = field(init=False)

    def __post_init__(self) -> None:
        self._semaphore_slots = self.cfg.server.max_concurrent_requests
        self.semaphore = asyncio.Semaphore(self._semaphore_slots)

    def get_model_id(self) -> str:
        model_id = self.cfg.model.id.strip()
        if not model_id:
            raise RuntimeError(
                "RFSN_MODEL_ID is not set.  "
                "Set it to a HuggingFace model ID, e.g.:\n"
                "  export RFSN_MODEL_ID="
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
            )
        return model_id

    def load_generator(self) -> Any:
        if self.generator is not None:
            return self.generator

        model_id = self.get_model_id()
        backend = self.cfg.backend.name.lower() or None
        self.sparse_decode_enabled = (
            self.cfg.runtime.sparse_decode_enabled
        )
        self.kv_compression_enabled = (
            self.cfg.runtime.enable_kv_compression
        )

        self.model, self.tokenizer = load_model_auto(
            model_id, backend=backend,
        )

        if self.kv_compression_enabled:
            # Safety guard: the incremental quantized cache adapter is
            # experimental and has known correctness issues (unbounded
            # blocks, corrupt trim, O(T²) dense reconstruction).
            # Require an explicit dev opt-in beyond the env flag.
            if os.getenv("RFSN_EXPERIMENTAL_KV_UNSAFE", "false").lower() != "true":
                import warnings
                warnings.warn(
                    "RFSN_ENABLE_KV_COMPRESSION is set but the adapter is "
                    "not ready for production. Set RFSN_EXPERIMENTAL_KV_UNSAFE=1 "
                    "to enable anyway. Falling back to dense KV cache.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.kv_compression_enabled = False
            else:
                # Use new incremental quantized cache adapter
                from rfsn_v10.integrations.mlx_lm_adapter.generator import RfsnMLXGenerator
                self.generator = RfsnMLXGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    num_layers=len(getattr(self.model, "layers", [])),
                    key_bits=8,
                    value_bits=5,
                    group_size=64,
                    staging_capacity=64,
                    dense_residual_window=0,
                    packed_reference=self.cfg.runtime.packed_reference,
                    strict=self.cfg.runtime.strict_packed_mode,
                )

        if not self.kv_compression_enabled:
            self.generator = RFSNGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                enable_sparse_decode=self.sparse_decode_enabled,
                enable_quantized_kv=False,
            )
        self.model_id_loaded = model_id
        self.metrics["model_loaded"] = True
        self.metrics["kv_compression"] = self.kv_compression_enabled
        return self.generator


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
# Dashboard HTML (static)
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
  <div class="row"><span class="label">API Key Required</span>
    <span id="apikey-val" class="val">...</span></div>
  <div class="row"><span class="label">Max Concurrent</span>
    <span id="concurrent-val" class="val">...</span></div>
</div>
<div class="card" id="card-metrics">
  <h2>Performance</h2>
  <div class="row"><span class="label">Requests Total</span>
    <span id="req-total-val" class="val">...</span></div>
  <div class="row"><span class="label">Last Latency</span>
    <span id="latency-val" class="val">...</span></div>
  <div class="row"><span class="label">Last Decode TPS</span>
    <span id="tps-val" class="val">...</span></div>
  <div class="row"><span class="label">Last Error</span>
    <span id="last-error-val" class="val">...</span></div>
</div>
<p class="footer">
  <a href="/docs">API Docs</a> &middot;
  <a href="/health">Raw Health JSON</a> &middot;
  <a href="/metrics">Metrics JSON</a> &middot;
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
    const [rh, rm] = await Promise.all([fetch('/health'), fetch('/metrics')]);
    const d = await rh.json();
    const m = await rm.json();
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
    document.getElementById('apikey-val').innerHTML =
      badge(d.api_key_required, 'Yes', 'badge-on', 'No', 'badge-off');
    document.getElementById('concurrent-val').textContent =
      d.max_concurrent_requests || '1';
    document.getElementById('req-total-val').textContent = m.requests_total ?? '0';
    document.getElementById('latency-val').textContent =
      m.last_latency_ms != null ? m.last_latency_ms + ' ms' : '—';
    document.getElementById('tps-val').textContent =
      m.last_decode_tps != null ? m.last_decode_tps + ' tok/s' : '—';
    document.getElementById('last-error-val').textContent =
      m.last_error || '—';
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


def _get_state(request: Request) -> ServerState:
    """Retrieve ServerState from the running app."""
    return request.app.state.server  # type: ignore[return-value]


def create_app(config: RFSNConfig | None = None) -> FastAPI:
    """Create a fully configured FastAPI app instance.

    Parameters
    ----------
    config : RFSNConfig | None
        Server configuration.  Falls back to ``RFSNConfig.from_env()``
        when *None* (production default).

    Returns
    -------
    FastAPI
        Ready-to-run app with all routes registered.
    """
    if config is None:
        config = RFSNConfig.from_env()

    srv_cfg = config.server
    state = ServerState(cfg=config)

    application = FastAPI(
        title="RFSN v10 Inference Server",
        version=__version__,
        docs_url="/docs" if srv_cfg.enable_docs else None,
        redoc_url="/redoc" if srv_cfg.enable_docs else None,
    )
    application.state.server = state  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Auth dependency (closure over state)
    # ------------------------------------------------------------------
    async def require_auth(
        credentials: (
            HTTPAuthorizationCredentials | None
        ) = Depends(_security),
    ) -> None:
        if not srv_cfg.require_api_key:
            return
        if not srv_cfg.api_key:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Server misconfigured: RFSN_API_KEY not set "
                    "but RFSN_REQUIRE_API_KEY=true"
                ),
            )
        if (
            credentials is None
            or credentials.credentials != srv_cfg.api_key
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @application.get("/health")
    async def health(request: Request) -> dict:
        s = _get_state(request)
        return {
            "status": "ok",
            "version": __version__,
            "backend": s.cfg.backend.name or "auto",
            "model_loaded": s.generator is not None,
            "model_id": s.model_id_loaded or None,
            "kv_compression": s.kv_compression_enabled,
            "sparse_decode": s.sparse_decode_enabled,
            "telemetry": False,
            "host": srv_cfg.host,
            "api_key_required": srv_cfg.require_api_key,
            "max_concurrent_requests": (
                srv_cfg.max_concurrent_requests
            ),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @application.get("/metrics")
    async def metrics(
        request: Request,
        _auth=Depends(require_auth),
    ) -> dict:
        return dict(_get_state(request).metrics)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    @application.get("/v1/models")
    async def list_models(
        request: Request,
        _auth=Depends(require_auth),
    ) -> dict:
        s = _get_state(request)
        models = []
        configured_model_id = s.cfg.model.id.strip()
        if configured_model_id:
            models.append({
                "id": configured_model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rfsn-v10",
                "loaded": s.generator is not None,
            })
        return {"object": "list", "data": models}

    # ------------------------------------------------------------------
    # Dashboard (conditionally mounted)
    # ------------------------------------------------------------------
    if srv_cfg.enable_dashboard:
        @application.get(
            "/dashboard",
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        async def dashboard(
            _auth=Depends(require_auth),
        ) -> str:
            return _DASHBOARD_HTML

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------
    @application.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        chat_request: ChatCompletionRequest,
        request: Request,
        _auth=Depends(require_auth),
    ) -> StreamingResponse | ChatCompletionResponse:
        s = _get_state(request)
        timeout_s = float(srv_cfg.request_timeout_seconds)
        max_prompt = srv_cfg.max_prompt_chars
        max_tok_limit = srv_cfg.max_tokens_limit

        # Concurrency gate: try to acquire without blocking
        if s.semaphore.locked():
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Server busy: max "
                    f"{srv_cfg.max_concurrent_requests} "
                    "concurrent request(s). Retry shortly."
                ),
            )

        # Load generator (may raise on bad config)
        try:
            generator = s.load_generator()
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc),
            ) from exc

        # Build prompt (guard against tokenizers without apply_chat_template)
        messages = [
            {"role": m.role, "content": m.content}
            for m in chat_request.messages
        ]
        try:
            prompt = str(
                s.tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    messages, tokenize=False, add_generation_prompt=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            # Fallback for tokenizers without chat template
            parts = []
            for m in messages:
                parts.append(f"{m['role']}: {m['content']}")
            parts.append("assistant:")
            prompt = "\n".join(parts)

        # Limit checks
        if len(prompt) > max_prompt:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Prompt too large ({len(prompt)} chars). "
                    f"Limit: {max_prompt} chars."
                ),
            )
        if chat_request.max_tokens > max_tok_limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"max_tokens ({chat_request.max_tokens}) "
                    f"exceeds configured limit "
                    f"({max_tok_limit})."
                ),
            )

        cfg = GenerationConfig(
            max_new_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            repetition_penalty=chat_request.repetition_penalty,
            stop_sequences=chat_request.stop or [],
            stream=chat_request.stream,
        )

        s.metrics["requests_total"] += 1

        if chat_request.stream:
            return StreamingResponse(
                _sse_stream(s, prompt, cfg, timeout_s),
                media_type="text/event-stream",
            )

        # Non-streaming
        t_start = time.monotonic()
        async with s.semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        generator.chat,
                        prompt,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        repetition_penalty=cfg.repetition_penalty,
                        stop_sequences=cfg.stop_sequences,
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                s.metrics["last_error"] = "Generation timed out"
                raise HTTPException(
                    status_code=504,
                    detail="Generation timed out",
                )
            except Exception as exc:
                s.metrics["last_error"] = str(exc)[:200]
                raise
        elapsed_ms = (time.monotonic() - t_start) * 1000
        s.metrics["last_latency_ms"] = round(elapsed_ms, 1)
        s.metrics["last_error"] = None

        # Token-based TPS (not word-based)
        decode_tokens = result.decode_token_count if hasattr(result, "decode_token_count") else 0
        if decode_tokens == 0:
            # Fallback: count tokens via tokenizer
            try:
                decode_tokens = len(s.tokenizer.encode(result.text))
            except Exception:
                decode_tokens = 0
        if elapsed_ms > 0 and decode_tokens > 0:
            s.metrics["last_decode_tps"] = round(
                decode_tokens / (elapsed_ms / 1000), 1,
            )

        # Determine correct finish reason
        finish_reason = "stop"
        if hasattr(result, "finish_reason") and result.finish_reason:
            finish_reason = result.finish_reason
        elif decode_tokens >= cfg.max_new_tokens:
            finish_reason = "length"

        return ChatCompletionResponse(
            id=f"rfsn-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=(
                chat_request.model
                or s.model_id_loaded
                or "rfsn-v10"
            ),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", content=result.text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    return application


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

async def _sse_stream(
    state: ServerState,
    prompt: str,
    cfg: GenerationConfig,
    timeout_s: float,
) -> AsyncIterator[str]:
    """Yield SSE events from a background thread via a queue bridge.

    Running synchronous token generation directly on the event loop would
    block all other requests.  We push tokens from a daemon thread through
    an asyncio.Queue so the event loop stays free between tokens.

    The semaphore is held for the duration of the stream so that
    streaming requests participate in the concurrency gate.
    """
    generator = state.generator
    assert generator is not None

    # Concurrency gate: acquire semaphore (held for stream duration)
    await state.semaphore.acquire()

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    created = int(time.time())
    id_prefix = f"rfsn-{created}"
    deadline = time.monotonic() + timeout_s
    stop_event = threading.Event()

    def _worker() -> None:
        finish_reason = "stop"
        accumulated_text = ""
        try:
            tokens_generated = 0
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
                if stop_event.is_set():
                    return
                tokens_generated += 1

                # Accumulate text and check stop sequences on the full buffer
                # so that multi-token stop sequences are caught correctly.
                accumulated_text += token
                truncated_token = token
                if cfg.stop_sequences:
                    for seq in cfg.stop_sequences:
                        pos = accumulated_text.find(seq)
                        if pos != -1:
                            # Determine how much of the current token to keep
                            text_before = accumulated_text[:pos]
                            already_yielded_len = len(accumulated_text) - len(token)
                            if already_yielded_len < len(text_before):
                                truncated_token = text_before[already_yielded_len:]
                            else:
                                truncated_token = ""
                            finish_reason = "stop"
                            stop_event.set()
                            break
                    if stop_event.is_set():
                        if truncated_token:
                            payload = {
                                "id": f"{id_prefix}-{idx}",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_id_loaded or "rfsn-v10",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": truncated_token},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                f"data: {json.dumps(payload)}\n\n",
                            )
                        break

                payload = {
                    "id": f"{id_prefix}-{idx}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_id_loaded or "rfsn-v10",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    f"data: {json.dumps(payload)}\n\n",
                )

            # Determine finish reason
            if tokens_generated >= cfg.max_new_tokens:
                finish_reason = "length"
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            # Send final chunk with finish_reason
            final_payload = {
                "id": f"{id_prefix}-final",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": state.model_id_loaded or "rfsn-v10",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            loop.call_soon_threadsafe(
                queue.put_nowait,
                f"data: {json.dumps(final_payload)}\n\n",
            )
            loop.call_soon_threadsafe(queue.put_nowait, None)

    Thread(target=_worker, daemon=True).start()

    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=remaining,
                )
            except asyncio.TimeoutError:
                timed_out = True
                break
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        if timed_out:
            error_payload = {
                "id": f"{id_prefix}-error",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": (
                    state.model_id_loaded or "rfsn-v10"
                ),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "length",
                    },
                ],
                "error": "Generation timed out",
            }
            yield (
                f"data: {json.dumps(error_payload)}\n\n"
            )

        yield "data: [DONE]\n\n"
    finally:
        stop_event.set()
        state.semaphore.release()


# ---------------------------------------------------------------------------
# Module-level default app (for uvicorn / CLI)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Module entry-point (python -m rfsn_v10.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RFSN_HOST", "127.0.0.1")
    port = int(os.environ.get("RFSN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
