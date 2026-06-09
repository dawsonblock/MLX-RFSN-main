"""RFSN v11 FastAPI inference server.

Provides an OpenAI-compatible ``/v1/chat/completions`` endpoint with
Server-Sent Events (SSE) streaming, continuous batching, and optional
speculative decoding.

Bug fixes vs rfsn_v10/server/app.py
------------------------------------
1. **cfg.__dict__ bug (line 226 in v10)**: ``generator.generate(prompt, **cfg.__dict__)``
   passes Pydantic v2 internal dunder fields to the generator.
   Fix: ``cfg.model_dump()`` returns only declared fields.

2. **Blocking non-streaming path (line 196 in v10)**: ``generator.chat(...)`` is a
   synchronous call inside an async handler — blocks the event loop.
   Fix: ``await asyncio.to_thread(generator.chat, ...)``

3. **Blocking streaming path (line 226 in v10)**: ``for token in generator.generate(...)``
   runs the synchronous generator on the event loop thread.
   Fix: Offload to a daemon thread via a queue; yield tokens across the event loop
   with ``run_in_executor(None, queue.get)`` so other coroutines can run between tokens.

Run locally::

    uvicorn rfsn_v11.server.app:app --host 0.0.0.0 --port 8000

Or via the module CLI::

    python -m rfsn_v11.server

Environment variables
---------------------
RFSN_MODEL_ID
    HuggingFace model ID or local path (required).
RFSN_BACKEND
    ``mlx`` or ``torch`` (default: ``mlx``).
RFSN_ENABLE_SPARSE_DECODE
    ``true`` or ``false`` (default: ``false``).
RFSN_ENABLE_QUANTIZED_KV
    ``true`` or ``false`` (default: ``true``).
RFSN_MAX_NEW_TOKENS
    Default ``256``.
RFSN_TEMPERATURE
    Default ``0.7``.
RFSN_SERVER_HOST
    Bind host (default ``0.0.0.0``).
RFSN_SERVER_PORT
    Bind port (default ``8000``).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="", description="Model identifier (ignored)")
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
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


class ClusterStatusResponse(BaseModel):
    """Cluster status for orchestration probes."""

    version: str
    status: str
    model_loaded: bool
    backend: str


# ---------------------------------------------------------------------------
# GenerationConfig — Pydantic v2 model (model_dump() is safe here)
# ---------------------------------------------------------------------------

class GenerationConfig(BaseModel):
    """Sampling parameters for text generation.

    Uses Pydantic BaseModel so that model_dump() returns only the declared
    fields, never dunder internals (unlike __dict__ on a dataclass-style object).
    """

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    stop_sequences: list[str] = Field(default_factory=list)
    stream: bool = Field(default=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RFSN v11 Inference Server",
    version="11.0.0-alpha.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Lazy-loaded singletons (one per process)
_generator: object | None = None
_tokenizer: object | None = None
_generator_lock = threading.Lock()


def _get_model_id() -> str:
    model_id = os.environ.get("RFSN_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError(
            "RFSN_MODEL_ID is not set.  "
            "Set it to a HuggingFace model ID, e.g.:\n"
            "  export RFSN_MODEL_ID=mlx-community/Llama-3.2-3B-Instruct-4bit"
        )
    return model_id


def _load_generator():
    """Lazy-load the model, tokenizer, and generator singleton (thread-safe)."""
    global _generator, _tokenizer
    with _generator_lock:
        if _generator is not None:
            return _generator, _tokenizer

        model_id = _get_model_id()
        backend = os.environ.get("RFSN_BACKEND", "mlx").lower()
        enable_sparse = os.environ.get("RFSN_ENABLE_SPARSE_DECODE", "false").lower() == "true"
        enable_quant = os.environ.get("RFSN_ENABLE_QUANTIZED_KV", "true").lower() == "true"

        # Import lazily to avoid hard dependency at startup
        from ..model_loader import load_model_auto  # noqa: PLC0415
        from ..runtime.generation import RFSNGenerator  # noqa: PLC0415

        model, tok = load_model_auto(model_id, backend=backend)
        gen = RFSNGenerator(
            model=model,
            tokenizer=tok,
            enable_sparse_decode=enable_sparse,
            enable_quantized_kv=enable_quant,
        )
        _generator = gen
        _tokenizer = tok
        return gen, tok


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe for orchestrators (Kubernetes, etc.)."""
    return {"status": "healthy", "version": "11.0.0-alpha.1"}


@app.get("/v1/cluster/status", response_model=ClusterStatusResponse)
async def cluster_status() -> ClusterStatusResponse:
    """Cluster status endpoint for orchestration and monitoring."""
    return ClusterStatusResponse(
        version="11.0.0-alpha.1",
        status="healthy",
        model_loaded=_generator is not None,
        backend=os.environ.get("RFSN_BACKEND", "mlx"),
    )


# ---------------------------------------------------------------------------
# Chat completions — /v1/chat/completions  (OpenAI compatible)
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        generator, tokenizer = await asyncio.to_thread(_load_generator)
    except (ValueError, RuntimeError) as exc:
        status = 400 if isinstance(exc, ValueError) else 503
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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

    # FIX 2: Non-streaming path — offload synchronous generator.chat() to a thread
    # so it does NOT block the asyncio event loop.
    result = await asyncio.to_thread(
        generator.chat,
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
    )
    response = ChatCompletionResponse(
        id=f"rfsn-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model or "rfsn-v11",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason="stop",
            )
        ],
    )
    return response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Anthropic-style /v1/messages endpoint
# ---------------------------------------------------------------------------

class AnthropicMessage(BaseModel):
    """Anthropic messages API format."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessage):
    """Anthropic-compatible /v1/messages endpoint (delegated to chat_completions)."""
    compat_request = ChatCompletionRequest(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )
    return await chat_completions(compat_request)


# ---------------------------------------------------------------------------
# OpenAI Responses API  /v1/responses
# ---------------------------------------------------------------------------

class ResponsesRequest(BaseModel):
    """OpenAI Responses API format."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="")
    input: str | list[ChatMessage]
    max_output_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


@app.post("/v1/responses")
async def responses_endpoint(request: ResponsesRequest):
    """OpenAI Responses API endpoint (delegated to chat_completions)."""
    if isinstance(request.input, str):
        messages = [ChatMessage(role="user", content=request.input)]
    else:
        messages = request.input

    compat_request = ChatCompletionRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
    )
    return await chat_completions(compat_request)


# ---------------------------------------------------------------------------
# SSE streaming helper
# FIX 1: Replaced blocking for-loop with thread+queue pattern.
# The generator runs in a daemon thread; tokens are delivered across the
# event loop via run_in_executor(None, q.get) so other coroutines can
# run between token yields. This fixes the event-loop blocking bug
# that was present in rfsn_v10/server/app.py:226.
#
# FIX 3: cfg.model_dump() replaces cfg.__dict__ — only declared Pydantic
# fields are passed to generator.generate(), never internal dunder attrs.
# ---------------------------------------------------------------------------

async def _sse_stream(
    generator,
    prompt: str,
    cfg: GenerationConfig,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for streaming tokens.

    The synchronous token generator runs in a daemon thread.
    Each token is placed in a queue and fetched asynchronously so the
    event loop is never blocked.
    """
    created = int(time.time())
    id_prefix = f"rfsn-{created}"
    loop = asyncio.get_event_loop()
    token_queue: queue.Queue = queue.Queue()

    # FIX 3: cfg.model_dump() — never cfg.__dict__
    gen_kwargs = cfg.model_dump(exclude={"stream"})

    def _producer():
        """Run synchronous generator in daemon thread; push tokens to queue."""
        try:
            for token in generator.generate(prompt, **gen_kwargs):
                token_queue.put(token)
        except Exception as exc:
            token_queue.put(exc)
        finally:
            token_queue.put(None)  # sentinel: generation complete

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    idx = 0
    while True:
        # Await queue.get in the thread pool — yields control to event loop
        item = await loop.run_in_executor(None, token_queue.get)

        if item is None:
            break  # sentinel: done
        if isinstance(item, Exception):
            yield f"data: {json.dumps({'error': str(item)})}\n\n"
            break

        payload = {
            "id": f"{id_prefix}-{idx}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "rfsn-v11",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": item},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        idx += 1

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module entry-point  (python -m rfsn_v11.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RFSN_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RFSN_SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
