"""RFSN v10 FastAPI inference server.

Provides an OpenAI-compatible ``/v1/chat/completions`` endpoint with
Server-Sent Events (SSE) streaming.  The server lazily loads the model
on first request and keeps it in memory for the process lifetime.

Run locally ::

    uvicorn rfsn_v10.server.app:app --host 0.0.0.0 --port 8000

Or via the module CLI ::

    python -m rfsn_v10.server

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
"""
from __future__ import annotations

import json
import os
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..model_loader import load_model_auto
from ..runtime.generation import GenerationConfig, RFSNGenerator


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RFSN v10 Inference Server",
    version="10.0.0-beta",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Lazy-loaded singletons
_model: object | None = None
_tokenizer: object | None = None
_generator: RFSNGenerator | None = None


def _get_model_id() -> str:
    """Return the model ID from environment or raise a clear error."""
    model_id = os.environ.get("RFSN_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError(
            "RFSN_MODEL_ID is not set.  "
            "Set it to a HuggingFace model ID, e.g.:\n"
            "  export RFSN_MODEL_ID=mlx-community/Llama-3-8B-Instruct-4bit"
        )
    return model_id


def _load_generator() -> RFSNGenerator:
    """Lazy-load the model, tokenizer, and generator singleton."""
    global _model, _tokenizer, _generator
    if _generator is not None:
        return _generator

    model_id = _get_model_id()
    backend = os.environ.get("RFSN_BACKEND", "mlx").lower()
    enable_sparse = (
        os.environ.get("RFSN_ENABLE_SPARSE_DECODE", "false").lower()
        == "true"
    )
    enable_quant = (
        os.environ.get("RFSN_ENABLE_QUANTIZED_KV", "true").lower()
        == "true"
    )

    _model, _tokenizer = load_model_auto(
        model_id, backend=backend
    )
    _generator = RFSNGenerator(
        model=_model,
        tokenizer=_tokenizer,
        enable_sparse_decode=enable_sparse,
        enable_quantized_kv=enable_quant,
    )
    return _generator


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe for orchestrators (Kubernetes, etc.)."""
    return {"status": "healthy", "version": "10.0.0-beta"}


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
) -> StreamingResponse:
    """OpenAI-compatible chat completions endpoint with SSE streaming."""
    try:
        generator = _load_generator()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Build the prompt from messages
    messages = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]
    prompt = _tokenizer.apply_chat_template(  # type: ignore[union-attr]
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

    # Non-streaming path
    result = generator.chat(
        prompt,  # already a full prompt string
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
    )
    response = ChatCompletionResponse(
        id=f"rfsn-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model or "rfsn-v10",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason="stop",
            )
        ],
    )
    return response  # type: ignore[return-value]


async def _sse_stream(
    generator: RFSNGenerator,
    prompt: str,
    cfg: GenerationConfig,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for streaming tokens."""
    created = int(time.time())
    id_prefix = f"rfsn-{created}"
    for idx, token in enumerate(generator.generate(prompt, **cfg.__dict__)):
        payload = {
            "id": f"{id_prefix}-{idx}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "rfsn-v10",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    # Final done event
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module entry-point (python -m rfsn_v10.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RFSN_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RFSN_SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
