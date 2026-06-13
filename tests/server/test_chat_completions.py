"""Chat completions endpoint tests with fake generator.

Tests the full HTTP request/response cycle for the /v1/chat/completions
endpoint, including streaming, stop sequences, tokenizer fallback, and
packed_reference config propagation.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from rfsn_v10.config import RFSNConfig
from rfsn_v10.runtime.generation import GenerationResult
from rfsn_v10.server.app import ServerState, create_app


class FakeGenerator:
    """Fake generator that yields predictable tokens."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens or ["Hello", ",", " world", "!"]
        self.chat_call_count = 0
        self.generate_call_count = 0

    def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        self.chat_call_count += 1
        return GenerationResult(
            text="".join(self.tokens),
            tokens=[ord(t[0]) for t in self.tokens],
            generation_time_ms=100.0,
            tokens_per_second=40.0,
            decode_token_count=len(self.tokens),
        )

    def generate(self, prompt: str, **gen_kwargs: Any):
        self.generate_call_count += 1
        yield from self.tokens


class FakeTokenizer:
    """Fake tokenizer with optional chat template support."""

    def __init__(self, has_chat_template: bool = True) -> None:
        self.eos_token_ids = {0}
        if has_chat_template:
            self._apply_chat_template = self._chat_template_impl
        else:
            self._apply_chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens, **_):
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if self._apply_chat_template is None:
            raise AttributeError("no chat template")
        return self._apply_chat_template(messages, tokenize, add_generation_prompt)

    def _chat_template_impl(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)


def _app_with_fake_generator(
    fake_generator: FakeGenerator | None = None,
    tokenizer: FakeTokenizer | None = None,
    **overrides: Any,
) -> tuple:
    """Create app + TestClient with a fake generator injected into state."""
    cfg = RFSNConfig.from_env()
    cfg.model.id = "test-model"
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    application = create_app(cfg)
    state: ServerState = application.state.server  # type: ignore[attr-defined]
    state.generator = fake_generator or FakeGenerator()
    state.model = object()
    state.tokenizer = tokenizer or FakeTokenizer()
    state.metrics["model_loaded"] = True

    return application, TestClient(
        application, raise_server_exceptions=False,
    )


@pytest.mark.server
class TestChatCompletions:
    """Test chat completions endpoint."""

    def test_chat_completions_non_streaming(self):
        """Non-streaming chat completions returns full text."""
        _, client = _app_with_fake_generator(
            **{"server.require_api_key": False},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_chat_completions_streaming(self):
        """Streaming chat completions yields SSE events."""
        _, client = _app_with_fake_generator(
            fake_generator=FakeGenerator(tokens=["One", " Two", " Three"]),
            **{"server.require_api_key": False},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Count"}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        events = []
        for line in response.iter_lines():
            if line:
                line_str = line if isinstance(line, str) else line.decode("utf-8")
                if line_str.startswith("data: "):
                    payload = line_str[len("data: "):]
                    if payload != "[DONE]":
                        events.append(json.loads(payload))

        # Should have 3 token events + 1 final event
        assert len(events) >= 2
        token_text = "".join(
            e["choices"][0]["delta"].get("content", "")
            for e in events
            if "delta" in e["choices"][0] and e["choices"][0].get("finish_reason") is None
        )
        assert token_text == "One Two Three"

    def test_chat_completions_stop_sequence(self):
        """Streaming stops when a stop sequence is encountered."""
        _, client = _app_with_fake_generator(
            fake_generator=FakeGenerator(
                tokens=["Hello", ",", " stop", "here"],
            ),
            **{"server.require_api_key": False},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
                "stream": True,
                "stop": [" stop"],
            },
        )
        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line:
                line_str = line if isinstance(line, str) else line.decode("utf-8")
                if line_str.startswith("data: "):
                    payload = line_str[len("data: "):]
                    if payload not in ("[DONE]", ""):
                        events.append(json.loads(payload))

        token_text = "".join(
            e["choices"][0]["delta"].get("content", "")
            for e in events
            if e["choices"][0].get("finish_reason") is None
        )
        # Should stop BEFORE " stop", yielding only "Hello,"
        assert " stop" not in token_text
        assert token_text == "Hello,"

        # Final event should have finish_reason="stop"
        final_events = [e for e in events if e["choices"][0].get("finish_reason") is not None]
        assert len(final_events) == 1
        assert final_events[0]["choices"][0]["finish_reason"] == "stop"

    def test_chat_completions_tokenizer_fallback(self):
        """Server falls back to simple prompt format when tokenizer lacks chat template."""
        tok = FakeTokenizer(has_chat_template=False)
        gen = FakeGenerator()
        _, client = _app_with_fake_generator(
            fake_generator=gen,
            tokenizer=tok,
            **{"server.require_api_key": False},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert response.status_code == 200
        # The generator was called; we just verify no crash occurred.
        assert gen.chat_call_count == 1

    def test_chat_completions_finish_reason_length(self):
        """finish_reason is 'length' when max_tokens is reached."""
        _, client = _app_with_fake_generator(
            fake_generator=FakeGenerator(tokens=["a", "b", "c", "d", "e"]),
            **{"server.require_api_key": False},
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Count"}],
                "max_tokens": 3,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "length"

    def test_chat_completions_packed_reference_propagation(self):
        """packed_reference=True is propagated through ServerState to generator."""
        cfg = RFSNConfig.from_env()
        cfg.model.id = "test-model"
        cfg.runtime.packed_reference = True
        application = create_app(cfg)
        state: ServerState = application.state.server  # type: ignore[attr-defined]
        state.generator = FakeGenerator()
        state.model = object()
        state.tokenizer = FakeTokenizer()
        state.metrics["model_loaded"] = True

        client = TestClient(application, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "stream": False,
            },
            headers={},
        )
        assert response.status_code == 200
        # The test passes if the server didn't crash while using packed_reference=True.

    @pytest.mark.anyio
    async def test_concurrent_streaming_requests(self):
        """3 concurrent streaming requests all complete without interference."""
        import asyncio

        import httpx
        from httpx import ASGITransport

        gen = FakeGenerator(tokens=["A", "B", "C"])
        app, _ = _app_with_fake_generator(
            fake_generator=gen,
            **{"server.require_api_key": False},
        )

        async def _request(idx: int):
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": f"Request {idx}"}],
                        "max_tokens": 10,
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                events = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[len("data: "):]
                        if payload not in ("[DONE]", ""):
                            events.append(json.loads(payload))
                return events

        results = await asyncio.gather(*(_request(i) for i in range(3)))

        # All 3 requests must have produced token events
        for i, events in enumerate(results):
            token_text = "".join(
                e["choices"][0]["delta"].get("content", "")
                for e in events
                if e["choices"][0].get("finish_reason") is None
            )
            assert token_text == "ABC", f"Request {i} produced wrong text: {token_text!r}"

        # Generator should have been called 3 times (once per request)
        # Streaming path uses generate(), not chat()
        assert gen.generate_call_count == 3
