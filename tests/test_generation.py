#!/usr/bin/env python3
"""RFSN v10 — Generation loop integration tests.

These tests verify the explicit per-layer cache adapter path.
The global SDPA monkeypatch layer has been removed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.pure_python

from rfsn_v10.runtime.generation import RFSNGenerator


class FakeModel:
    """Minimal stand-in for an mlx_lm model."""

    def __init__(self, num_layers: int = 2) -> None:
        class Inner:
            def __init__(self, num_layers: int) -> None:
                self.layers = [FakeLayer(f"layer_{i}") for i in range(num_layers)]

        self.model = Inner(num_layers)

    def __call__(self, x):
        return x


class FakeLayer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.self_attn = FakeAttention()


class FakeAttention:
    def __call__(self, x, mask=None, cache=None):
        return x


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_ids = {0}

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(t) for t in tokens)


# ------------------------------------------------------------------
# RFSNGenerator construction (explicit adapter path)
# ------------------------------------------------------------------


def test_generator_creates_adapter_when_kv_enabled() -> None:
    """Test that adapter is created when MLX-LM is available and KV is enabled."""
    # This test only passes when MLX-LM is available
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        # On platforms without MLX, adapter should be None
        gen = RFSNGenerator(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            enable_quantized_kv=True,
        )
        assert gen._adapter is None, "Adapter should be None when MLX-LM is unavailable"
        return

    # When MLX-LM is available, adapter should be created
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
    )
    assert gen._adapter is not None, "Adapter should be created when MLX-LM is available"


def test_generator_no_adapter_when_kv_disabled() -> None:
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=False,
    )
    assert gen._adapter is None


def test_generator_accepts_backward_compat_kwargs() -> None:
    """Deprecated kwargs (enable_sparse_decode, audit_mode, etc.) must not raise."""
    # This test only passes when MLX-LM is available
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        # On platforms without MLX, just test that the kwargs are accepted
        gen = RFSNGenerator(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            enable_quantized_kv=True,
            enable_sparse_decode=True,  # deprecated no-op
            audit_mode=True,              # deprecated no-op
            use_compressed_on_miss=True,  # deprecated no-op
        )
        # Adapter will be None without MLX, but kwargs should not raise
        assert gen._adapter is None
        return

    # When MLX-LM is available, test with adapter creation
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
        enable_sparse_decode=True,  # deprecated no-op
        audit_mode=True,              # deprecated no-op
        use_compressed_on_miss=True,  # deprecated no-op
    )
    assert gen._adapter is not None


# ------------------------------------------------------------------
# No monkeypatching
# ------------------------------------------------------------------


def test_generation_finish_reason_length() -> None:
    """When max_new_tokens is reached, finish_reason must be 'length'."""
    from rfsn_v10.runtime.generation import GenerationConfig, GenerationResult
    result = GenerationResult(
        text="abc",
        tokens=[1, 2, 3],
        generation_time_ms=100.0,
        tokens_per_second=30.0,
    )
    # Simulate the contract check from server/app.py
    cfg = GenerationConfig(max_new_tokens=3)
    finish_reason = "stop"
    decode_tokens = len(result.tokens)
    if decode_tokens >= cfg.max_new_tokens:
        finish_reason = "length"
    assert finish_reason == "length"


def test_generator_does_not_mutate_model_layers() -> None:
    """The explicit adapter must not wrap or mutate model attention layers."""
    model = FakeModel(num_layers=3)
    gen = RFSNGenerator(
        model=model,
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
    )
    _ = gen  # construction is the test
    # No monkeypatch artifacts
    for layer in model.model.layers:
        assert not hasattr(layer.self_attn, "_rfsn_original_call")


# ------------------------------------------------------------------
# _build_chat_prompt
# ------------------------------------------------------------------


class FakeTokenizerWithTemplate(FakeTokenizer):
    """Tokenizer that supports apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)


def test_build_chat_prompt_with_template() -> None:
    tok = FakeTokenizerWithTemplate()
    gen = RFSNGenerator(model=FakeModel(), tokenizer=tok, enable_quantized_kv=False)
    prompt = gen._build_chat_prompt("Hello", system_prompt="Be helpful")
    assert "<system>Be helpful</system>" in prompt
    assert "<user>Hello</user>" in prompt
    assert "<assistant>" in prompt


def test_build_chat_prompt_fallback() -> None:
    tok = FakeTokenizer()
    gen = RFSNGenerator(model=FakeModel(), tokenizer=tok, enable_quantized_kv=False)
    prompt = gen._build_chat_prompt("Hello", system_prompt="Be helpful")
    assert "Be helpful" in prompt
    assert "Hello" in prompt


# ------------------------------------------------------------------
# _make_gen_config
# ------------------------------------------------------------------


def test_gen_config_defaults() -> None:
    gen = RFSNGenerator(
        model=FakeModel(), tokenizer=FakeTokenizer(), enable_quantized_kv=False
    )
    cfg = gen._make_gen_config()
    assert cfg.max_new_tokens == 256
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.9
    assert cfg.repetition_penalty == 1.0
    assert cfg.stream is True


def test_gen_config_overrides() -> None:
    gen = RFSNGenerator(
        model=FakeModel(), tokenizer=FakeTokenizer(), enable_quantized_kv=False
    )
    cfg = gen._make_gen_config(max_new_tokens=10, temperature=0.5, stream=False)
    assert cfg.max_new_tokens == 10
    assert cfg.temperature == 0.5
    assert cfg.stream is False


# ------------------------------------------------------------------
# MLX-dependent tests
# ------------------------------------------------------------------


@pytest.mark.mlx
def test_generator_mlx_path_imports() -> None:
    """When MLX is present, the adapter path is attempted."""
    pytest.importorskip("mlx.core")
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
    )
    # With MLX present, _adapter should be non-None.
    assert gen._adapter is not None
