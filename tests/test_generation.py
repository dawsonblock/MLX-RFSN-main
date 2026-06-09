#!/usr/bin/env python3
"""RFSN v10 — Generation loop integration tests."""

from __future__ import annotations

from rfsn_v10.runtime.generation import (
    RFSNGenerator,
    _rfsn_sdpa_wrapper,
    _wrap_layers_for_rfsn,
    _unwrap_layers_for_rfsn,
)


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
# RFSNGenerator construction
# ------------------------------------------------------------------


def test_generator_initializes_runtime_when_kv_enabled():
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
        enable_sparse_decode=True,
    )
    assert gen._runtime is not None
    assert gen._kv_manager is not None


def test_generator_no_runtime_when_quantized_kv_disabled():
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=False,
    )
    assert gen._runtime is None
    assert gen._kv_manager is None


# ------------------------------------------------------------------
# Layer wrapping / unwrapping helpers
# ------------------------------------------------------------------


def test_wrap_and_unwrap_layers():
    model = FakeModel(num_layers=3)
    _wrap_layers_for_rfsn(model)

    for _layer in model.model.layers:
        assert hasattr(_layer.self_attn, "_rfsn_original_call")

    _unwrap_layers_for_rfsn(model)

    for _layer in model.model.layers:
        assert not hasattr(_layer.self_attn, "_rfsn_original_call")


def test_wrap_is_idempotent():
    model = FakeModel(num_layers=2)
    _wrap_layers_for_rfsn(model)
    _wrap_layers_for_rfsn(model)  # second call should be a no-op
    assert hasattr(model.model.layers[0].self_attn, "_rfsn_original_call")


# ------------------------------------------------------------------
# SDPA wrapper fallback behaviour
# ------------------------------------------------------------------


def test_rfsn_sdpa_wrapper_falls_back_when_no_runtime():
    """When thread-local runtime is absent, wrapper must call original."""
    calls = []

    def original(_q, _k, _v, _cache, _scale, _mask, _sinks=None):
        calls.append("original")
        return "original_output"

    result = _rfsn_sdpa_wrapper(
        original, "q", "k", "v", cache="cache", scale=1.0, mask=None
    )
    assert result == "original_output"
    assert calls == ["original"]
