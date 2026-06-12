#!/usr/bin/env python3
"""RFSN v10 — Generation loop integration tests.

These tests verify the explicit per-layer cache adapter path.
The global SDPA monkeypatch layer has been removed.
"""

from __future__ import annotations

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
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
    )
    assert gen._adapter is not None


def test_generator_no_adapter_when_kv_disabled() -> None:
    gen = RFSNGenerator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=False,
    )
    assert gen._adapter is None


def test_generator_accepts_backward_compat_kwargs() -> None:
    """Deprecated kwargs (enable_sparse_decode, audit_mode, etc.) must not raise."""
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


def test_generator_does_not_mutate_model_layers() -> None:
    """The explicit adapter must not wrap or mutate model attention layers."""
    model = FakeModel(num_layers=3)
    gen = RFSNGenerator(
        model=model,
        tokenizer=FakeTokenizer(),
        enable_quantized_kv=True,
    )
    # No monkeypatch artifacts
    for layer in model.model.layers:
        assert not hasattr(layer.self_attn, "_rfsn_original_call")
