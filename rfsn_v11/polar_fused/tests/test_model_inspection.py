"""Tests for ModelInspector."""
from __future__ import annotations

import pytest

from rfsn_v11.polar_fused.adapters.model_inspection import ModelInspector
from rfsn_v11.polar_fused.config import PolarFusedConfig


class MockAttention:
    n_heads = 16
    n_kv_heads = 4
    head_dim = 64


class MockLayer:
    self_attn = MockAttention()


class MockModel:
    layers = [MockLayer() for _ in range(12)]


def test_standard_attention_detected() -> None:
    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    classes = inspector.inspect(MockModel())
    assert len(classes) == 12
    for c in classes:
        assert c.layer_type == "STANDARD_ATTENTION"
        assert c.n_q_heads == 16
        assert c.n_kv_heads == 4
        assert c.head_dim == 64
        assert c.supports_polar


def test_get_eligible_layers() -> None:
    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    eligible = inspector.get_polar_eligible_layers(MockModel())
    assert eligible == list(range(12))


def test_get_fallback_layers() -> None:
    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    fallback = inspector.get_fallback_layers(MockModel())
    assert fallback == []


def test_summary() -> None:
    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    summary = inspector.summary(MockModel())
    assert summary["total_layers"] == 12
    assert summary["polar_eligible"] == 12
    assert summary["fallback"] == 0


def test_unsupported_head_dim() -> None:
    class BadAttention:
        n_heads = 16
        n_kv_heads = 4
        head_dim = 32

    class BadLayer:
        self_attn = BadAttention()

    class BadModel:
        layers = [BadLayer()]

    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    classes = inspector.inspect(BadModel())
    assert not classes[0].supports_polar
    assert "head_dim" in (classes[0].reason or "")


def test_gqa_incompatible() -> None:
    class BadAttention:
        n_heads = 15
        n_kv_heads = 4
        head_dim = 64

    class BadLayer:
        self_attn = BadAttention()

    class BadModel:
        layers = [BadLayer()]

    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    classes = inspector.inspect(BadModel())
    assert not classes[0].supports_polar
    assert "divisible" in (classes[0].reason or "")


def test_no_attention_module() -> None:
    class EmptyLayer:
        pass

    class EmptyModel:
        layers = [EmptyLayer()]

    inspector = ModelInspector(PolarFusedConfig.polar_safe())
    classes = inspector.inspect(EmptyModel())
    assert not classes[0].supports_polar
    assert classes[0].layer_type == "UNKNOWN"
