"""Tests for model architecture inspection and support validation."""
from __future__ import annotations

import pytest


class FakeArgs:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeModel:
    def __init__(self, model_type: str, num_layers: int, args: FakeArgs) -> None:
        self.model_type = model_type
        self.args = args
        self.layers = [object() for _ in range(num_layers)]


def test_inspect_qwen2_like_architecture() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        inspect_model_architecture,
    )

    model = FakeModel(
        model_type="qwen2",
        num_layers=24,
        args=FakeArgs(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            rope_theta=1000000.0,
            rope_traditional=False,
        ),
    )

    arch = inspect_model_architecture(model)
    assert arch.model_type == "qwen2"
    assert arch.num_layers == 24
    assert arch.num_heads == 14
    assert arch.num_kv_heads == 2
    assert arch.head_dim == 64  # 896 // 14
    assert arch.hidden_size == 896
    assert arch.rope_theta == 1000000.0
    assert arch.rope_traditional is False
    assert arch.attention_scale == 64 ** -0.5
    assert arch.gqa_ratio == 7


def test_inspect_missing_model_type_raises() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        inspect_model_architecture,
    )

    model = FakeModel(
        model_type=None,  # type: ignore[arg-type]
        num_layers=2,
        args=FakeArgs(),
    )

    with pytest.raises(ValueError, match="model_type"):
        inspect_model_architecture(model)


def test_inspect_missing_args_raises() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        inspect_model_architecture,
    )

    model = FakeModel(
        model_type="qwen2",
        num_layers=2,
        args=None,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="model.args"):
        inspect_model_architecture(model)


def test_supported_qwen2_passes() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        ModelArchitecture,
        is_supported_architecture,
    )

    arch = ModelArchitecture(
        model_type="qwen2",
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=896,
        rope_theta=1000000.0,
        rope_traditional=False,
        attention_scale=64 ** -0.5,
    )
    ok, reason = is_supported_architecture(arch)
    assert ok is True
    assert "supported" in reason


def test_unsupported_model_type_rejected() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        ModelArchitecture,
        is_supported_architecture,
    )

    arch = ModelArchitecture(
        model_type="llama",
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=896,
        rope_theta=10000.0,
        rope_traditional=False,
        attention_scale=64 ** -0.5,
    )
    ok, reason = is_supported_architecture(arch)
    assert ok is False
    assert "llama" in reason


def test_unsupported_head_dim_rejected() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        ModelArchitecture,
        is_supported_architecture,
    )

    arch = ModelArchitecture(
        model_type="qwen2",
        num_layers=24,
        num_heads=16,
        num_kv_heads=2,
        head_dim=32,
        hidden_size=512,
        rope_theta=10000.0,
        rope_traditional=False,
        attention_scale=32 ** -0.5,
    )
    ok, reason = is_supported_architecture(arch)
    assert ok is False
    assert "head_dim" in reason


def test_invalid_gqa_ratio_rejected() -> None:
    from rfsn_v10.integrations.mlx_lm_model_support.model_support import (
        ModelArchitecture,
        is_supported_architecture,
    )

    arch = ModelArchitecture(
        model_type="qwen2",
        num_layers=24,
        num_heads=15,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=960,
        rope_theta=10000.0,
        rope_traditional=False,
        attention_scale=64 ** -0.5,
    )
    ok, reason = is_supported_architecture(arch)
    assert ok is False
    assert "divisible" in reason
