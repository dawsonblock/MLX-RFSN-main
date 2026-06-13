"""Model architecture inspection and support validation.

Only one exact model family is supported initially.
All others are rejected with a clear error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """Validated architecture metadata for a supported model."""

    model_type: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    rope_traditional: bool
    attention_scale: float

    @property
    def gqa_ratio(self) -> int:
        if self.num_kv_heads == 0:
            raise ValueError("num_kv_heads is zero")
        return self.num_heads // self.num_kv_heads


def inspect_model_architecture(model: Any) -> ModelArchitecture:
    """Inspect an MLX-LM model and return validated architecture metadata.

    Raises
    ------
    ValueError
        If required fields cannot be determined.
    """
    model_type = getattr(model, "model_type", None)
    if model_type is None:
        raise ValueError("model_type is not set on the model object")

    args = getattr(model, "args", None)
    if args is None:
        raise ValueError("model.args is not present")

    num_layers = len(getattr(model, "layers", []))
    if num_layers == 0:
        raise ValueError("model has no layers")

    num_heads = getattr(args, "num_attention_heads", 0)
    num_kv_heads = getattr(args, "num_key_value_heads", num_heads)
    hidden_size = getattr(args, "hidden_size", 0)
    head_dim = getattr(args, "head_dim", 0)

    if head_dim == 0 and num_heads > 0:
        head_dim = hidden_size // num_heads

    if head_dim == 0:
        raise ValueError("head_dim cannot be determined")
    if num_heads == 0:
        raise ValueError("num_attention_heads is zero")
    if num_kv_heads == 0:
        raise ValueError("num_key_value_heads is zero")

    rope_theta = getattr(args, "rope_theta", 10000.0)
    rope_traditional = getattr(args, "rope_traditional", False)
    attention_scale = head_dim ** -0.5

    return ModelArchitecture(
        model_type=model_type,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        attention_scale=attention_scale,
    )


# ------------------------------------------------------------------
# Supported architectures (Phase 13: one exact family)
# ------------------------------------------------------------------

_SUPPORTED_TYPES = {"qwen2", "qwen2_5"}

_SUPPORTED_HEAD_DIMS = {64, 128}

# Canonical configuration requirements
_REQUIRED_GS64 = True


def is_supported_architecture(arch: ModelArchitecture) -> tuple[bool, str]:
    """Check if an architecture is supported for direct packed attention.

    Returns
    -------
    ok, reason
        ``ok`` is True if supported; ``reason`` explains why not.
    """
    if arch.model_type not in _SUPPORTED_TYPES:
        return False, (
            f"model_type '{arch.model_type}' is not in the supported set: "
            f"{_SUPPORTED_TYPES}"
        )

    if arch.head_dim not in _SUPPORTED_HEAD_DIMS:
        return False, (
            f"head_dim {arch.head_dim} is not supported; "
            f"supported values: {_SUPPORTED_HEAD_DIMS}"
        )

    if arch.num_heads % arch.num_kv_heads != 0:
        return False, (
            f"GQA ratio invalid: num_heads ({arch.num_heads}) is not "
            f"divisible by num_kv_heads ({arch.num_kv_heads})"
        )

    return True, "architecture is supported"
