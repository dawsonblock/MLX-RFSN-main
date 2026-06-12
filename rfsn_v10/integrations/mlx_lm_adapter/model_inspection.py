"""Model inspection helpers for MLX-LM integration.

Determines head dimensions, layer counts, and GQA configuration
from the model object without hard-coding architecture specifics.
"""
from __future__ import annotations

from typing import Any


def inspect_model(model: Any) -> dict[str, Any]:
    """Inspect an MLX-LM model and return configuration metadata.

    Returns
    -------
    dict with keys:
        num_layers, num_heads, num_kv_heads, head_dim, hidden_size,
        model_type, rope_theta
    """
    result: dict[str, Any] = {
        "num_layers": len(getattr(model, "layers", [])),
        "model_type": getattr(model, "model_type", "unknown"),
    }

    # Try to get args from the model
    args = getattr(model, "args", None)
    if args is None:
        return result

    result["num_heads"] = getattr(args, "num_attention_heads", 0)
    result["num_kv_heads"] = getattr(args, "num_key_value_heads", result["num_heads"])
    result["hidden_size"] = getattr(args, "hidden_size", 0)
    result["head_dim"] = getattr(args, "head_dim", 0)
    result["rope_theta"] = getattr(args, "rope_theta", 10000.0)

    # Infer head_dim if not explicitly set
    if result["head_dim"] == 0 and result["num_heads"] > 0:
        result["head_dim"] = result["hidden_size"] // result["num_heads"]

    return result


def infer_num_layers(model: Any) -> int:
    """Return the number of transformer layers."""
    return len(getattr(model, "layers", []))


def infer_head_dim(model: Any) -> int:
    """Return the attention head dimension."""
    args = getattr(model, "args", None)
    if args is not None:
        hd = getattr(args, "head_dim", 0)
        if hd:
            return hd
        hidden = getattr(args, "hidden_size", 0)
        heads = getattr(args, "num_attention_heads", 0)
        if hidden and heads:
            return hidden // heads

    # Fallback: inspect first layer
    layers = getattr(model, "layers", [])
    if layers:
        attn = getattr(layers[0], "self_attn", None)
        if attn is not None and hasattr(attn, "head_dim"):
            return int(attn.head_dim)

    return 0
