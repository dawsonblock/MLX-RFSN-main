"""Model inspection for polar_fused compatibility.

Determines which layers can safely use Polar attention and which must fall back
to standard MLX attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfsn_v11.polar_fused.config import PolarFusedConfig


@dataclass(frozen=True)
class LayerClassification:
    """Classification of a single model layer."""
    layer_id: int
    layer_type: str  # "STANDARD_ATTENTION", "LINEAR_ATTENTION", etc.
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    supports_polar: bool
    reason: str | None = None


class ModelInspector:
    """Inspect a loaded MLX model and classify each layer."""

    def __init__(self, config: PolarFusedConfig) -> None:
        self.cfg = config

    def inspect(self, model: Any) -> list[LayerClassification]:
        """Inspect model layers and return classifications."""
        results: list[LayerClassification] = []

        for i, layer in enumerate(getattr(model, "layers", [])):
            classification = self._classify_layer(i, layer, model)
            results.append(classification)

        return results

    def _classify_layer(self, layer_id: int, layer: Any, model: Any) -> LayerClassification:
        """Classify a single layer."""
        # Extract attention module
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            return LayerClassification(
                layer_id=layer_id,
                layer_type="UNKNOWN",
                n_q_heads=0,
                n_kv_heads=0,
                head_dim=0,
                supports_polar=False,
                reason="No attention module found",
            )

        # Extract dimensions
        n_q_heads = getattr(attn, "n_heads", 0)
        n_kv_heads = getattr(attn, "n_kv_heads", n_q_heads)

        # Determine head_dim — try multiple sources
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            # Try model args/config
            model_args = getattr(model, "args", None)
            if model_args is not None:
                hidden_size = getattr(model_args, "hidden_size", None)
                num_heads = getattr(model_args, "num_attention_heads", None)
                if hidden_size and num_heads:
                    head_dim = hidden_size // num_heads
        if head_dim is None:
            # Infer from q_proj weight shape: (n_heads * head_dim, hidden_size)
            if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
                q_out = attn.q_proj.weight.shape[0]
                if n_q_heads > 0 and q_out % n_q_heads == 0:
                    head_dim = q_out // n_q_heads
        if head_dim is None:
            head_dim = 0

        # Check head_dim compatibility
        if head_dim not in (64, 128):
            return LayerClassification(
                layer_id=layer_id,
                layer_type="STANDARD_ATTENTION",
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                supports_polar=False,
                reason=f"head_dim {head_dim} not in {{64, 128}}",
            )

        # Check GQA compatibility
        if n_q_heads % n_kv_heads != 0:
            return LayerClassification(
                layer_id=layer_id,
                layer_type="STANDARD_ATTENTION",
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                supports_polar=False,
                reason="n_q_heads not divisible by n_kv_heads",
            )

        # Check for unsupported layer types
        layer_type = "STANDARD_ATTENTION"
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gating"):
            layer_type = "GATED_MLP"

        # Check for sliding window
        if hasattr(attn, "sliding_window") and attn.sliding_window is not None:
            return LayerClassification(
                layer_id=layer_id,
                layer_type="SLIDING_ATTENTION",
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                supports_polar=False,
                reason="Sliding window not supported",
            )

        # Check for KV sharing across layers
        if hasattr(attn, "shared_kv") and attn.shared_kv:
            return LayerClassification(
                layer_id=layer_id,
                layer_type="KV_SHARED",
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                supports_polar=False,
                reason="KV sharing across layers not supported",
            )

        return LayerClassification(
            layer_id=layer_id,
            layer_type=layer_type,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            supports_polar=True,
        )

    def get_polar_eligible_layers(self, model: Any) -> list[int]:
        """Return layer IDs that can use Polar attention."""
        return [c.layer_id for c in self.inspect(model) if c.supports_polar]

    def get_fallback_layers(self, model: Any) -> list[int]:
        """Return layer IDs that must use standard MLX attention."""
        return [c.layer_id for c in self.inspect(model) if not c.supports_polar]

    def summary(self, model: Any) -> dict[str, Any]:
        """Human-readable summary."""
        classes = self.inspect(model)
        eligible = [c for c in classes if c.supports_polar]
        fallback = [c for c in classes if not c.supports_polar]
        return {
            "total_layers": len(classes),
            "polar_eligible": len(eligible),
            "fallback": len(fallback),
            "eligible_layer_ids": [c.layer_id for c in eligible],
            "fallback_layer_ids": [c.layer_id for c in fallback],
            "fallback_reasons": {c.layer_id: c.reason for c in fallback if c.reason},
        }
