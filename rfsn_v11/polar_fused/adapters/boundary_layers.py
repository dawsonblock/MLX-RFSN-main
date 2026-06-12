"""Boundary-layer protection for polar_fused.

The first and last N attention layers can be more sensitive to quantization.
This module applies configurable boundary protection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfsn_v11.polar_fused.config import PolarFusedConfig


@dataclass(frozen=True)
class BoundaryConfig:
    """Configuration for boundary layer protection."""
    first_n: int = 2
    last_n: int = 2
    middle_mode: str = "polar"
    boundary_mode: str = "fp16"


class BoundaryLayerPolicy:
    """Apply boundary layer policy to a set of layer classifications."""

    def __init__(self, config: PolarFusedConfig) -> None:
        self.cfg = config
        self.boundary = BoundaryConfig(
            first_n=config.boundary_layers,
            last_n=config.boundary_layers,
        )

    def apply(
        self,
        eligible_layers: list[int],
        total_layers: int,
    ) -> dict[int, str]:
        """Return mapping from layer_id to mode ("polar" or "fp16").

        First N and last N eligible layers are kept in FP16.
        Middle eligible layers use Polar.
        """
        result: dict[int, str] = {}

        # Sort eligible layers
        sorted_layers = sorted(eligible_layers)

        # First N → FP16
        for layer_id in sorted_layers[:self.boundary.first_n]:
            result[layer_id] = self.boundary.boundary_mode

        # Last N → FP16
        for layer_id in sorted_layers[-self.boundary.last_n:]:
            result[layer_id] = self.boundary.boundary_mode

        # Middle → Polar
        for layer_id in sorted_layers[self.boundary.first_n:-self.boundary.last_n]:
            result[layer_id] = self.boundary.middle_mode

        return result

    def summary(self, layer_modes: dict[int, str]) -> dict[str, Any]:
        """Human-readable summary."""
        polar_count = sum(1 for m in layer_modes.values() if m == "polar")
        fp16_count = sum(1 for m in layer_modes.values() if m == "fp16")
        return {
            "total_eligible": len(layer_modes),
            "polar_layers": polar_count,
            "fp16_layers": fp16_count,
            "boundary_first_n": self.boundary.first_n,
            "boundary_last_n": self.boundary.last_n,
        }
