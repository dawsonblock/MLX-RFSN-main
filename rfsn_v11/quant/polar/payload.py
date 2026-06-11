"""PolarQuant payload dataclass for compressed K blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass
class PolarKeyBlock:
    """Compressed K block using PolarQuant.

    For each token vector in the block:
      - Dimensions are optionally rotated by an orthogonal matrix.
      - Adjacent dimension pairs are converted to (radius, angle) polar coords.
      - Angles are quantized to `angle_bits` levels.
      - Radii are stored as float16.
    """

    radii: mx.array          # fp16, shape (..., D/2)
    angle_codes_l1: mx.array # uint8, shape (..., D/2) for level-1 angles
    angle_codes_deep: mx.array | None  # uint8 or None, for deep refinement
    shape: tuple[int, ...]   # original shape before flattening pairs
    block_size: int
    head_dim: int
    angle_bits_level1: int
    angle_bits_deep: int
    metadata: dict[str, Any]

    def compressed_nbytes(self) -> int:
        """Return actual byte count of stored arrays."""
        total = self.radii.nbytes
        total += self.angle_codes_l1.nbytes
        if self.angle_codes_deep is not None:
            total += self.angle_codes_deep.nbytes
        return int(total)

    def original_nbytes(self) -> int:
        """Return equivalent FP16 byte count for the original K block."""
        prod = 1
        for d in self.shape:
            prod *= d
        return prod * 2  # fp16 = 2 bytes

    def compression_factor(self) -> float:
        orig = self.original_nbytes()
        comp = self.compressed_nbytes()
        return orig / comp if comp > 0 else 0.0
