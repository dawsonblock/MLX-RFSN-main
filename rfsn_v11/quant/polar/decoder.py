"""PolarQuant decoder for K-cache blocks.

Reconstructs K vectors from PolarKeyBlock payloads.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .payload import PolarKeyBlock


def _random_orthogonal(dim: int, seed: int) -> mx.array:
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    a = rng.randn(dim, dim).astype(np.float32)
    q, _ = np.linalg.qr(a)
    return mx.array(q.astype(np.float32))


class PolarQuantDecoder:
    """Decode PolarKeyBlock back to K arrays."""

    def __init__(
        self,
        head_dim: int = 128,
        use_rotation: bool = True,
        rotation_seed: int = 42,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for polar pairing")
        self.head_dim = head_dim
        self.use_rotation = use_rotation
        self.rotation_seed = rotation_seed
        self._rotation_matrix: mx.array | None = None
        if use_rotation:
            self._rotation_matrix = _random_orthogonal(head_dim, rotation_seed)

    def decode(self, block: PolarKeyBlock) -> mx.array:
        """Decompress a PolarKeyBlock to original-shaped K array.

        Args:
            block: PolarKeyBlock payload.

        Returns:
            Reconstructed keys of shape `block.shape`.
        """
        radii = block.radii.astype(mx.float32)
        angle_codes_l1 = block.angle_codes_l1.astype(mx.float32)

        n_levels_l1 = 1 << block.angle_bits_level1
        bin_width = (2 * math.pi) / n_levels_l1

        # Reconstruct angles from level-1 codes
        angles = angle_codes_l1 * bin_width

        # Add deep refinement if present
        if block.angle_codes_deep is not None and block.angle_bits_deep > 0:
            n_levels_deep = 1 << block.angle_bits_deep
            deep_width = bin_width / n_levels_deep
            angles = angles + block.angle_codes_deep.astype(mx.float32) * deep_width

        # Add half-bin offset for dequantization (midpoint of each bin)
        if block.angle_codes_deep is not None and block.angle_bits_deep > 0:
            n_levels_deep = 1 << block.angle_bits_deep
            deep_width = bin_width / n_levels_deep
            angles = angles + deep_width / 2.0
        else:
            angles = angles + bin_width / 2.0

        # Back to Cartesian
        x = radii * mx.cos(angles)
        y = radii * mx.sin(angles)

        # Interleave back to (..., D)
        paired = mx.stack([x, y], axis=-1)  # (..., D/2, 2)
        reconstructed = paired.reshape(paired.shape[0], self.head_dim)

        # Inverse rotation
        if self.use_rotation and self._rotation_matrix is not None:
            reconstructed = reconstructed @ self._rotation_matrix

        return reconstructed.reshape(block.shape).astype(mx.float32)
