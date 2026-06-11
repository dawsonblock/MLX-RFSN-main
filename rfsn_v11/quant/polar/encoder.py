"""PolarQuant encoder for K-cache blocks.

Algorithm:
  1. Optional orthogonal rotation.
  2. Pair up adjacent dimensions: (d0,d1), (d2,d3), ...
  3. Convert each pair to polar (radius, angle).
  4. Uniformly quantize angles into `angle_bits` bins over [0, 2*pi).
  5. Store radii as fp16, angles as uint8 codes.
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


class PolarQuantEncoder:
    """Encode K blocks into PolarKeyBlock payloads."""

    def __init__(
        self,
        angle_bits_level1: int = 4,
        angle_bits_deep: int = 2,
        head_dim: int = 128,
        use_rotation: bool = True,
        rotation_seed: int = 42,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for polar pairing")
        self.angle_bits_level1 = angle_bits_level1
        self.angle_bits_deep = angle_bits_deep
        self.head_dim = head_dim
        self.use_rotation = use_rotation
        self.rotation_seed = rotation_seed
        self._rotation_matrix: mx.array | None = None
        if use_rotation:
            self._rotation_matrix = _random_orthogonal(head_dim, rotation_seed)

    def encode(self, keys: mx.array) -> PolarKeyBlock:
        """Compress a K block.

        Args:
            keys: array of shape (..., head_dim)  e.g. (B, H, block_size, D)

        Returns:
            PolarKeyBlock with quantized polar representation.
        """
        orig_shape = keys.shape
        # Flatten to (-1, head_dim)
        flat = keys.reshape(-1, self.head_dim)

        # Optional rotation
        if self.use_rotation and self._rotation_matrix is not None:
            rotated = flat @ self._rotation_matrix.T
        else:
            rotated = flat

        # Pair dimensions: (..., D/2, 2)
        paired = rotated.reshape(rotated.shape[0], self.head_dim // 2, 2)
        x = paired[:, :, 0]
        y = paired[:, :, 1]

        # Polar transform
        radii = mx.sqrt(x * x + y * y).astype(mx.float16)
        angles = mx.arctan2(y, x)  # [-pi, pi]

        # Normalize to [0, 2*pi)
        angles_norm = mx.where(angles < 0, angles + 2 * math.pi, angles)

        # Quantize angles: uniform over [0, 2*pi)
        n_levels_l1 = 1 << self.angle_bits_level1
        angle_codes_l1 = mx.clip(
            (angles_norm / (2 * math.pi) * n_levels_l1).astype(mx.uint8),
            0,
            n_levels_l1 - 1,
        )

        # Deep refinement (optional, disabled when bits=0 or deep not used)
        angle_codes_deep = None
        if self.angle_bits_deep > 0:
            # Quantize residual within each level-1 bin
            bin_width = (2 * math.pi) / n_levels_l1
            bin_start = angle_codes_l1.astype(mx.float32) * bin_width
            residual = angles_norm - bin_start
            n_levels_deep = 1 << self.angle_bits_deep
            angle_codes_deep = mx.clip(
                (residual / bin_width * n_levels_deep).astype(mx.uint8),
                0,
                n_levels_deep - 1,
            )

        return PolarKeyBlock(
            radii=radii,
            angle_codes_l1=angle_codes_l1,
            angle_codes_deep=angle_codes_deep,
            shape=orig_shape,
            block_size=orig_shape[-2] if len(orig_shape) >= 2 else orig_shape[0],
            head_dim=self.head_dim,
            angle_bits_level1=self.angle_bits_level1,
            angle_bits_deep=self.angle_bits_deep,
            metadata={
                "use_rotation": self.use_rotation,
                "rotation_seed": self.rotation_seed,
            },
        )
