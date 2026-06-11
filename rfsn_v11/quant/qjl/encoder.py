"""QJL encoder: residual sketch for TurboPolar.

Projects the residual E = K_original - K_reconstructed onto a random
low-dimensional subspace and stores sign bits + per-vector norms.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class QJLPayload:
    """QJL sketch payload for a K block."""

    packed_signs: mx.array   # uint8, shape (..., proj_dim // 8 + pad)
    norms: mx.array          # fp16, shape (..., 1) or (...)
    proj_dim: int
    seed: int
    shape: tuple[int, ...]   # original residual shape

    def sketch_nbytes(self) -> int:
        return int(self.packed_signs.nbytes + self.norms.nbytes)


def _random_projection_matrix(dim: int, proj_dim: int, seed: int) -> mx.array:
    """Generate a random Gaussian projection matrix."""
    rng = np.random.RandomState(seed)
    mat = rng.randn(dim, proj_dim).astype(np.float32)
    # Normalize columns to unit length so dot-product estimates are scaled correctly
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    mat = mat / (norms + 1e-12)
    return mx.array(mat)


def _pack_signs(signs: mx.array) -> mx.array:
    """Pack binary signs (+1 -> 1, -1 -> 0) into uint8.

    signs: uint8 array of 0/1 values, shape (..., proj_dim)
    Returns uint8 packed array.
    """
    orig_shape = signs.shape
    n = int(np.prod(orig_shape[:-1]))
    proj_dim = orig_shape[-1]
    flat = signs.reshape(n, proj_dim)

    # Pad to multiple of 8
    pad = (8 - proj_dim % 8) % 8
    if pad:
        pad_arr = mx.zeros((n, pad), dtype=mx.uint8)
        flat = mx.concatenate([flat, pad_arr], axis=-1)
    packed_dim = flat.shape[-1] // 8

    # Bit-pack: each byte stores 8 sign bits
    # (shift and sum — MLX does not have bit operations, so use arithmetic)
    out = mx.zeros((n, packed_dim), dtype=mx.uint8)
    for i in range(8):
        out = out + (flat[:, i::8] << i)

    # Reshape back to leading dimensions + (packed_dim,)
    new_shape = orig_shape[:-1] + (packed_dim,)
    return out.reshape(new_shape)


def _unpack_signs(packed: mx.array, proj_dim: int) -> mx.array:
    """Unpack uint8 packed signs back to 0/1 values.

    packed: uint8, shape (..., packed_dim)
    Returns uint8 array of shape (..., proj_dim).
    """
    orig_shape = packed.shape
    n = int(np.prod(orig_shape[:-1]))
    packed_dim = orig_shape[-1]
    flat = packed.reshape(n, packed_dim)

    # Extract bits
    bits = []
    for i in range(8):
        bit = (flat >> i) & 1
        bits.append(bit)
    signs = mx.stack(bits, axis=-1).reshape(n, packed_dim * 8)
    return signs[:, :proj_dim].reshape(orig_shape[:-1] + (proj_dim,))


class QJLEncoder:
    """Encode residual E into QJL sketch."""

    def __init__(self, proj_dim: int = 64, seed: int = 42) -> None:
        self.proj_dim = proj_dim
        self.seed = seed

    def encode(self, residual: mx.array) -> QJLPayload:
        """Compress residual tensor.

        Args:
            residual: shape (..., D) float array.

        Returns:
            QJLPayload with packed sign bits and per-vector norms.
        """
        orig_shape = residual.shape
        flat = residual.reshape(-1, residual.shape[-1])
        D = flat.shape[-1]

        # Per-vector norms
        norms = mx.linalg.norm(flat, axis=-1, keepdims=True).astype(mx.float16)

        # Random projection
        proj_mat = _random_projection_matrix(D, self.proj_dim, self.seed)
        projected = flat @ proj_mat  # (..., proj_dim)

        # Signs: +1 -> 1, -1/0 -> 0
        signs = (projected > 0).astype(mx.uint8)
        packed_signs = _pack_signs(signs)

        return QJLPayload(
            packed_signs=packed_signs,
            norms=norms,
            proj_dim=self.proj_dim,
            seed=self.seed,
            shape=orig_shape,
        )
