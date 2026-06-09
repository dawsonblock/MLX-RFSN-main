#!/usr/bin/env python3
"""
IsoQuant-style quaternion preconditioner.
Correct math:
    forward:  y = q_L * x * conjugate(q_R)
    inverse:  x = conjugate(q_L) * y * q_R
Each contiguous 4D block is treated as a quaternion.
"""
from __future__ import annotations

from dataclasses import dataclass

from rfsn_v10.compat import mx


def _validate_quat_tensor(x: mx.array) -> None:
    if len(x.shape) < 1:
        raise ValueError(
            f"Expected tensor with feature dimension, got shape={x.shape}"
        )
    if x.shape[-1] % 4 != 0:
        raise ValueError(
            "Feature dimension must be divisible by 4 for quaternion blocks. "
            f"Got D={x.shape[-1]}"
        )


def quat_conjugate(q: mx.array) -> mx.array:
    """Quaternion conjugate: [w, x, y, z] -> [w, -x, -y, -z]."""
    return mx.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_normalize(q: mx.array, eps: float = 1e-8) -> mx.array:
    norm = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + eps)
    return q / norm


def quat_multiply(a: mx.array, b: mx.array) -> mx.array:
    """
    Quaternion multiplication.
    Both a and b must have final dimension 4:
        a = [aw, ax, ay, az]
        b = [bw, bx, by, bz]
    """
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return mx.concatenate([w, x, y, z], axis=-1)


@dataclass
class IsoQuantMetadata:
    original_shape: tuple[int, ...]
    feature_dim: int
    q_l: mx.array
    q_r: mx.array


class IsoQuantPreconditioner:
    """
    Deterministic quaternion block rotation.
    This is not learned. It is a fixed, seeded preconditioner for quantization.
    """

    def __init__(
        self,
        feature_dim: int,
        seed: int = 42,
        q_l: mx.array | None = None,
        q_r: mx.array | None = None,
    ):
        if feature_dim % 4 != 0:
            raise ValueError(
                f"feature_dim must be divisible by 4. Got {feature_dim}"
            )
        self.feature_dim = feature_dim
        self.seed = seed
        if q_l is None or q_r is None:
            # Deterministic default rotations. Keep simple and stable.
            q_l = mx.array(
                [0.70710678, 0.70710678, 0.0, 0.0], dtype=mx.float32
            )
            q_r = mx.array(
                [0.70710678, 0.0, 0.70710678, 0.0], dtype=mx.float32
            )
        self.q_l = quat_normalize(q_l.astype(mx.float32))
        self.q_r = quat_normalize(q_r.astype(mx.float32))

    def forward(self, x: mx.array) -> tuple[mx.array, IsoQuantMetadata]:
        """
        Rotate x into IsoQuant space.
        Input/output shape is preserved.
        """
        _validate_quat_tensor(x)
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected D={self.feature_dim}, got D={x.shape[-1]}"
            )
        original_shape = tuple(x.shape)
        blocks = x.astype(mx.float32).reshape(-1, self.feature_dim // 4, 4)
        q_l = self.q_l.reshape(1, 1, 4)
        q_r_conj = quat_conjugate(self.q_r).reshape(1, 1, 4)
        rotated = quat_multiply(quat_multiply(q_l, blocks), q_r_conj)
        rotated = rotated.reshape(original_shape)
        meta = IsoQuantMetadata(
            original_shape=original_shape,
            feature_dim=self.feature_dim,
            q_l=self.q_l,
            q_r=self.q_r,
        )
        return rotated, meta

    def inverse(
        self, y: mx.array, meta: IsoQuantMetadata | None = None
    ) -> mx.array:
        """
        Inverse rotation back into original coordinate space.
        If forward used:
            y = q_L * x * conjugate(q_R)
        Then inverse is:
            x = conjugate(q_L) * y * q_R
        """
        _validate_quat_tensor(y)
        if meta is None:
            q_l = self.q_l
            q_r = self.q_r
            feature_dim = self.feature_dim
        else:
            q_l = meta.q_l
            q_r = meta.q_r
            feature_dim = meta.feature_dim
        if y.shape[-1] != feature_dim:
            raise ValueError(
                f"Expected D={feature_dim}, got D={y.shape[-1]}"
            )
        blocks = y.astype(mx.float32).reshape(-1, feature_dim // 4, 4)
        q_l_conj = quat_conjugate(q_l).reshape(1, 1, 4)
        q_r = q_r.reshape(1, 1, 4)
        restored = quat_multiply(quat_multiply(q_l_conj, blocks), q_r)
        return restored.reshape(y.shape)
