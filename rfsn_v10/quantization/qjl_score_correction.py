#!/usr/bin/env python3
"""
QJL score correction.
Correct use:
    QJL sketch corrects inner products:
        q · (k_base + residual)
It does NOT reconstruct the residual vector directly.
Dense Gaussian reference estimator:
    g_i ~ N(0, I)
    y_i = sign(g_i · r)
    E[ y_i * (g_i · q) ] = sqrt(2/pi) * (r · q) / ||r||
Therefore:
    r · q ≈ ||r|| * sqrt(pi/2) * mean_i[ y_i * (g_i · q) ]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rfsn_v10.compat import mx


@dataclass
class QJLSketch:
    signs: mx.array  # [..., proj_dim], values ±1
    residual_norm: mx.array  # [..., 1]
    proj_dim: int
    feature_dim: int


class QJLScoreCorrector:
    """
    Dense Gaussian QJL reference.
    This is intentionally not the fast structured version.
    Use this to prove math first.
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int = 64,
        seed: int = 42,
        eps: float = 1e-8,
    ):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.eps = eps
        # Rows are standard Gaussian vectors g_i ~ N(0, I).
        self.projection = mx.random.normal(
            (proj_dim, feature_dim)
        ).astype(mx.float32)

    def sketch_residual(self, residual: mx.array) -> QJLSketch:
        """
        residual: [..., D]
        Returns signs of Gaussian projections and residual norm.
        """
        if residual.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected D={self.feature_dim}, "
                f"got D={residual.shape[-1]}"
            )
        flat = residual.astype(mx.float32).reshape(
            -1, self.feature_dim
        )
        projected = flat @ self.projection.T  # [N, proj_dim]
        signs = mx.where(
            projected >= 0,
            mx.ones_like(projected),
            -mx.ones_like(projected),
        )
        residual_norm = mx.sqrt(
            mx.sum(flat * flat, axis=-1, keepdims=True) + self.eps
        )
        signs = signs.reshape(
            residual.shape[:-1] + (self.proj_dim,)
        )
        residual_norm = residual_norm.reshape(
            residual.shape[:-1] + (1,)
        )
        return QJLSketch(
            signs=signs.astype(mx.float32),
            residual_norm=residual_norm,
            proj_dim=self.proj_dim,
            feature_dim=self.feature_dim,
        )

    def estimate_residual_dot(
        self, queries: mx.array, key_sketch: QJLSketch
    ) -> mx.array:
        """
        Estimate q · residual_k for attention.
        queries:
            [B, H, T_q, D]
        key_sketch.signs:
            [B, H, T_k, M]
        key_sketch.residual_norm:
            [B, H, T_k, 1]
        Returns:
            [B, H, T_q, T_k]
        """
        if queries.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected query D={self.feature_dim}, "
                f"got {queries.shape[-1]}"
            )
        if key_sketch.feature_dim != self.feature_dim:
            raise ValueError("Sketch feature_dim mismatch")
        if key_sketch.proj_dim != self.proj_dim:
            raise ValueError("Sketch proj_dim mismatch")
        bq, hq, tq, d = queries.shape
        bs, hs, _, m = key_sketch.signs.shape
        if (bq, hq) != (bs, hs):
            raise ValueError(
                f"Query/sketch batch-head mismatch: "
                f"queries={(bq, hq)}, sketch={(bs, hs)}"
            )
        q_flat = queries.astype(mx.float32).reshape(-1, d)
        q_proj = q_flat @ self.projection.T
        q_proj = q_proj.reshape(bq, hq, tq, m)
        # Sum over projection dimension:
        # [B,H,Tq,M] @ [B,H,M,Tk] -> [B,H,Tq,Tk]
        dot_est = q_proj @ key_sketch.signs.transpose(0, 1, 3, 2)
        dot_est = dot_est / float(self.proj_dim)
        correction = (
            key_sketch.residual_norm.transpose(0, 1, 3, 2)
            * math.sqrt(math.pi / 2.0)
            * dot_est
        )
        return correction

    def corrected_attention_scores(
        self, queries: mx.array, k_base: mx.array, key_sketch: QJLSketch
    ) -> mx.array:
        """
        Return scaled attention scores:
            (q · k_base + estimated q · residual) / sqrt(D)
        """
        if k_base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected key D={self.feature_dim}, "
                f"got {k_base.shape[-1]}"
            )
        base_dot = queries.astype(mx.float32) @ k_base.astype(
            mx.float32
        ).transpose(0, 1, 3, 2)
        residual_dot = self.estimate_residual_dot(queries, key_sketch)
        return (base_dot + residual_dot) / math.sqrt(
            float(self.feature_dim)
        )
