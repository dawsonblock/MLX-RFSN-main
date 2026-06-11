"""QJL score estimation: correct attention scores using residual sketch.

Given:
  Q: query vector (D,)
  K_polar: reconstructed keys from PolarQuant (T, D)
  E: residual = K_original - K_polar (T, D)
  QJL sketch of E: packed_signs, norms, projection_seed

Score = Q @ K_polar.T + qjl_estimate(Q, E_sketch)

The qjl_estimate uses random-projection sign bits to approximate dot(Q, E_i).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from .encoder import QJLPayload, _random_projection_matrix, _unpack_signs


def qjl_dot_estimate(
    query: mx.array,
    payload: QJLPayload,
) -> mx.array:
    """Estimate dot(query, residual_i) for all i using QJL sketch.

    Args:
        query: shape (D,) float array.
        payload: QJLPayload for the residual block.

    Returns:
        shape (...,) float array of estimated dot products.
    """
    D = int(query.shape[-1])
    proj_mat = _random_projection_matrix(D, payload.proj_dim, payload.seed)

    # Project query
    q_proj = query @ proj_mat  # (proj_dim,)

    # Unpack signs
    signs = _unpack_signs(payload.packed_signs, payload.proj_dim)
    signs = signs.astype(mx.float32)  # 0/1 -> treat as +1/-1 mapping
    # Map 0 -> -1, 1 -> +1
    signs = signs * 2.0 - 1.0  # (..., proj_dim)

    # Estimate: sum_j (q_proj_j * sign_ij) * norm_i / sqrt(proj_dim)
    # The factor 1/sqrt(proj_dim) normalizes the random projection.
    # The signs approximate the direction of E_i in projected space;
    # multiplying by norm_i recovers approximate magnitude.
    scale = 1.0 / (payload.proj_dim ** 0.5)

    norms = payload.norms.astype(mx.float32)
    # Ensure norms broadcasts over the trailing proj_dim dimension
    if norms.ndim == signs.ndim - 1:
        norms = mx.expand_dims(norms, axis=-1)

    # Dot product in projected space
    proj_dots = signs @ q_proj  # (...)
    estimate = proj_dots * norms.squeeze(-1) * scale
    return estimate


def correct_scores(
    query: mx.array,
    polar_keys: mx.array,
    qjl_payload: QJLPayload,
) -> mx.array:
    """Compute QK scores with QJL residual correction.

    Args:
        query: (D,)
        polar_keys: (T, D) reconstructed from PolarQuant
        qjl_payload: QJL sketch of the residual E = K_original - K_polar

    Returns:
        scores: (T,) float array.
    """
    base_scores = query @ polar_keys.T
    correction = qjl_dot_estimate(query, qjl_payload)
    return base_scores + correction
