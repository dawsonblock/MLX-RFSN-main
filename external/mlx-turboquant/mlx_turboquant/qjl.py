"""
QJL: Quantized Johnson-Lindenstrauss residual correction (Stage 2 of TurboQuant).

After PolarQuant (Stage 1) quantizes vectors, there is residual error.
QJL captures this residual using a random projection followed by sign-bit
quantization (1 bit per projected dimension).

The combined estimator produces UNBIASED inner product estimates:
    <q, k> ≈ <q, k_mse> + ||r|| * sqrt(π/2) / m * <S@q, sign(S@r)>

where r = k - k_mse is the quantization residual, S is the random projection.
"""

import math

import mlx.core as mx
import numpy as np


class QJL:
    """Quantized Johnson-Lindenstrauss residual correction.

    Args:
        dim: Vector dimension.
        proj_dim: Projection dimension (default: same as dim for best correction).
        seed: Random seed for projection matrix.
    """

    def __init__(self, dim: int, proj_dim: int | None = None, seed: int = 137):
        self.dim = dim
        self.proj_dim = proj_dim or dim
        self.seed = seed

        # Generate random Gaussian projection matrix S: (proj_dim, dim)
        # Scaled by 1/sqrt(proj_dim) for proper JL guarantee
        self.projection = _generate_projection_matrix(dim, self.proj_dim, seed)

        # Precompute correction scale factor: sqrt(π/2) / proj_dim
        self.correction_scale = math.sqrt(math.pi / 2.0) / self.proj_dim

    def compress_residual(
        self, original: mx.array, reconstructed: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compute and compress the quantization residual.

        Args:
            original: (..., dim) original vectors
            reconstructed: (..., dim) PolarQuant reconstruction

        Returns:
            signs: (..., proj_dim) bool array — sign bits of projected residual
            residual_norms: (..., 1) float — norms of residuals
        """
        residual = original - reconstructed
        residual_norms = mx.linalg.norm(residual, axis=-1, keepdims=True)

        # Project residual: (..., dim) @ (dim, proj_dim) → (..., proj_dim)
        projected = residual @ self.projection.T
        signs = projected > 0  # bool array

        return signs, residual_norms

    def corrected_inner_product(
        self,
        queries: mx.array,
        reconstructed_keys: mx.array,
        signs: mx.array,
        residual_norms: mx.array,
    ) -> mx.array:
        """Compute QJL-corrected inner product (attention scores).

        Args:
            queries: (B, n_heads, L_q, dim) query vectors
            reconstructed_keys: (B, n_heads, L_kv, dim) PolarQuant-reconstructed keys
            signs: (B, n_heads, L_kv, proj_dim) bool sign bits
            residual_norms: (B, n_heads, L_kv, 1) residual norms

        Returns:
            scores: (B, n_heads, L_q, L_kv) corrected attention scores
        """
        # Base inner product from Stage 1 reconstruction
        # queries @ keys^T: (B, n_heads, L_q, dim) @ (B, n_heads, dim, L_kv)
        base_scores = queries @ mx.swapaxes(reconstructed_keys, -2, -1)

        # QJL correction term
        # Project queries: (B, n_heads, L_q, dim) @ (dim, proj_dim) → (B, n_heads, L_q, proj_dim)
        projected_queries = queries @ self.projection.T

        # Convert signs to +1/-1: (B, n_heads, L_kv, proj_dim)
        sign_values = mx.where(signs, 1.0, -1.0)

        # Inner product: projected_queries @ sign_values^T
        # (B, n_heads, L_q, proj_dim) @ (B, n_heads, proj_dim, L_kv) → (B, n_heads, L_q, L_kv)
        correction = projected_queries @ mx.swapaxes(sign_values, -2, -1)

        # Scale by residual norms: (B, n_heads, L_kv, 1) → broadcast (B, n_heads, 1, L_kv)
        residual_norms_t = mx.swapaxes(residual_norms, -2, -1)
        correction = correction * residual_norms_t * self.correction_scale

        return base_scores + correction


def _generate_projection_matrix(dim: int, proj_dim: int, seed: int) -> mx.array:
    """Generate random Gaussian projection matrix for JL transform."""
    rng = np.random.RandomState(seed)
    S = rng.randn(proj_dim, dim).astype(np.float32)
    # No normalization needed — the correction_scale handles it
    return mx.array(S)
