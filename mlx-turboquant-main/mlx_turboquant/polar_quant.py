"""
PolarQuant: Stage 1 of TurboQuant.

Applies a fixed random orthogonal rotation to unit-normalized vectors,
then quantizes each coordinate using precomputed Lloyd-Max codebooks.

The key insight: after rotation, each coordinate follows a known Beta distribution,
enabling data-oblivious quantization with near-optimal distortion.
"""

import mlx.core as mx
import numpy as np

from .codebooks import load_codebook


class PolarQuant:
    """PolarQuant quantizer for vectors of a fixed dimension.

    Args:
        bits: Bits per coordinate (1-4). Total storage = bits * dim per vector.
        dim: Vector dimension (must match precomputed codebooks).
        seed: Random seed for rotation matrix generation.
    """

    def __init__(self, bits: int, dim: int, seed: int = 42):
        self.bits = bits
        self.dim = dim
        self.n_levels = 2**bits

        # Load precomputed Lloyd-Max codebook
        self.centroids, self.boundaries = load_codebook(bits, dim)

        # Generate fixed random orthogonal rotation matrix
        self.rotation = _generate_rotation_matrix(dim, seed)
        # Transpose for inverse rotation (orthogonal: R^-1 = R^T)
        self.rotation_t = self.rotation.T

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize vectors using PolarQuant.

        Args:
            vectors: (..., dim) float array

        Returns:
            indices: (..., dim) uint8 array of codebook indices
            norms: (..., 1) float array of vector norms
        """
        # Store norms
        norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        # Normalize to unit sphere (avoid div by zero)
        unit = vectors / mx.maximum(norms, 1e-8)

        # Apply rotation: unit @ R^T (each row is a vector)
        rotated = unit @ self.rotation_t

        # Quantize: binary search on sorted boundaries (avoids O(d*n_levels) broadcast)
        # boundaries[1:-1] are the decision boundaries between centroids
        inner_bounds = self.boundaries[1:-1]  # (n_levels - 1,)
        # Digitize: count how many boundaries each value exceeds
        # For n_levels centroids, there are n_levels-1 inner boundaries
        indices = mx.zeros(rotated.shape, dtype=mx.uint8)
        for i in range(self.n_levels - 1):
            indices = indices + (rotated > inner_bounds[i]).astype(mx.uint8)

        return indices, norms

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Reconstruct vectors from quantized representation.

        Args:
            indices: (..., dim) uint8 codebook indices
            norms: (..., 1) vector norms

        Returns:
            reconstructed: (..., dim) float array
        """
        # Look up centroids
        rotated_recon = self.centroids[indices]

        # Inverse rotation: recon @ R
        unit_recon = rotated_recon @ self.rotation

        # Rescale by norms
        return unit_recon * norms

    def quantize_and_reconstruct(self, vectors: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize and immediately reconstruct. Returns (reconstructed, indices, norms)."""
        indices, norms = self.quantize(vectors)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms


def _generate_rotation_matrix(dim: int, seed: int) -> mx.array:
    """Generate a random orthogonal matrix via QR decomposition of Gaussian matrix.

    This is the Haar-distributed random orthogonal matrix used in PolarQuant.
    Generated once, reused for all vectors of the same dimension.
    """
    rng = np.random.RandomState(seed)
    gaussian = rng.randn(dim, dim).astype(np.float32)
    q, r = np.linalg.qr(gaussian)
    # Ensure deterministic sign (make diagonal of R positive)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph[np.newaxis, :]
    return mx.array(q)
