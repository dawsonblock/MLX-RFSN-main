"""
Value quantization for RFSN v11 KV cache.

Implements Lloyd-Max / rotation-based quantization for value vectors.

Source: mlx-turboquant-main/mlx_turboquant/polar_quant.py (PolarQuant)
        mlx-turboquant-main/mlx_turboquant/turbo_quant.py (_SplitPolarQuant)

IMPORTANT: This is NOT the same as rfsn_v10/quantization/polar_quant.py, which
implements a completely different hierarchical atan2/radius decomposition.
This module uses the simpler and better-validated rotation + Lloyd-Max codebook
approach from mlx-turboquant.

Algorithm:
  Quantize:
    1. Compute vector norm; normalize to unit sphere.
    2. Apply fixed random orthogonal rotation R: unit @ R^T
    3. After rotation, each coordinate ~ N(0, 1/dim) — data-oblivious.
    4. Assign each coordinate to nearest Lloyd-Max centroid (binary search).
    5. Return (indices, norms).
  Dequantize:
    1. Look up centroids[indices].
    2. Apply inverse rotation: recon @ R
    3. Rescale by norms.

Fractional bits (2.5, 3.5) are supported via _SplitPolarQuant: the first
`split` channels use bits+1 and the remaining use bits.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .codebooks import load_codebook


class PolarQuant:
    """PolarQuant quantizer for value vectors of a fixed dimension.

    Args:
        bits: Bits per coordinate (1-4). Total storage = bits * dim per vector.
        dim: Vector dimension (must match precomputed codebooks).
        seed: Random seed for rotation matrix generation.
    """

    def __init__(self, bits: int, dim: int, seed: int = 42):
        self.bits = bits
        self.dim = dim
        self.n_levels = 2 ** bits

        # Load precomputed Lloyd-Max codebook (cached)
        self.centroids, self.boundaries = load_codebook(bits, dim)

        # Generate fixed random orthogonal rotation matrix (Haar distribution)
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

        # Quantize: binary search on sorted boundaries
        # boundaries[1:-1] are the decision boundaries between centroids
        inner_bounds = self.boundaries[1:-1]  # (n_levels - 1,)
        # Digitize: count how many boundaries each value exceeds
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

    def quantize_and_reconstruct(
        self, vectors: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize and immediately reconstruct. Returns (reconstructed, indices, norms)."""
        indices, norms = self.quantize(vectors)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms


class _SplitPolarQuant:
    """Wraps two PolarQuant instances for fractional-bit channel split.

    First `split` channels use pq_hi (higher bits), remaining use pq_lo.
    Presents the same interface as PolarQuant.

    Example: For 3.5-bit quantization on dim=128:
        split = 64
        pq_hi = PolarQuant(bits=4, dim=64)   # first 64 channels at 4-bit
        pq_lo = PolarQuant(bits=3, dim=64)   # last  64 channels at 3-bit
        → effective rate = (64*4 + 64*3) / 128 = 3.5 bits/coord
    """

    def __init__(self, pq_hi: PolarQuant, pq_lo: PolarQuant, split: int):
        self.pq_hi = pq_hi
        self.pq_lo = pq_lo
        self.split = split
        self.dim = pq_hi.dim + pq_lo.dim

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array]:
        v_hi = vectors[..., : self.split]
        v_lo = vectors[..., self.split :]

        idx_hi, norms_hi = self.pq_hi.quantize(v_hi)
        idx_lo, norms_lo = self.pq_lo.quantize(v_lo)

        indices = mx.concatenate([idx_hi, idx_lo], axis=-1)
        # Store both norms concatenated: (..., 1) + (..., 1) -> (..., 2)
        norms = mx.concatenate([norms_hi, norms_lo], axis=-1)
        return indices, norms

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        idx_hi = indices[..., : self.split]
        idx_lo = indices[..., self.split :]
        norms_hi = norms[..., :1]
        norms_lo = norms[..., 1:2]

        v_hi = self.pq_hi.dequantize(idx_hi, norms_hi)
        v_lo = self.pq_lo.dequantize(idx_lo, norms_lo)
        return mx.concatenate([v_hi, v_lo], axis=-1)

    def quantize_and_reconstruct(
        self, vectors: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        indices, norms = self.quantize(vectors)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms


def make_value_quantizer(
    bits: float,
    dim: int,
    seed: int = 42,
) -> PolarQuant | _SplitPolarQuant:
    """Create the appropriate value quantizer for the given (possibly fractional) bit rate.

    Args:
        bits: Bits per coordinate. Integer (2-4) or fractional (2.5, 3.5).
        dim:  Vector dimension.
        seed: Random seed for rotation matrix.

    Returns:
        A PolarQuant or _SplitPolarQuant instance.
    """
    if bits == int(bits):
        return PolarQuant(bits=int(bits), dim=dim, seed=seed)

    # Fractional bits: split channels
    bits_lo = int(bits)
    bits_hi = bits_lo + 1
    split = dim // 2  # equal split for 0.5-bit fractions
    pq_hi = PolarQuant(bits=bits_hi, dim=split, seed=seed)
    pq_lo = PolarQuant(bits=bits_lo, dim=dim - split, seed=seed + 1)
    return _SplitPolarQuant(pq_hi=pq_hi, pq_lo=pq_lo, split=split)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_rotation_matrix(dim: int, seed: int) -> mx.array:
    """Generate a random orthogonal matrix via QR decomposition.

    This is the Haar-distributed random orthogonal matrix used in PolarQuant.
    Generated once per (dim, seed) pair; the caller caches the PolarQuant object.
    """
    rng = np.random.RandomState(seed)
    gaussian = rng.randn(dim, dim).astype(np.float32)
    q, r = np.linalg.qr(gaussian)
    # Ensure deterministic sign (make diagonal of R positive)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph[np.newaxis, :]
    return mx.array(q)
