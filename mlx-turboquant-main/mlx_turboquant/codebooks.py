"""
Load precomputed Lloyd-Max codebooks for TurboQuant.

Codebooks are precomputed for N(0,1) and scaled by 1/sqrt(dim) at runtime.
"""

import os
from functools import lru_cache

import mlx.core as mx
import numpy as np


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_raw_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw N(0,1) codebook from disk."""
    path = os.path.join(_DATA_DIR, "codebooks.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Codebook file not found at {path}. "
            "Run: python precompute/generate_codebooks.py"
        )
    data = np.load(path)
    return data[f"bits{bits}_centroids"], data[f"bits{bits}_boundaries"]


@lru_cache(maxsize=32)
def load_codebook(bits: int, dim: int) -> tuple[mx.array, mx.array]:
    """Load Lloyd-Max codebook scaled for a given dimension.

    Codebooks are precomputed for N(0,1) and scaled by 1/sqrt(dim)
    since rotated unit vector coordinates are ~ N(0, 1/dim).

    Args:
        bits: Quantization bits per coordinate (1-4)
        dim: Head dimension

    Returns:
        centroids: (2^bits,) array of reconstruction values
        boundaries: (2^bits + 1,) array of decision boundaries
    """
    raw_centroids, raw_boundaries = _load_raw_codebook(bits)
    scale = 1.0 / np.sqrt(dim)
    centroids = mx.array((raw_centroids * scale).astype(np.float32))
    boundaries = mx.array((raw_boundaries * scale).astype(np.float32))
    return centroids, boundaries
