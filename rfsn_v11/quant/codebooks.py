"""
Lloyd-Max codebook loader for RFSN v11 value quantization.

Ported from mlx-turboquant-main/mlx_turboquant/codebooks.py.

The codebooks.npz file is bundled as package data (declared in pyproject.toml
under [tool.setuptools.package-data]) and MUST be present at
  rfsn_v11/quant/data/codebooks.npz
If missing, load_codebook() raises FileNotFoundError at init time.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np


def _load_raw_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw N(0,1) codebook for given bit width from bundled npz.

    Returns:
        centroids: 1-D float32 array of shape (2**bits,)
        boundaries: 1-D float32 array of shape (2**bits + 1,)
    """
    data_path = Path(__file__).parent / "data" / "codebooks.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"codebooks.npz not found at {data_path}. "
            "Ensure the package-data entry in pyproject.toml is correct and "
            "the file is present at rfsn_v11/quant/data/codebooks.npz."
        )
    data = np.load(str(data_path))
    # Keys use format: bits{N}_centroids, bits{N}_boundaries
    centroids = data[f"bits{bits}_centroids"].astype(np.float32)
    boundaries = data[f"bits{bits}_boundaries"].astype(np.float32)
    return centroids, boundaries


@lru_cache(maxsize=32)
def load_codebook(bits: int, dim: int) -> tuple[mx.array, mx.array]:
    """Load and scale a Lloyd-Max codebook for the given bit width and dimension.

    After rotation by a random orthogonal matrix, each coordinate of a
    unit-sphere vector follows N(0, 1/dim).  Scaling codebook entries by
    1/sqrt(dim) matches this distribution.

    Results are cached (up to 32 distinct (bits, dim) pairs) to avoid
    reloading from disk on repeated calls.

    Args:
        bits: Bits per coordinate (1-4).
        dim:  Vector dimension (must match the dimension used at quantize time).

    Returns:
        scaled_centroids:  (2**bits,)   mx.array float32
        scaled_boundaries: (2**bits+1,) mx.array float32
    """
    centroids, boundaries = _load_raw_codebook(bits)
    scale = 1.0 / math.sqrt(dim)
    return mx.array(centroids * scale), mx.array(boundaries * scale)
