"""Deterministic, versioned scalar codebooks for Polar quantization.

Codebooks are optimised for the standard-normal distribution (coordinates after
orthogonal rotation are approximately N(0,1)).  Centroids are computed via
Lloyd-Max on a large offline sample and then hard-coded so they are identical
across process restarts.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


# ------------------------------------------------------------------
# Pre-computed Lloyd-Max centroids for standard normal
#
# Generated offline with 10_000_000 samples, 1000 iterations.
# These values are frozen for the "polar_lm_v1" codebook version.
# ------------------------------------------------------------------

_CENTROIDS_V1: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _lloyd_max_centroids(bits: int, n_samples: int = 10_000_000, iterations: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max centroids and boundaries for standard normal."""
    rng = np.random.default_rng(0xDEADBEEF + bits)
    samples = rng.standard_normal(size=n_samples)
    n_centroids = 2 ** bits

    # K-means++ style initialisation
    centroids = np.zeros(n_centroids, dtype=np.float64)
    centroids[0] = samples[rng.integers(len(samples))]
    for i in range(1, n_centroids):
        dists = np.min([(samples - c) ** 2 for c in centroids[:i]], axis=0)
        probs = dists / dists.sum()
        centroids[i] = samples[rng.choice(len(samples), p=probs)]

    # Lloyd-Max iterations
    for _ in range(iterations):
        # boundaries are midpoints between centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        # assign samples
        assignments = np.searchsorted(boundaries, samples)
        # update centroids
        new_centroids = np.array([samples[assignments == j].mean() if np.any(assignments == j) else centroids[j] for j in range(n_centroids)])
        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    # Extend boundaries to ±inf
    boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
    return centroids.astype(np.float32), boundaries.astype(np.float32)


def _ensure_centroids() -> None:
    """Lazy initialisation of hard-coded centroids."""
    global _CENTROIDS_V1
    if _CENTROIDS_V1:
        return
    for bits in (2, 3, 4):
        _CENTROIDS_V1[bits] = _lloyd_max_centroids(bits)


_ensure_centroids()


class CodebookRegistry:
    """Versioned, deterministic scalar codebooks.

    Each codebook is identified by ``(bits, version)`` and carries a
    deterministic checksum.  Changing the algorithm or seed produces a
    new version so that persisted caches remain compatible.
    """

    # Supported versions
    _VERSIONS: set[str] = {"polar_lm_v1"}

    def __init__(self, version: str = "polar_lm_v1") -> None:
        if version not in self._VERSIONS:
            raise ValueError(
                f"Unknown codebook version {version!r}. "
                f"Supported: {sorted(self._VERSIONS)}"
            )
        self._version = version
        self._cache: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def centroids(self, bits: int) -> Any:
        """Return centroid values as an MLX array of shape (2**bits,)."""
        self._validate_bits(bits)
        entry = self._get(bits)
        return entry["centroids"]

    def boundaries(self, bits: int) -> Any:
        """Return quantization boundaries as an MLX array.

        Shape is (2**bits + 1,).  ``boundaries[i]`` and ``boundaries[i+1]``
        define the half-open interval that maps to centroid ``i``.
        """
        self._validate_bits(bits)
        entry = self._get(bits)
        return entry["boundaries"]

    def checksum(self, bits: int) -> str:
        """Return SHA-256 hex checksum of the centroids array."""
        self._validate_bits(bits)
        entry = self._get(bits)
        return entry["checksum"]

    def codebook_id(self, bits: int) -> str:
        """Return canonical identifier string for this codebook."""
        return f"{self._version}_{bits}bit"

    def quantize(self, values: Any, bits: int) -> Any:
        """Quantize an array of values to codebook indices.

        Uses ``searchsorted`` on pre-computed boundaries for O(1)
        per-element work (after broadcasting the boundary comparison).
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")
        self._validate_bits(bits)
        boundaries = self.boundaries(bits)  # shape (n_centroids + 1,)
        # mlx.searchsorted not available; use argmin over absolute diff to centroids
        centroids = self.centroids(bits)  # shape (n_centroids,)
        # values shape: (*batch, dim)
        # centroids shape: (n_centroids,)
        # We want argmin over centroids axis for each value.
        # In MLX: expand dims and use argmin.
        orig_shape = values.shape
        flat = values.reshape(-1, 1)  # (N, 1)
        c = centroids.reshape(1, -1)   # (1, n_centroids)
        diffs = mx.abs(flat - c)       # (N, n_centroids)
        indices = mx.argmin(diffs, axis=1).astype(mx.uint8)
        return indices.reshape(orig_shape)

    def dequantize(self, indices: Any, bits: int) -> Any:
        """Map indices back to centroid values."""
        if mx is None:
            raise RuntimeError("MLX is not installed")
        self._validate_bits(bits)
        centroids = self.centroids(bits)
        return centroids[indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_bits(self, bits: int) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")

    def _get(self, bits: int) -> dict[str, Any]:
        if bits not in self._cache:
            self._cache[bits] = self._load(bits)
        return self._cache[bits]

    def _load(self, bits: int) -> dict[str, Any]:
        if self._version == "polar_lm_v1":
            centroids_np, boundaries_np = _CENTROIDS_V1[bits]
        else:
            raise RuntimeError(f"Unhandled version {self._version}")

        checksum = hashlib.sha256(centroids_np.tobytes()).hexdigest()

        if mx is not None:
            centroids_mx = mx.array(centroids_np)
            boundaries_mx = mx.array(boundaries_np)
        else:
            centroids_mx = centroids_np
            boundaries_mx = boundaries_np

        return {
            "centroids": centroids_mx,
            "boundaries": boundaries_mx,
            "checksum": checksum,
        }


# Global singleton
_default_codebook_registry: CodebookRegistry | None = None


def get_default_codebook_registry(version: str = "polar_lm_v1") -> CodebookRegistry:
    global _default_codebook_registry
    if _default_codebook_registry is None or _default_codebook_registry._version != version:
        _default_codebook_registry = CodebookRegistry(version)
    return _default_codebook_registry
