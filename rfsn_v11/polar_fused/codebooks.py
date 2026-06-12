"""Deterministic, versioned scalar codebooks for Polar quantization.

Codebooks are optimised for the standard-normal distribution (coordinates after
orthogonal rotation are approximately N(0,1)).  Centroids are pre-computed and
hard-coded so they are identical across process restarts.
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
# Pre-computed Lloyd-Max centroids for standard normal (polar_lm_v1)
#
# Computed offline with 1_000_000 samples, 200 iterations.
# ------------------------------------------------------------------

_CENTROIDS_V1: dict[int, np.ndarray] = {
    2: np.array([
        -1.5032894611358643, -0.45074549317359924,
        0.4528157711029053, 1.5073493719100952,
    ], dtype=np.float32),
    3: np.array([
        -2.137267827987671, -1.3348439931869507, -0.7523118853569031, -0.24255864322185516,
        0.24621115624904633, 0.7544039487838745, 1.3418755531311035, 2.146153211593628,
    ], dtype=np.float32),
    4: np.array([
        -2.745058059692383, -2.0904152393341064, -1.6391898393630981, -1.2766743898391724,
        -0.9614418745040894, -0.6744636297225952, -0.40526825189590454, -0.14460529386997223,
        0.11706199496984482, 0.3830487132072449, 0.6579284071922302, 0.9480444192886353,
        1.2650624513626099, 1.6254150867462158, 2.0742762088775635, 2.73093843460083,
    ], dtype=np.float32),
}


# Pre-computed boundaries (midpoints between centroids, extended to ±inf)
_BOUNDARIES_V1: dict[int, np.ndarray] = {}


def _ensure_boundaries() -> None:
    """Lazy init of boundaries from centroids."""
    global _BOUNDARIES_V1
    if _BOUNDARIES_V1:
        return
    for bits, cents in _CENTROIDS_V1.items():
        mids = (cents[:-1] + cents[1:]) / 2.0
        bounds = np.concatenate([[-np.inf], mids, [np.inf]]).astype(np.float32)
        _BOUNDARIES_V1[bits] = bounds


_ensure_boundaries()


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

        Uses argmin over absolute differences to centroids.
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")
        self._validate_bits(bits)
        centroids = self.centroids(bits)  # shape (n_centroids,)
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
            centroids_np = _CENTROIDS_V1[bits]
            boundaries_np = _BOUNDARIES_V1[bits]
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
