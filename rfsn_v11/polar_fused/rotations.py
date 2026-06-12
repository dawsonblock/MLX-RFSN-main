"""Deterministic rotation matrix generation and registry."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# MLX is optional for import-time; real usage checks availability at runtime.
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


class RotationRegistry:
    """Deterministic orthogonal rotation matrices, cached per (dim, seed, dtype).

    Each rotation is generated once via numpy QR decomposition, verified for
    orthogonality, and cached in both numpy and MLX formats.
    """

    _ORTHO_TOL: float = 1e-5

    def __init__(self) -> None:
        # Key: (dim, seed, str_dtype) -> dict with "R", "R_T", "checksum", "ortho_error"
        self._cache: dict[tuple[int, int, str], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        dim: int,
        seed: int,
        dtype: Any = None,
    ) -> mx.array:
        """Return the rotation matrix R for the given parameters.

        R is an orthogonal matrix of shape (dim, dim).  The caller should
        use ``R.T`` for the transpose.
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        str_dtype = str(dtype) if dtype is not None else "float32"
        key = (dim, seed, str_dtype)

        if key not in self._cache:
            self._cache[key] = self._generate(dim, seed, dtype)

        return self._cache[key]["R"]

    def get_transpose(
        self,
        dim: int,
        seed: int,
        dtype: Any = None,
    ) -> mx.array:
        """Return R.T (cached so we do not recompute)."""
        if mx is None:
            raise RuntimeError("MLX is not installed")

        str_dtype = str(dtype) if dtype is not None else "float32"
        key = (dim, seed, str_dtype)

        if key not in self._cache:
            self._cache[key] = self._generate(dim, seed, dtype)

        return self._cache[key]["R_T"]

    def checksum(
        self,
        dim: int,
        seed: int,
        dtype: Any = None,
    ) -> str:
        """Return SHA-256 hex checksum of the rotation matrix."""
        str_dtype = str(dtype) if dtype is not None else "float32"
        key = (dim, seed, str_dtype)

        if key not in self._cache:
            self._cache[key] = self._generate(dim, seed, dtype)

        return self._cache[key]["checksum"]

    def orthogonality_error(
        self,
        dim: int,
        seed: int,
        dtype: Any = None,
    ) -> float:
        """Return max(abs(R @ R.T - I))."""
        str_dtype = str(dtype) if dtype is not None else "float32"
        key = (dim, seed, str_dtype)

        if key not in self._cache:
            self._cache[key] = self._generate(dim, seed, dtype)

        return self._cache[key]["ortho_error"]

    def validate(
        self,
        dim: int,
        seed: int,
        dtype: Any = None,
    ) -> None:
        """Raise ValueError if orthogonality gate is violated."""
        err = self.orthogonality_error(dim, seed, dtype)
        if err > self._ORTHO_TOL:
            raise ValueError(
                f"Rotation matrix orthogonality error {err:.3e} exceeds "
                f"tolerance {self._ORTHO_TOL:.3e} for (dim={dim}, seed={seed})"
            )

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        dim: int,
        seed: int,
        dtype: Any,
    ) -> dict[str, Any]:
        """Generate rotation matrix via numpy QR decomposition."""
        rng = np.random.default_rng(seed)
        # Gaussian random matrix
        raw = rng.standard_normal(size=(dim, dim)).astype(np.float32)
        # QR decomposition; Q is orthogonal
        q, _r = np.linalg.qr(raw)
        # Householder QR can produce sign flips; fix determinism by forcing
        # the first column to have positive sum (consistent across runs).
        if q[:, 0].sum() < 0:
            q = -q

        # Convert to requested dtype
        if dtype is not None and str(dtype) != "float32":
            q_np = q.astype(np.float32)
        else:
            q_np = q

        # Convert to MLX
        if mx is not None:
            r_mx = mx.array(q_np, dtype=dtype)
            r_t_mx = r_mx.T
        else:
            r_mx = q_np
            r_t_mx = q_np.T

        # Verify orthogonality
        if mx is not None:
            identity = mx.eye(dim, dtype=mx.float32)
            prod = mx.matmul(r_mx.astype(mx.float32), r_t_mx.astype(mx.float32))
            ortho_err = float(mx.max(mx.abs(prod - identity)))
        else:
            prod = q_np @ q_np.T
            ortho_err = float(np.max(np.abs(prod - np.eye(dim, dtype=np.float32))))

        # Checksum
        checksum = hashlib.sha256(q_np.tobytes()).hexdigest()

        return {
            "R": r_mx,
            "R_T": r_t_mx,
            "checksum": checksum,
            "ortho_error": ortho_err,
        }


# Global singleton for convenience
_default_registry: RotationRegistry | None = None


def get_default_rotation_registry() -> RotationRegistry:
    """Return the process-global rotation registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RotationRegistry()
    return _default_registry
