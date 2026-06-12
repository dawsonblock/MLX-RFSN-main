"""Reference Polar quantizer / dequantizer — pure MLX, no hidden mutation."""
from __future__ import annotations

from typing import Any

from .codebooks import get_default_codebook_registry
from .contracts import QuantizedVectors
from .rotations import get_default_rotation_registry

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


class PolarQuantizer:
    """Reference implementation of Polar quantization.

    Pipeline for every vector x:
        norm = max(||x||_2, epsilon)
        unit = x / norm
        rotated = unit @ R.T
        index[d] = quantize(rotated[d], codebook)

    Reconstruction:
        x_hat = codebook[indices] @ R * norm
    """

    def __init__(
        self,
        bits: int,
        head_dim: int,
        rotation_seed: int,
        registry: Any = None,
        codebook_version: str = "polar_lm_v1",
        epsilon: float = 1e-8,
    ) -> None:
        if mx is None:
            raise RuntimeError("MLX is not installed")
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
        if head_dim not in (64, 128):
            raise ValueError(f"head_dim must be 64 or 128; got {head_dim}")

        self.bits = bits
        self.head_dim = head_dim
        self.rotation_seed = rotation_seed
        self.epsilon = epsilon

        self._rot_registry = registry or get_default_rotation_registry()
        self._rot_registry.validate(head_dim, rotation_seed)

        self._R = self._rot_registry.get(head_dim, rotation_seed)
        self._R_T = self._rot_registry.get_transpose(head_dim, rotation_seed)
        self._rotation_id = f"R_{head_dim}_seed{rotation_seed}"

        self._cb_registry = get_default_codebook_registry(codebook_version)
        self._codebook_id = self._cb_registry.codebook_id(bits)

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(self, x: Any) -> QuantizedVectors:
        """Quantize vectors x of shape (*, head_dim).

        Returns a structured :class:`QuantizedVectors` with indices and norms.
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # Reject NaN / inf
        self._assert_finite(x)

        # Compute norms
        norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        safe_norms = mx.maximum(norms, self.epsilon)

        # Normalize
        unit = x / safe_norms

        # Rotate
        rotated = unit @ self._R_T

        # Quantize each coordinate independently
        indices = self._cb_registry.quantize(rotated, self.bits)

        return QuantizedVectors(
            indices=indices,
            norms=safe_norms.squeeze(-1),
            original_dim=self.head_dim,
            bits=self.bits,
            rotation_id=self._rotation_id,
            codebook_id=self._codebook_id,
        )

    def dequantize(self, qv: QuantizedVectors) -> Any:
        """Reconstruct vectors from quantized representation.

        Returns array of shape (*, head_dim).
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # Map indices back to centroids
        centroids = self._cb_registry.dequantize(qv.indices, qv.bits)

        # Inverse rotation
        recon = centroids @ self._R

        # Rescale by norms
        norms = qv.norms[..., None]
        return recon * norms

    def reconstruct(self, x: Any) -> Any:
        """Round-trip quantize + dequantize for testing."""
        return self.dequantize(self.quantize(x))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _assert_finite(self, x: Any) -> None:
        """Raise ValueError if x contains NaN or infinity."""
        if mx is None:
            return
        if mx.any(mx.isnan(x)).item():
            raise ValueError("Input contains NaN")
        if mx.any(mx.isinf(x)).item():
            raise ValueError("Input contains infinity")
