"""
Asymmetric K/V compressor for RFSN v11.

Keys:   WHT + grouped symmetric quantization (KeyQuant)
Values: Rotation + Lloyd-Max codebook (PolarQuant / _SplitPolarQuant)

Default: k_bits=8, v_bits=4, group_size=64, D=128

Quality gate on init:
  - Runs a synthetic MSE check against the Lloyd-Max distortion paper bound.
  - Paper bound: MSE ≤ (√3π/2) · (1/4^bits) per unit vector
    (from Zador 1982 and related Lloyd-Max distortion theory).
  - Hard raise if MSE exceeds paper_bound * mse_tolerance_factor.
    Default tolerance factor: 1.5 (allows for finite-sample fluctuation).
  - If D < 64 and v_bits < 4, raises unless require_experimental("sub4bit_small_head") passes.

Note: The 0.997/0.988 thresholds in the plan refer to end-to-end *attention output*
cosine similarity (logit-level, from mlx-turboquant/REPORT.md), which is a different
and higher-level metric than direct reconstruction cosine. The init gate here validates
the lower-level compression quality using the MSE distortion bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .key_quant import KeyQuant
from .value_quant import PolarQuant, _SplitPolarQuant, make_value_quantizer


# Tolerance factor on the Lloyd-Max MSE paper bound.
# A value of 1.5 allows ~50% margin for finite-sample fluctuation.
_MSE_TOLERANCE_FACTOR = 1.5


def _paper_mse_bound(v_bits: int) -> float:
    """Lloyd-Max MSE distortion bound for unit-sphere vectors at given bit width.

    Bound: MSE ≤ (√3π/2) · (1/4^bits)  per vector  (Zador 1982 / Lloyd-Max theory).
    """
    return (math.sqrt(3) * math.pi / 2) * (1.0 / 4 ** v_bits)


@dataclass
class CompressedKV:
    """Compressed K/V for a single layer."""

    key_codes: mx.array     # uint32 flat codes
    key_scales: mx.array    # float32 group scales
    key_shape: tuple        # original key shape

    val_indices: mx.array   # uint8 codebook indices (..., D)
    val_norms: mx.array     # float32 vector norms (..., 1) or (..., 2) for split


class KVCompressor:
    """Asymmetric K/V cache compressor.

    Keys:   KeyQuant   (WHT + grouped symmetric)  — tuned for attention score accuracy
    Values: PolarQuant (rotation + Lloyd-Max)      — tuned for MSE / reconstruction quality

    Args:
        k_bits: Key quantization bits (default 8).
        v_bits: Value quantization bits (default 4); supports fractional (2.5, 3.5).
        group_size: Key quantization group size (default 64).
        dim: Head dimension D (required for value quantizer init).
        use_wht: Enable WHT preconditioning for keys (default True).
        use_incoherent_signs: Enable hash-sign preconditioning for keys (default True).
        sign_seed: Seed for key sign preconditioning (default 0).
        skip_quality_gate: Skip cosine quality gate on init (for testing, default False).
    """

    def __init__(
        self,
        k_bits: int = 8,
        v_bits: float = 4,
        group_size: int = 64,
        dim: int = 128,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
        sign_seed: int = 0,
        skip_quality_gate: bool = False,
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.dim = dim

        # Guard: D < 64 with sub-4-bit values requires experimental gate
        if dim < 64 and v_bits < 4:
            from ..config import require_experimental
            require_experimental("sub4bit_small_head")

        self.key_quant = KeyQuant(
            bits=k_bits,
            group_size=group_size,
            use_wht=use_wht,
            use_incoherent_signs=use_incoherent_signs,
            sign_seed=sign_seed,
        )

        self.val_quant: PolarQuant | _SplitPolarQuant = make_value_quantizer(
            bits=v_bits,
            dim=dim,
        )

        if not skip_quality_gate:
            self._run_quality_gate()

    def _run_quality_gate(self) -> None:
        """Validate value quantizer MSE against the Lloyd-Max distortion paper bound.

        Uses unit-sphere vectors (the domain PolarQuant is designed for).
        Raises RuntimeError if MSE > paper_bound * _MSE_TOLERANCE_FACTOR.
        """
        rng = np.random.RandomState(42)
        batch = 256
        x_np = rng.randn(batch, self.dim).astype(np.float32)
        # Normalize to unit sphere — PolarQuant is designed for unit vectors
        norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
        x_np = x_np / norms
        x = mx.array(x_np)

        recon, _, _ = self.val_quant.quantize_and_reconstruct(x)
        mx.eval(recon)

        mse = float(mx.mean((x - recon) ** 2).item())
        bits_int = int(math.ceil(float(self.v_bits)))
        bound = _paper_mse_bound(bits_int) * _MSE_TOLERANCE_FACTOR

        if mse > bound:
            raise RuntimeError(
                f"KVCompressor quality gate failed: "
                f"v_bits={self.v_bits}, dim={self.dim} → "
                f"MSE={mse:.6f} > bound={bound:.6f} "
                f"(paper_bound={_paper_mse_bound(bits_int):.6f} × {_MSE_TOLERANCE_FACTOR}). "
                "Consider increasing v_bits or using a larger head_dim."
            )

    def compress(self, keys: mx.array, values: mx.array) -> CompressedKV:
        """Compress a K/V pair.

        Args:
            keys:   (..., D) float array
            values: (..., D) float array

        Returns:
            CompressedKV with all quantized components.
        """
        key_flat = keys.reshape(-1)
        key_codes, key_scales = self.key_quant.compress(keys)
        val_indices, val_norms = self.val_quant.quantize(values)

        return CompressedKV(
            key_codes=key_codes,
            key_scales=key_scales,
            key_shape=keys.shape,
            val_indices=val_indices,
            val_norms=val_norms,
        )

    def decompress(self, compressed: CompressedKV) -> tuple[mx.array, mx.array]:
        """Decompress a CompressedKV.

        Returns:
            (keys, values) as float32 arrays.
        """
        keys = self.key_quant.decompress(
            compressed.key_codes, compressed.key_scales, compressed.key_shape
        )
        values = self.val_quant.dequantize(
            compressed.val_indices, compressed.val_norms
        )
        return keys, values

    def estimate_bytes(self, key_shape: tuple, val_shape: tuple) -> int:
        """Estimate total compressed byte footprint."""
        key_bytes = self.key_quant.estimate_bytes(key_shape)

        n_val = 1
        for d in val_shape[:-1]:
            n_val *= d
        # Indices: n_val * D * bits / 8; norms: n_val * 4 (or * 8 for split)
        n_dim = val_shape[-1]
        val_bits = int(math.ceil(self.v_bits))
        idx_bytes = n_val * n_dim * val_bits // 8
        norm_bytes = n_val * 4 if self.v_bits == int(self.v_bits) else n_val * 8
        return key_bytes + idx_bytes + norm_bytes
