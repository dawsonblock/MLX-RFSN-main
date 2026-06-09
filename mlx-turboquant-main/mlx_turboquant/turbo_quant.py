"""
TurboQuant: Combined PolarQuant + QJL compressor.

Supports integer bits (2, 3, 4) and fractional bits (2.5, 3.5) via channel split.
Keys: Full TurboQuant (PolarQuant + QJL) for inner product preservation.
Values: PolarQuant only (MSE optimization sufficient for weighted sums).
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from .polar_quant import PolarQuant
from .qjl import QJL


@dataclass
class CompressedKeys:
    """Compressed key cache storage."""
    indices: mx.array      # (..., dim) uint8 — PolarQuant codebook indices
    norms: mx.array        # (..., 1) float — vector norms
    signs: mx.array        # (..., proj_dim) bool — QJL sign bits
    residual_norms: mx.array  # (..., 1) float — residual norms


@dataclass
class CompressedValues:
    """Compressed value cache storage."""
    indices: mx.array      # (..., dim) uint8 — PolarQuant codebook indices
    norms: mx.array        # (..., 1) float — vector norms


class TurboQuantCompressor:
    """Combined TurboQuant compressor for KV cache.

    Supports fractional bits (e.g., 3.5) via channel split:
    For 3.5-bit with dim=128: first 64 channels at 4-bit, last 64 at 3-bit.

    Args:
        bits: Bits per coordinate (2-4, supports 0.5 increments like 2.5, 3.5).
        dim: Head dimension.
        use_qjl: Whether to use QJL residual correction. Default False.
        key_seed: Random seed for key rotation matrix.
        value_seed: Random seed for value rotation matrix.
        qjl_seed: Random seed for QJL projection.
    """

    def __init__(
        self,
        bits: float,
        dim: int,
        use_qjl: bool | None = None,
        key_seed: int = 42,
        value_seed: int = 43,
        qjl_seed: int = 137,
    ):
        self.bits = bits
        self.dim = dim

        if use_qjl is None:
            use_qjl = False

        self.use_qjl = use_qjl
        self.fractional = (bits % 1) != 0

        if self.fractional:
            # Split channels: half at ceil(bits), half at floor(bits)
            self.bits_hi = int(math.ceil(bits))
            self.bits_lo = int(math.floor(bits))
            self.split = dim // 2  # first half gets higher bits

            self.key_pq_hi = PolarQuant(bits=self.bits_hi, dim=self.split, seed=key_seed)
            self.key_pq_lo = PolarQuant(bits=self.bits_lo, dim=dim - self.split, seed=key_seed + 100)
            self.value_pq_hi = PolarQuant(bits=self.bits_hi, dim=self.split, seed=value_seed)
            self.value_pq_lo = PolarQuant(bits=self.bits_lo, dim=dim - self.split, seed=value_seed + 100)
            # For dequantize compatibility, expose a combined interface
            self.key_pq = _SplitPolarQuant(self.key_pq_hi, self.key_pq_lo, self.split)
            self.value_pq = _SplitPolarQuant(self.value_pq_hi, self.value_pq_lo, self.split)
        else:
            bits_int = int(bits)
            if use_qjl:
                pq_bits_keys = max(1, bits_int - 1)
            else:
                pq_bits_keys = bits_int
            self.key_pq = PolarQuant(bits=pq_bits_keys, dim=dim, seed=key_seed)
            self.value_pq = PolarQuant(bits=bits_int, dim=dim, seed=value_seed)

        if self.use_qjl:
            self.qjl = QJL(dim=dim, seed=qjl_seed)

    def compress_keys(self, keys: mx.array) -> CompressedKeys:
        """Compress key vectors."""
        reconstructed, indices, norms = self.key_pq.quantize_and_reconstruct(keys)

        if self.use_qjl:
            signs, residual_norms = self.qjl.compress_residual(keys, reconstructed)
        else:
            signs = mx.zeros((*keys.shape[:-1], self.dim), dtype=mx.bool_)
            residual_norms = mx.zeros((*keys.shape[:-1], 1), dtype=keys.dtype)

        return CompressedKeys(
            indices=indices, norms=norms,
            signs=signs, residual_norms=residual_norms,
        )

    def compress_values(self, values: mx.array) -> CompressedValues:
        """Compress value vectors using PolarQuant only."""
        indices, norms = self.value_pq.quantize(values)
        return CompressedValues(indices=indices, norms=norms)

    def attention_scores(
        self, queries: mx.array, compressed_keys: CompressedKeys, scale: float
    ) -> mx.array:
        """Compute attention scores with optional QJL correction."""
        queries_scaled = queries * scale
        reconstructed_keys = self.key_pq.dequantize(
            compressed_keys.indices, compressed_keys.norms
        )

        if self.use_qjl:
            scores = self.qjl.corrected_inner_product(
                queries_scaled, reconstructed_keys,
                compressed_keys.signs, compressed_keys.residual_norms,
            )
        else:
            scores = queries_scaled @ mx.swapaxes(reconstructed_keys, -2, -1)

        return scores

    def reconstruct_values(self, compressed_values: CompressedValues) -> mx.array:
        """Reconstruct value vectors."""
        return self.value_pq.dequantize(compressed_values.indices, compressed_values.norms)


class _SplitPolarQuant:
    """Wraps two PolarQuant instances for fractional-bit channel split.

    First `split` channels use pq_hi, remaining use pq_lo.
    Presents the same interface as PolarQuant.
    """

    def __init__(self, pq_hi: PolarQuant, pq_lo: PolarQuant, split: int):
        self.pq_hi = pq_hi
        self.pq_lo = pq_lo
        self.split = split
        self.dim = pq_hi.dim + pq_lo.dim

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array]:
        v_hi = vectors[..., :self.split]
        v_lo = vectors[..., self.split:]

        idx_hi, norms_hi = self.pq_hi.quantize(v_hi)
        idx_lo, norms_lo = self.pq_lo.quantize(v_lo)

        indices = mx.concatenate([idx_hi, idx_lo], axis=-1)
        # Store both norms concatenated: ((..., 1), (..., 1)) -> (..., 2)
        norms = mx.concatenate([norms_hi, norms_lo], axis=-1)
        return indices, norms

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        idx_hi = indices[..., :self.split]
        idx_lo = indices[..., self.split:]
        norms_hi = norms[..., :1]
        norms_lo = norms[..., 1:2]

        v_hi = self.pq_hi.dequantize(idx_hi, norms_hi)
        v_lo = self.pq_lo.dequantize(idx_lo, norms_lo)
        return mx.concatenate([v_hi, v_lo], axis=-1)

    def quantize_and_reconstruct(self, vectors: mx.array):
        indices, norms = self.quantize(vectors)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms
