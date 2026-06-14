"""NumPy reference oracle for the V4 packed codec.

Produces identical PackedBlockV4 objects to the MLX CartesianCodec.
This is the ground-truth specification for the V4 binary format.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from rfsn_v10.cache.contracts import (
    PackedBlockV4,
    PackingLayout,
    Preconditioner,
    ScaleLayout,
    TensorLayout,
)


def _numpy_wht64(x: np.ndarray) -> np.ndarray:
    """Pure-NumPy reference WHT-64 (orthonormal, self-inverse).

    Matches the MLX iterative butterfly in ``_reference_wht64``.
    """
    h = x.astype(np.float32)
    n = int(h.shape[-1])
    if n < 2:
        return h
    original_shape = h.shape
    step = 1
    while step < n:
        h_reshaped = h.reshape(*h.shape[:-1], -1, 2 * step)
        a = h_reshaped[..., :step]
        b = h_reshaped[..., step:]
        h = np.concatenate([a + b, a - b], axis=-1)
        h = h.reshape(*original_shape)
        step *= 2
    return h / math.sqrt(n)


def _numpy_hash_signs(
    x: np.ndarray, seed: int = 42, layer_id: int = 0, stream_id: str = ""
) -> np.ndarray:
    """Pure-NumPy reference hash signs (Murmur32-avalanche-v1).

    Matches ``_reference_hash_signs`` exactly.
    """
    flat = x.reshape(-1)
    n = flat.size

    # Mix layer_id and stream_id into the seed (same algorithm as MLX)
    stream_hash = 0
    for ch in stream_id:
        stream_hash = (stream_hash * 31 + ord(ch)) & 0xFFFFFFFF
    mixed = np.uint32(seed)
    mixed = np.uint32(mixed ^ np.uint32((layer_id * 0x9E3779B9) & 0xFFFFFFFF))
    mixed = np.uint32(mixed ^ np.uint32(stream_hash & 0xFFFFFFFF))
    seed_val = int(mixed) & 0xFFFFFFFF

    indices = np.arange(n, dtype=np.uint32)
    state = (indices ^ seed_val) & 0xFFFFFFFF
    state = (state + 0x9E3779B9) & 0xFFFFFFFF
    state = state ^ (state >> 16)
    state = (state * 0x85EBCA6B) & 0xFFFFFFFF
    state = state ^ (state >> 13)
    state = (state * 0xC2B2AE35) & 0xFFFFFFFF
    state = state ^ (state >> 16)
    signs = np.where((state & 1) == 1, -1.0, 1.0).astype(x.dtype)
    return (flat * signs).reshape(x.shape)


def _numpy_dtype_name(dtype: Any) -> str:
    """Mirror MLX dtype naming."""
    name = str(dtype)
    if "float16" in name:
        return "float16"
    if "float32" in name or "float" in name:
        return "float32"
    if "bfloat16" in name:
        return "bfloat16"
    return name


def _vector_aligned_pack_numpy(codes_bhtd: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes with per-vector alignment (NumPy version).

    Matches ``CartesianCodec._vector_aligned_pack`` exactly.
    """
    B, H, T, D = codes_bhtd.shape
    codes_per_word = 32 // bits
    words_per_vector = math.ceil(D / codes_per_word)
    padded_D = words_per_vector * codes_per_word

    if padded_D > D:
        pad_shape = list(codes_bhtd.shape)
        pad_shape[-1] = padded_D - D
        codes_bhtd = np.concatenate(
            [codes_bhtd, np.zeros(pad_shape, dtype=np.uint32)],
            axis=-1,
        )

    grouped = codes_bhtd.reshape(B, H, T, words_per_vector, codes_per_word)
    shifts = np.arange(codes_per_word, dtype=np.uint32) * bits
    packed = np.sum(
        grouped.astype(np.uint32) << shifts.reshape(1, 1, 1, 1, -1),
        axis=-1,
    )
    return packed


def _vector_aligned_unpack_numpy(
    packed: np.ndarray, bits: int, head_dim: int
) -> np.ndarray:
    """Unpack vector-aligned uint32 words back to individual codes (NumPy).

    Matches ``CartesianCodec._vector_aligned_unpack`` exactly.
    """
    B, H, T, words_per_vector = packed.shape
    codes_per_word = 32 // bits
    padded_D = words_per_vector * codes_per_word

    packed_view = packed.reshape(B, H, T, words_per_vector, 1).astype(np.uint32)
    shifts = (np.arange(codes_per_word, dtype=np.uint32) * bits).reshape(1, 1, 1, 1, -1)
    mask = np.array((1 << bits) - 1, dtype=np.uint32)
    codes = (packed_view >> shifts) & mask
    codes = codes.reshape(B, H, T, padded_D)
    if padded_D > head_dim:
        codes = codes[..., :head_dim]
    return codes


class NumpyCartesianCodec:
    """NumPy-only codec that mirrors ``CartesianCodec`` exactly.

    Produces ``PackedBlockV4`` objects with the same packed codes, scales,
    and metadata as the MLX implementation.
    """

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        eps: float = 1e-8,
        use_wht: bool = True,
        sign_seed: int = 42,
    ) -> None:
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16]; got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        if group_size % 64 != 0:
            raise ValueError(
                f"group_size must be a multiple of 64 for vector alignment; got {group_size}"
            )
        if use_wht and group_size != 64:
            raise ValueError(
                f"canonical WHT path requires group_size=64; got {group_size}"
            )
        self.bits = bits
        self.group_size = group_size
        self.eps = eps
        self.use_wht = use_wht
        self.sign_seed = sign_seed
        self.qmax = (1 << (bits - 1)) - 1

    @property
    def codec_signature(self) -> str:
        """Canonical signature for decoder compatibility verification."""
        from .contracts import (
            PackingLayout,
            Preconditioner,
            ScaleLayout,
            TensorLayout,
        )

        preconditioner = (
            Preconditioner.WHT64_HASH_SIGN_V1
            if self.use_wht
            else Preconditioner.NONE
        )
        sign_algorithm = "murmur32-avalanche-v1"
        return (
            f"rfsn-v4-{self.bits}-{self.group_size}-"
            f"{preconditioner.value}-{sign_algorithm}-{self.sign_seed}-"
            f"{PackingLayout.VECTOR_ALIGNED_UINT32_V4.value}-"
            f"{ScaleLayout.BHTG_V4.value}-"
            f"{TensorLayout.BHTD.value}"
        )

    def encode_bhtd(
        self,
        x: np.ndarray,
        *,
        logical_start: int = 0,
        layer_id: int = 0,
        stream_id: str = "",
    ) -> PackedBlockV4:
        """Quantize and pack a BHTD tensor with vector-aligned codes.

        Returns a PackedBlockV4 that is byte-identical to the MLX codec.
        """
        if x.ndim != 4:
            raise ValueError("expected BHTD input")
        B, H, T, D = x.shape
        original_dtype_str = _numpy_dtype_name(x.dtype)
        original_value_count = int(B * H * T * D)

        # Pad feature axis to multiple of group_size
        pad = (self.group_size - (D % self.group_size)) % self.group_size
        if pad:
            pad_shape = [B, H, T, pad]
            x_padded = np.concatenate(
                [x.astype(np.float32), np.zeros(pad_shape, dtype=np.float32)],
                axis=-1,
            )
        else:
            x_padded = x.astype(np.float32)

        padded_D = D + pad
        groups_per_vector = padded_D // self.group_size

        # Reshape to (B, H, T, groups_per_vector, group_size)
        grouped = x_padded.reshape(B, H, T, groups_per_vector, self.group_size)

        # Optional WHT + deterministic signs
        preconditioner = Preconditioner.NONE
        if self.use_wht:
            grouped = _numpy_wht64(grouped)
            preconditioner = Preconditioner.WHT64_HASH_SIGN_V1
        if self.sign_seed != 0:
            grouped = _numpy_hash_signs(
                grouped, self.sign_seed, layer_id=layer_id, stream_id=stream_id
            )

        # Per-vector scales: max over group_size axis
        max_abs = np.maximum(
            np.max(np.abs(grouped), axis=-1),
            np.array(self.eps, dtype=np.float32),
        )
        scale_bhtg = max_abs / float(self.qmax)

        # Quantize
        q_signed = np.round(grouped / scale_bhtg[..., None])
        q_signed = np.clip(q_signed, -self.qmax, self.qmax)
        codes_bhtd = (q_signed + self.qmax).astype(np.uint32).reshape(B, H, T, padded_D)

        # Vector-aligned packing
        if self.bits <= 8:
            packed = _vector_aligned_pack_numpy(codes_bhtd, self.bits)
            codes_per_word = 32 // self.bits
            words_per_vector = math.ceil(padded_D / codes_per_word)
            padded_value_count = int(B * H * T * words_per_vector * codes_per_word)
        else:
            packed = codes_bhtd
            codes_per_word = 1
            words_per_vector = padded_D
            padded_value_count = original_value_count

        block = PackedBlockV4(
            packed_codes=packed,
            scales=scale_bhtg.astype(np.float32),
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=preconditioner,
            batch_size=B,
            n_kv_heads=H,
            token_count=T,
            head_dim=D,
            logical_start=logical_start,
            logical_end=logical_start + T,
            bits=self.bits,
            group_size=self.group_size,
            groups_per_vector=groups_per_vector,
            codes_per_word=codes_per_word,
            words_per_vector=words_per_vector,
            original_value_count=original_value_count,
            padded_value_count=padded_value_count,
            original_dtype=original_dtype_str,
            sign_seed=self.sign_seed if self.sign_seed != 0 else 0,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=layer_id,
            stream_id=stream_id,
            codec_signature=self.codec_signature,
        )
        block.validate()
        return block

    def decode_bhtd(self, block: PackedBlockV4) -> np.ndarray:
        """Reconstruct a BHTD tensor from a PackedBlockV4 (NumPy)."""
        if block.format_version != 4:
            raise ValueError(f"Unsupported PackedBlock version: {block.format_version}")
        if block.bits != self.bits:
            raise ValueError(f"Block bits={block.bits}, codec bits={self.bits}")

        # Strict signature validation
        if block.codec_signature and block.codec_signature != self.codec_signature:
            raise ValueError(
                f"Codec signature mismatch: block has {block.codec_signature}, "
                f"codec expects {self.codec_signature}"
            )

        # Convert MLX arrays to NumPy if needed
        packed_codes = block.packed_codes
        scales = block.scales
        if hasattr(packed_codes, "__class__") and str(type(packed_codes)).startswith("<class 'mlx"):
            packed_codes = np.array(packed_codes)
        if hasattr(scales, "__class__") and str(type(scales)).startswith("<class 'mlx"):
            scales = np.array(scales)

        # Unpack codes
        if block.bits <= 8:
            codes_bhtd = _vector_aligned_unpack_numpy(
                packed_codes, block.bits, block.head_dim
            )
        else:
            codes_bhtd = packed_codes

        B, H, T, D = codes_bhtd.shape
        padded_D = D
        pad = (self.group_size - (padded_D % self.group_size)) % self.group_size
        if pad:
            codes_bhtd = np.concatenate(
                [codes_bhtd, np.zeros((B, H, T, pad), dtype=np.uint32)],
                axis=-1,
            )
            padded_D = padded_D + pad

        groups_per_vector = padded_D // self.group_size
        grouped = codes_bhtd.reshape(B, H, T, groups_per_vector, self.group_size)
        qmax = (1 << (block.bits - 1)) - 1
        q_signed = grouped.astype(np.float32) - float(qmax)

        scales_bhtg = scales
        if scales_bhtg.ndim == 1:
            expected = B * H * T * groups_per_vector
            if int(scales_bhtg.size) != expected:
                raise ValueError(
                    f"Flat scales size {scales_bhtg.size} != expected {expected}"
                )
            scales_bhtg = scales_bhtg.reshape(B, H, T, groups_per_vector)

        restored = q_signed * scales_bhtg[..., None]

        if block.sign_seed != 0:
            restored = _numpy_hash_signs(
                restored,
                block.sign_seed,
                layer_id=getattr(block, "layer_id", 0),
                stream_id=getattr(block, "stream_id", ""),
            )
        if block.preconditioner == Preconditioner.WHT64_HASH_SIGN_V1:
            restored = _numpy_wht64(restored)

        flat_restored = restored.reshape(B, H, T, padded_D)
        if block.num_elements > 0:
            expected_elements = B * H * T * block.head_dim
            if block.num_elements == expected_elements:
                flat_restored = flat_restored[..., : block.head_dim]
            elif block.num_elements < int(flat_restored.size):
                flat_restored = flat_restored.reshape(-1)[: block.num_elements]
                flat_restored = flat_restored.reshape(B, H, T, block.head_dim)

        if block.original_dtype:
            if block.original_dtype == "float16":
                flat_restored = flat_restored.astype(np.float16)
            elif block.original_dtype == "bfloat16":
                flat_restored = flat_restored.astype(np.float32)
            elif block.original_dtype == "float32":
                flat_restored = flat_restored.astype(np.float32)

        return flat_restored
