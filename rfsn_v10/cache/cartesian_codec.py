"""Stateless Cartesian codec for K8/V5 grouped symmetric quantization.

Extracts the proven v10 primitives:
  * Deterministic signs (WHT-64)
  * Grouped symmetric quantization
  * Bit packing via BitPackedQuantizer
  * Exact payload accounting

The codec is stateless: all context (scales, shapes, bits) is carried
in PackedBlock.  No global mutable state.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.compat import mx

from .contracts import (
    PackedBlock,
    PackedBlockV4,
    PackingLayout,
    Preconditioner,
    ScaleLayout,
    TensorLayout,
)


class CartesianCodec:
    """Encode / decode grouped symmetric Cartesian blocks.

    Parameters
    ----------
    bits
        Quantization bit width (8 for keys, 5 for values).
    group_size
        Number of elements sharing one scale (64).
    eps
        Minimum scale to avoid division by zero.
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
        """Canonical signature for decoder compatibility verification.

        Includes every field that affects the wire format so that decode
        can reject blocks created with an incompatible configuration.
        """
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

    # ------------------------------------------------------------------
    # Vector-aligned helpers
    # ------------------------------------------------------------------

    def _vector_aligned_pack(self, codes_bhtd: Any, bits: int) -> Any:
        """Pack integer codes with per-vector alignment.

        Parameters
        ----------
        codes_bhtd
            uint32 array of shape (B, H, T, D) where each element is a code.
        bits
            Bit width per code.

        Returns
        -------
        packed
            uint32 array of shape (B, H, T, words_per_vector).
        """
        B, H, T, D = codes_bhtd.shape
        codes_per_word = 32 // bits
        words_per_vector = math.ceil(D / codes_per_word)
        padded_D = words_per_vector * codes_per_word

        if padded_D > D:
            pad_shape = list(codes_bhtd.shape)
            pad_shape[-1] = padded_D - D
            codes_bhtd = mx.concatenate(
                [codes_bhtd, mx.zeros(pad_shape, dtype=mx.uint32)],
                axis=-1,
            )

        grouped = codes_bhtd.reshape(B, H, T, words_per_vector, codes_per_word)
        shifts = mx.arange(codes_per_word, dtype=mx.uint32) * bits
        packed = mx.sum(
            grouped.astype(mx.uint32) << shifts.reshape(1, 1, 1, 1, -1),
            axis=-1,
        )
        return packed

    def _vector_aligned_unpack(
        self, packed: Any, bits: int, head_dim: int
    ) -> Any:
        """Unpack vector-aligned uint32 words back to individual codes.

        Parameters
        ----------
        packed
            uint32 array of shape (B, H, T, words_per_vector).
        bits
            Bit width per code.
        head_dim
            Original head dimension (codes beyond this are padding).

        Returns
        -------
        codes
            uint32 array of shape (B, H, T, head_dim).
        """
        B, H, T, words_per_vector = packed.shape
        codes_per_word = 32 // bits
        padded_D = words_per_vector * codes_per_word

        packed_view = packed.reshape(B, H, T, words_per_vector, 1).astype(mx.uint32)
        shifts = (mx.arange(codes_per_word, dtype=mx.uint32) * bits).reshape(1, 1, 1, 1, -1)
        mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
        codes = (packed_view >> shifts) & mask
        codes = codes.reshape(B, H, T, padded_D)
        if padded_D > head_dim:
            codes = codes[..., :head_dim]
        return codes

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_bhtd(
        self,
        x: Any,
        *,
        logical_start: int = 0,
        layer_id: int = 0,
        stream_id: str = "",
    ) -> PackedBlockV4:
        """Quantize and pack a BHTD tensor with vector-aligned codes.

        Parameters
        ----------
        x
            Array of shape (B, H, T, D).
        logical_start
            Global token offset of the first token in this block.
        layer_id
            Layer index for deterministic signs.
        stream_id
            "K" or "V" for stream-specific sign derivation.

        Returns
        -------
        PackedBlockV4
            Immutable sealed block with BHTG scales and BHTW packed codes.
        """
        if x.ndim != 4:
            raise ValueError("expected BHTD input")
        B, H, T, D = x.shape
        original_dtype_str = _mlx_dtype_name(x.dtype)
        original_value_count = int(B * H * T * D)

        # Pad feature axis to multiple of group_size
        pad = (self.group_size - (D % self.group_size)) % self.group_size
        if pad:
            pad_shape = [B, H, T, pad]
            x_padded = mx.concatenate(
                [x.astype(mx.float32), mx.zeros(pad_shape, dtype=mx.float32)],
                axis=-1,
            )
        else:
            x_padded = x.astype(mx.float32)

        padded_D = D + pad
        groups_per_vector = padded_D // self.group_size

        # Reshape to (B, H, T, groups_per_vector, group_size)
        grouped = x_padded.reshape(B, H, T, groups_per_vector, self.group_size)

        # Optional WHT + deterministic signs
        preconditioner = Preconditioner.NONE
        if self.use_wht:
            grouped = CartesianCodec.apply_wht(grouped)
            preconditioner = Preconditioner.WHT64_HASH_SIGN_V1
        if self.sign_seed != 0:
            grouped = CartesianCodec.apply_hash_signs(
                grouped, self.sign_seed, layer_id=layer_id, stream_id=stream_id
            )

        # Per-vector scales: max over group_size axis
        max_abs = mx.maximum(
            mx.max(mx.abs(grouped), axis=-1),
            mx.array(self.eps, dtype=mx.float32),
        )
        scale_bhtg = max_abs / float(self.qmax)

        # Quantize
        q_signed = mx.round(grouped / scale_bhtg[..., None])
        q_signed = mx.clip(q_signed, -self.qmax, self.qmax)
        codes_bhtd = (q_signed + self.qmax).astype(mx.uint32).reshape(B, H, T, padded_D)

        # Vector-aligned packing
        if self.bits <= 8:
            packed = self._vector_aligned_pack(codes_bhtd, self.bits)
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
            scales=scale_bhtg,
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

    def encode(self, x: Any) -> PackedBlock:
        """Quantize and pack a tensor.

        Parameters
        ----------
        x
            Array of any shape.  Internally flattened to (-1,).

        Returns
        -------
        PackedBlock
            Immutable sealed block with exact payload_bytes().
        """
        _ = tuple(x.shape)  # shape captured in num_elements below
        original_dtype_str = _mlx_dtype_name(x.dtype)
        flat = x.astype(mx.float32).reshape(-1)
        original_size = int(flat.size)

        # Pad to multiple of group_size
        pad = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad:
            flat = mx.concatenate([flat, mx.zeros((pad,), dtype=mx.float32)])
        _ = int(flat.size)  # padded size for debugging if needed

        # Grouped quantization
        grouped = flat.reshape(-1, self.group_size)

        # Optional WHT + deterministic signs
        if self.use_wht:
            grouped = CartesianCodec.apply_wht(grouped)
        if self.sign_seed != 0:
            grouped = CartesianCodec.apply_hash_signs(grouped, self.sign_seed)

        max_abs = mx.maximum(mx.max(mx.abs(grouped), axis=1), mx.array(self.eps, dtype=mx.float32))
        scale = max_abs / float(self.qmax)
        q_signed = mx.round(grouped / scale[:, None])
        q_signed = mx.clip(q_signed, -self.qmax, self.qmax)
        codes = (q_signed + self.qmax).astype(mx.uint32).reshape(-1)

        # Bit packing
        if self.bits <= 8:
            packed, n_values = BitPackedQuantizer.pack(codes, self.bits)
        else:
            packed = codes.astype(mx.uint32)
            n_values = int(codes.size)

        block = PackedBlock(
            packed_codes=packed,
            scales=scale,
            token_count=0,               # caller sets semantic token count
            bits=self.bits,
            group_size=self.group_size,
            n_values=n_values,
            format_version=3,          # BUMP
            num_elements=original_size,
            original_dtype=original_dtype_str,
            wht_applied=self.use_wht,
            sign_seed=self.sign_seed if self.sign_seed != 0 else 0,
            vector_alignment=64,       # NEW
        )
        block.validate()               # NEW: fail fast
        return block

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, block: PackedBlock) -> Any:
        """Reconstruct the original tensor from a PackedBlock.

        Trims group padding and restores the original dtype from V2/V3 metadata.
        For V4 blocks with BHTD shapes, delegates to decode_bhtd.
        """
        if block.bits != self.bits:
            raise ValueError(f"Block bits={block.bits}, codec bits={self.bits}")

        # V4 blocks with BHTG scales are handled by decode_bhtd
        if block.format_version == 4 or (
            block.scales is not None and block.scales.ndim >= 3
        ):
            return self.decode_bhtd(block)

        if block.format_version not in (1, 2, 3):
            raise ValueError(f"Unsupported PackedBlock version: {block.format_version}")

        # Unpack codes
        if block.bits <= 8:
            codes = BitPackedQuantizer.unpack(block.packed_codes, block.n_values, block.bits)
        else:
            codes = block.packed_codes[:block.n_values]

        flat = codes.astype(mx.float32).reshape(-1)
        if int(flat.size) != block.n_values:
            raise ValueError(f"Expected {block.n_values} codes, got {flat.size}")

        qmax = (1 << (block.bits - 1)) - 1
        grouped = flat.reshape(-1, block.group_size)
        q_signed = grouped - float(qmax)
        # Scales may be BHTG (B,H,T,G) or flat (n_groups,); flatten for broadcast
        scales_flat = block.scales.reshape(-1)
        restored = q_signed * scales_flat[:, None]

        # Inverse hash signs and WHT (both are self-inverse when normalized)
        if block.sign_seed != 0:
            restored = CartesianCodec.apply_hash_signs(restored, block.sign_seed)
        if block.wht_applied:
            restored = CartesianCodec.apply_wht(restored)

        # Flatten and trim group padding (V2 only; V1 blocks have num_elements==0)
        flat_restored = restored.reshape(-1)
        if block.num_elements > 0 and block.num_elements < int(flat_restored.size):
            flat_restored = flat_restored[:block.num_elements]

        # Restore original dtype only for V2+ blocks (V1 default is unreliable)
        if block.format_version >= 2 and block.original_dtype:
            target_dtype = _str_to_mlx_dtype(block.original_dtype)
            if target_dtype is not None:
                flat_restored = flat_restored.astype(target_dtype)

        return flat_restored

    def decode_bhtd(self, block: PackedBlock) -> Any:
        """Reconstruct a BHTD tensor from a vector-aligned PackedBlock.

        Expects packed_codes of shape (B,H,T,W) and scales of shape (B,H,T,G).
        Returns array of shape (B,H,T,D).
        """
        if block.format_version not in (3, 4):
            raise ValueError(f"Unsupported PackedBlock version: {block.format_version}")
        if block.bits != self.bits:
            raise ValueError(f"Block bits={block.bits}, codec bits={self.bits}")

        # Strict signature validation — reject blocks created with an
        # incompatible codec configuration (different signs, layout, etc.).
        if block.codec_signature and block.codec_signature != self.codec_signature:
            raise ValueError(
                f"Codec signature mismatch: block has {block.codec_signature}, "
                f"codec expects {self.codec_signature}"
            )

        # Unpack codes
        if block.bits <= 8:
            codes_bhtd = self._vector_aligned_unpack(
                block.packed_codes, block.bits, block.head_dim
            )
        else:
            codes_bhtd = block.packed_codes

        B, H, T, D = codes_bhtd.shape
        padded_D = D
        pad = (self.group_size - (padded_D % self.group_size)) % self.group_size
        if pad:
            codes_bhtd = mx.concatenate(
                [codes_bhtd, mx.zeros((B, H, T, pad), dtype=mx.uint32)],
                axis=-1,
            )
            padded_D = padded_D + pad

        groups_per_vector = padded_D // self.group_size
        grouped = codes_bhtd.reshape(B, H, T, groups_per_vector, self.group_size)
        qmax = (1 << (block.bits - 1)) - 1
        q_signed = grouped.astype(mx.float32) - float(qmax)

        # Scales are BHTG
        scales_bhtg = block.scales
        if scales_bhtg.ndim == 1:
            # Legacy flat scales — reshape to BHTG
            expected = B * H * T * groups_per_vector
            if int(scales_bhtg.size) != expected:
                raise ValueError(
                    f"Flat scales size {scales_bhtg.size} != expected {expected}"
                )
            scales_bhtg = scales_bhtg.reshape(B, H, T, groups_per_vector)

        restored = q_signed * scales_bhtg[..., None]

        # Inverse hash signs and WHT
        if block.sign_seed != 0:
            # V4 blocks require layer_id and stream_id for correct sign patterns
            layer_id = getattr(block, "layer_id", None)
            stream_id = getattr(block, "stream_id", None)
            if layer_id is None or stream_id is None:
                raise ValueError(
                    "Block missing required layer_id or stream_id attributes "
                    "(V4 format requires both for hash sign correctness)"
                )
            restored = CartesianCodec.apply_hash_signs(
                restored,
                block.sign_seed,
                layer_id=layer_id,
                stream_id=stream_id,
            )
        if block.wht_applied:
            restored = CartesianCodec.apply_wht(restored)

        # Flatten feature axis and trim group padding
        flat_restored = restored.reshape(B, H, T, padded_D)
        if block.num_elements > 0:
            expected_elements = B * H * T * block.head_dim
            if block.num_elements == expected_elements:
                flat_restored = flat_restored[..., : block.head_dim]
            elif block.num_elements < int(flat_restored.size):
                # Flat fallback for older blocks
                flat_restored = flat_restored.reshape(-1)[: block.num_elements]
                flat_restored = flat_restored.reshape(B, H, T, block.head_dim)
            else:
                raise ValueError(
                    f"Unexpected num_elements ({block.num_elements}) "
                    f"vs expected ({expected_elements}) and tensor size "
                    f"({int(flat_restored.size)})"
                )

        # Restore original dtype
        if block.original_dtype:
            target_dtype = _str_to_mlx_dtype(block.original_dtype)
            if target_dtype is not None:
                flat_restored = flat_restored.astype(target_dtype)

        return flat_restored

    # ------------------------------------------------------------------
    # Analytical size (no materialisation)
    # ------------------------------------------------------------------

    def estimate_bytes(self, block: PackedBlock) -> int:
        """Exact bytes from actual stored arrays."""
        return block.payload_bytes()

    def estimate_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Analytical byte estimate without materialising arrays."""
        n = math.prod(shape)
        pad = (self.group_size - (n % self.group_size)) % self.group_size
        padded = n + pad
        if self.bits <= 8:
            cpw = 32 // self.bits
            words = (padded + cpw - 1) // cpw
        else:
            words = padded
        code_bytes = words * 4
        groups = padded // self.group_size
        scale_bytes = groups * 4
        return code_bytes + scale_bytes

    # ------------------------------------------------------------------
    # WHT helpers (stateless)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_wht(x: Any) -> Any:
        """Apply Walsh-Hadamard Transform (WHT-64).

        Uses the pure-MLX reference implementation.  The Metal kernel path is
        currently disabled pending correctness validation (see WHT identity tests).
        """
        return _reference_wht64(x)

    @staticmethod
    def apply_hash_signs(
        x: Any, seed: int = 42, layer_id: int = 0, stream_id: str = ""
    ) -> Any:
        """Apply deterministic hash-based sign randomisation.

        Uses the pure-MLX reference implementation.  The Metal kernel path is
        currently disabled pending correctness validation.
        """
        return _reference_hash_signs(x, seed, layer_id=layer_id, stream_id=stream_id)


def _reference_wht64(x: Any) -> Any:
    """Pure-MLX reference WHT-64 (orthonormal, self-inverse).

    Iterative vectorised butterfly.  Preserves input shape.
    Normalised by sqrt(n) so that ``WHT(WHT(x)) == x``.
    """
    h = x.astype(mx.float32)
    n = int(h.shape[-1])
    if n < 2:
        return h
    original_shape = h.shape
    step = 1
    while step < n:
        # Reshape so we can vectorise the butterfly on pairs separated by `step`
        h_reshaped = h.reshape(*h.shape[:-1], -1, 2 * step)
        a = h_reshaped[..., :step]
        b = h_reshaped[..., step:]
        h = mx.concatenate([a + b, a - b], axis=-1)
        # Flatten back to the original rank so the next iteration works
        h = h.reshape(*original_shape)
        step *= 2
    return h / math.sqrt(n)


def _reference_hash_signs(
    x: Any, seed: int, layer_id: int = 0, stream_id: str = ""
) -> Any:
    """Pure-MLX reference hash signs (SplitMix64-v1).

    Deterministic: same (shape, seed, layer_id, stream_id) always produces
    the same signs.  Uses an integer hash so that NumPy and MLX produce
    identical signs.
    """
    flat = x.reshape(-1)
    n = int(flat.size)

    # Mix layer_id and stream_id into the seed so that different layers
    # and K/V streams get independent sign patterns.
    stream_hash = 0
    for ch in stream_id:
        stream_hash = (stream_hash * 31 + ord(ch)) & 0xFFFFFFFF
    mixed = np.uint32(seed)
    mixed = np.uint32(mixed ^ np.uint32((layer_id * 0x9E3779B9) & 0xFFFFFFFF))
    mixed = np.uint32(mixed ^ np.uint32(stream_hash & 0xFFFFFFFF))
    # Mask to signed-32-bit range so that the same seed can be passed as an
    # MLX inline-kernel template argument (which rejects unsigned > INT_MAX).
    mixed = np.uint32(int(mixed) & 0x7FFFFFFF)
    seed_val = mx.array(mixed)

    # Build signs using the same integer hash as the NumPy backend
    indices = mx.arange(n, dtype=mx.uint32)
    state = mx.bitwise_xor(indices, seed_val)
    state = (state + mx.array(np.uint32(0x9E3779B9))) & mx.array(np.uint32(0xFFFFFFFF))
    state = mx.bitwise_xor(state, state >> 16)
    state = (state * mx.array(np.uint32(0x85EBCA6B))) & mx.array(np.uint32(0xFFFFFFFF))
    state = mx.bitwise_xor(state, state >> 13)
    state = (state * mx.array(np.uint32(0xC2B2AE35))) & mx.array(np.uint32(0xFFFFFFFF))
    state = mx.bitwise_xor(state, state >> 16)
    signs = mx.where((state & 1) == 1, mx.array(-1.0, dtype=x.dtype), mx.array(1.0, dtype=x.dtype))
    return (flat * signs).reshape(x.shape)


# ------------------------------------------------------------------
# Dtype helpers
# ------------------------------------------------------------------

_DTYPE_NAME_MAP: dict[Any, str] = {}
_STR_DTYPE_MAP: dict[str, Any] = {}


def _build_dtype_maps() -> None:
    """Build bidirectional dtype name maps (called once at import)."""
    global _DTYPE_NAME_MAP, _STR_DTYPE_MAP
    pairs = [
        (getattr(mx, "float16", None), "float16"),
        (getattr(mx, "float32", None), "float32"),
        (getattr(mx, "bfloat16", None), "bfloat16"),
        (getattr(mx, "int8", None), "int8"),
        (getattr(mx, "int16", None), "int16"),
        (getattr(mx, "int32", None), "int32"),
        (getattr(mx, "uint8", None), "uint8"),
        (getattr(mx, "uint32", None), "uint32"),
        (getattr(mx, "bool_", None), "bool"),
    ]
    _DTYPE_NAME_MAP = {dt: name for dt, name in pairs if dt is not None}
    _STR_DTYPE_MAP = {name: dt for dt, name in pairs if dt is not None}


def _mlx_dtype_name(dtype: Any) -> str:
    """Return a stable string name for an MLX dtype object."""
    if not _DTYPE_NAME_MAP:
        _build_dtype_maps()
    return _DTYPE_NAME_MAP.get(dtype, "float32")


def _str_to_mlx_dtype(name: str) -> Any | None:
    """Return the MLX dtype object for a string name, or None."""
    if not _STR_DTYPE_MAP:
        _build_dtype_maps()
    return _STR_DTYPE_MAP.get(name)
