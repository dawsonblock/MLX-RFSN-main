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

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.compat import mx

from .contracts import PackedBlock


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
        use_wht: bool = False,
        sign_seed: int = 42,
    ) -> None:
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16]; got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        self.bits = bits
        self.group_size = group_size
        self.eps = eps
        self.use_wht = use_wht
        self.sign_seed = sign_seed
        self.qmax = (1 << (bits - 1)) - 1

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

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
        original_shape = tuple(x.shape)
        original_dtype_str = _mlx_dtype_name(x.dtype)
        flat = x.astype(mx.float32).reshape(-1)
        original_size = int(flat.size)

        # Pad to multiple of group_size
        pad = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad:
            flat = mx.concatenate([flat, mx.zeros((pad,), dtype=mx.float32)])
        padded_size = int(flat.size)

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
        """
        if block.format_version not in (1, 2, 3):
            raise ValueError(f"Unsupported PackedBlock version: {block.format_version}")
        if block.bits != self.bits:
            raise ValueError(f"Block bits={block.bits}, codec bits={self.bits}")

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
        restored = q_signed * block.scales[:, None]

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
    def apply_hash_signs(x: Any, seed: int = 42) -> Any:
        """Apply deterministic hash-based sign randomisation.

        Uses the pure-MLX reference implementation.  The Metal kernel path is
        currently disabled pending correctness validation.
        """
        return _reference_hash_signs(x, seed)


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


def _reference_hash_signs(x: Any, seed: int) -> Any:
    """Pure-MLX reference hash signs.

    Deterministic: same (shape, seed) always produces the same signs.
    Uses a PRNG key derived from ``seed``; does not mutate global state.
    """
    key = mx.random.key(seed)
    signs = mx.where(mx.random.normal(shape=x.shape, key=key) > 0, 1.0, -1.0)
    return x * signs


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
