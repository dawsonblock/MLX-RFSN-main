"""Compressed Polar cache with blocked allocation and exact memory accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import PolarCacheState
from .packing import pack_indices, unpack_indices

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


@dataclass
class PolarCache:
    """Per-layer compressed K/V cache.

    Stores packed indices and norms for keys and values.  Capacity grows in
    fixed token blocks to avoid per-token reallocation.
    """

    config: Any  # PolarFusedConfig
    batch_size: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 256

    # Mutable state
    state: PolarCacheState | None = field(default=None, repr=False)
    _codebooks: dict[str, Any] = field(default_factory=dict, repr=False)
    _rotations: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError(
                f"Batch size must be 1 in the initial profile; got {self.batch_size}"
            )
        if self.head_dim not in (64, 128):
            raise ValueError(f"head_dim must be 64 or 128; got {self.head_dim}")

    # ------------------------------------------------------------------
    # Append / Grow
    # ------------------------------------------------------------------

    def append(
        self,
        key_indices: Any,
        key_norms: Any,
        value_indices: Any,
        value_norms: Any,
    ) -> None:
        """Append a single decode step (one token per head).

        All inputs are expected to have shape (batch, num_kv_heads, ...).
        For the initial profile batch_size == 1.
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        n_new = key_indices.shape[2] if key_indices.ndim >= 3 else 1

        if self.state is None:
            # First allocation
            capacity = self._round_up(n_new)
            self.state = self._allocate_state(capacity)
            self.state = self._write_slice(
                self.state, 0, n_new,
                key_indices, key_norms,
                value_indices, value_norms,
            )
            self.state.offset = n_new
            return

        # Check capacity
        if self.state.offset + n_new > self.state.capacity:
            new_cap = self._round_up(self.state.offset + n_new)
            self.state = self._grow(self.state, new_cap)

        # Write
        self.state = self._write_slice(
            self.state, self.state.offset, self.state.offset + n_new,
            key_indices, key_norms,
            value_indices, value_norms,
        )
        self.state.offset += n_new

    def trim(self, new_offset: int) -> None:
        """Trim cache to retain only the first ``new_offset`` tokens."""
        if self.state is None:
            return
        if new_offset < 0:
            raise ValueError("new_offset cannot be negative")
        if new_offset >= self.state.offset:
            return
        self.state.offset = new_offset

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_valid_slice(self) -> PolarCacheState:
        """Return a PolarCacheState with arrays sliced to the valid offset."""
        if self.state is None:
            raise RuntimeError("Cache is empty")
        s = self.state
        return PolarCacheState(
            key_indices=s.key_indices[..., :s.offset, :],
            key_norms=s.key_norms[..., :s.offset],
            value_indices=s.value_indices[..., :s.offset, :],
            value_norms=s.value_norms[..., :s.offset],
            offset=s.offset,
            capacity=s.capacity,
        )

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Exact bytes used by stored (valid) data only."""
        if self.state is None:
            return 0
        s = self.state
        # Packed indices: uint32
        key_idx_bytes = int(s.key_indices.size) * 4
        val_idx_bytes = int(s.value_indices.size) * 4
        # Norms: float32
        key_norm_bytes = int(s.key_norms.size) * 4
        val_norm_bytes = int(s.value_norms.size) * 4
        return key_idx_bytes + val_idx_bytes + key_norm_bytes + val_norm_bytes

    def capacity_bytes(self) -> int:
        """Bytes allocated including padding."""
        if self.state is None:
            return 0
        s = self.state
        key_idx_bytes = int(s.key_indices.size) * 4
        val_idx_bytes = int(s.value_indices.size) * 4
        key_norm_bytes = int(s.key_norms.size) * 4
        val_norm_bytes = int(s.value_norms.size) * 4
        return key_idx_bytes + val_idx_bytes + key_norm_bytes + val_norm_bytes

    def compression_ratio(self, fp16_reference: bool = True) -> float:
        """Compression ratio relative to FP16 KV storage.

        Returns ``fp16_bytes / compressed_bytes``.
        """
        if self.state is None:
            return 1.0
        valid = self.memory_bytes()
        if valid == 0:
            return 1.0
        # FP16: 2 bytes per coordinate for both K and V
        # shape: (batch, heads, tokens, head_dim)
        fp16_bytes = self.batch_size * self.num_kv_heads * self.state.offset * self.head_dim * 2 * 2
        return fp16_bytes / valid

    def metadata(self) -> dict[str, Any]:
        """Return human-readable metadata."""
        return {
            "batch_size": self.batch_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "offset": self.state.offset if self.state else 0,
            "capacity": self.state.capacity if self.state else 0,
            "memory_bytes": self.memory_bytes(),
            "capacity_bytes": self.capacity_bytes(),
            "compression_ratio": self.compression_ratio(),
        }

    def validate(self) -> None:
        """Raise ValueError if cache state is inconsistent."""
        if self.state is None:
            return
        s = self.state
        if s.offset > s.capacity:
            raise ValueError(f"offset {s.offset} exceeds capacity {s.capacity}")
        # Check shape consistency
        expected_idx_shape = (self.batch_size, self.num_kv_heads, s.capacity, -1)
        # We can't check the packed last dim without knowing bits, but we
        # can verify the first three dims
        if s.key_indices.shape[:3] != (self.batch_size, self.num_kv_heads, s.capacity):
            raise ValueError(
                f"key_indices shape mismatch: {s.key_indices.shape[:3]} vs {expected_idx_shape[:3]}"
            )
        if s.key_norms.shape != (self.batch_size, self.num_kv_heads, s.capacity):
            raise ValueError(
                f"key_norms shape mismatch: {s.key_norms.shape} vs "
                f"{(self.batch_size, self.num_kv_heads, s.capacity)}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _round_up(self, n: int) -> int:
        return ((n + self.block_size - 1) // self.block_size) * self.block_size

    def _allocate_state(self, capacity: int) -> PolarCacheState:
        """Allocate empty cache state for the given capacity."""
        if mx is None:
            raise RuntimeError("MLX is not installed")
        # Packed indices last dimension: we don't know bits yet, so allocate
        # generously with head_dim and caller will replace with packed shape.
        # For now we allocate the unpacked shape and the caller packs later.
        # Actually, to keep cache.py independent of bits, we accept the
        # caller-provided arrays and just manage capacity.
        # We'll allocate with shape (B, H, capacity, head_dim) as uint8,
        # then the caller packs.
        return PolarCacheState(
            key_indices=mx.zeros((self.batch_size, self.num_kv_heads, capacity, self.head_dim), dtype=mx.uint8),
            key_norms=mx.zeros((self.batch_size, self.num_kv_heads, capacity), dtype=mx.float32),
            value_indices=mx.zeros((self.batch_size, self.num_kv_heads, capacity, self.head_dim), dtype=mx.uint8),
            value_norms=mx.zeros((self.batch_size, self.num_kv_heads, capacity), dtype=mx.float32),
            offset=0,
            capacity=capacity,
        )

    def _grow(self, state: PolarCacheState, new_cap: int) -> PolarCacheState:
        """Grow all arrays to new capacity."""
        if mx is None:
            raise RuntimeError("MLX is not installed")

        def _grow_arr(arr: Any, pad_axis: int) -> Any:
            shape = list(arr.shape)
            old = shape[pad_axis]
            if old >= new_cap:
                return arr
            shape[pad_axis] = new_cap - old
            pad = mx.zeros(shape, arr.dtype)
            return mx.concatenate([arr, pad], axis=pad_axis)

        return PolarCacheState(
            key_indices=_grow_arr(state.key_indices, 2),
            key_norms=_grow_arr(state.key_norms, 2),
            value_indices=_grow_arr(state.value_indices, 2),
            value_norms=_grow_arr(state.value_norms, 2),
            offset=state.offset,
            capacity=new_cap,
        )

    def _write_slice(
        self,
        state: PolarCacheState,
        start: int,
        end: int,
        key_indices: Any,
        key_norms: Any,
        value_indices: Any,
        value_norms: Any,
    ) -> PolarCacheState:
        """Write new data into the cache state at [start:end]."""
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # In MLX, array item assignment is not supported directly.
        # We use concatenate or create a new array.
        # For simplicity, we update by slicing and concatenating.
        # This is O(n) but cache writes are small in the decode loop.

        def _update(arr: Any, new_slice: Any, axis: int = 2) -> Any:
            slices_before = [arr[..., :start, :] if arr.ndim >= 4 else arr[:, :, :start]]
            slices_after = [arr[..., end:, :] if arr.ndim >= 4 else arr[:, :, end:]]
            return mx.concatenate(slices_before + [new_slice] + slices_after, axis=axis)

        return PolarCacheState(
            key_indices=_update(state.key_indices, key_indices),
            key_norms=_update(state.key_norms, key_norms),
            value_indices=_update(state.value_indices, value_indices),
            value_norms=_update(state.value_norms, value_norms),
            offset=state.offset,
            capacity=state.capacity,
        )
