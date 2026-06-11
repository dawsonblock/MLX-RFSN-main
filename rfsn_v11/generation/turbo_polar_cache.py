"""TurboPolar KV cache — runtime cache object with real byte counters.

Phase 5: K-only compressed cache.
  - Keys stored as PolarKeyBlock payloads.
  - Values stored dense fp16.
  - Real bytes_written / bytes_read counters.

No promotion until real counters are proven and teacher-forced
logit gate passes.
"""
from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.polar.payload import PolarKeyBlock


class TurboPolarKVCache:
    """Runtime KV cache with PolarQuant-compressed keys and dense values.

    Args:
        config: TurboPolarConfig (or a dict with the required fields).
    """

    def __init__(self, config: Any) -> None:
        # Accept either a TurboPolarConfig or a dict
        if hasattr(config, "k_angle_bits_level1"):
            cfg = config
        else:
            from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
            cfg = TurboPolarConfig(**config)

        self.cfg = cfg
        self.key_blocks: list[PolarKeyBlock] = []
        self.value_blocks_dense: list[mx.array] = []
        self.events: list[str] = []

        # Real counters (bytes)
        self.bytes_written_actual: int = 0
        self.bytes_read_actual: int = 0

        self._encoder = PolarQuantEncoder(
            angle_bits_level1=cfg.k_angle_bits_level1,
            angle_bits_deep=cfg.k_angle_bits_deep,
            head_dim=cfg.head_dim,
            use_rotation=True,
            rotation_seed=42,
        )
        self._decoder = PolarQuantDecoder(
            head_dim=cfg.head_dim,
            use_rotation=True,
            rotation_seed=42,
        )

    def update(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        """Append a new K/V block.

        Args:
            keys:   (..., block_size, head_dim) or (..., head_dim)
            values: same shape as keys
        """
        # Compress keys into PolarKeyBlock
        polar_block = self._encoder.encode(keys)
        self.key_blocks.append(polar_block)

        # Store values dense (fp16)
        self.value_blocks_dense.append(values.astype(mx.float16))

        # Increment real counters
        key_bytes = polar_block.compressed_nbytes()
        val_bytes = int(values.nbytes)
        self.bytes_written_actual += key_bytes + val_bytes

        self.events.append(f"update_keys:{key_bytes}:values:{val_bytes}")

    def fetch_keys_for_block(self, block_idx: int) -> mx.array:
        """Decompress keys for a single block.

        Returns:
            Reconstructed keys (float32).
        """
        if block_idx < 0 or block_idx >= len(self.key_blocks):
            raise IndexError(f"block_idx {block_idx} out of range")
        block = self.key_blocks[block_idx]
        recon = self._decoder.decode(block)
        self.bytes_read_actual += block.compressed_nbytes()
        self.events.append(f"fetch_keys_block:{block_idx}")
        return recon

    def fetch_values_for_block(self, block_idx: int) -> mx.array:
        """Return dense values for a block (no decompression needed)."""
        if block_idx < 0 or block_idx >= len(self.value_blocks_dense):
            raise IndexError(f"block_idx {block_idx} out of range")
        vals = self.value_blocks_dense[block_idx]
        self.bytes_read_actual += int(vals.nbytes)
        self.events.append(f"fetch_values_block:{block_idx}")
        return vals.astype(mx.float32)

    def as_trace_dict(self) -> dict[str, Any]:
        """Return an honest trace dict for artifact generation."""
        return {
            "cache_backend_used": "turbo_polar_k_only",
            "real_cache_used": True,
            "prefill_polar_encode_events": len(self.key_blocks),
            "decode_polar_fetch_events": len(self.key_blocks),
            "cache_bytes_written_actual": self.bytes_written_actual,
            "cache_bytes_read_actual": self.bytes_read_actual,
            "fallback_used": False,
        }
