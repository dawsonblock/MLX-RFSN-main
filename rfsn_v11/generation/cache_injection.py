"""RFSN v11 cache injector for benchmark-only minimal decode loop.

Provides a wrapper around the RFSN v11 KVCompressor so it can be used
as a prompt_cache in the minimal decode loop without monkey-patching
MLX-LM globally.
"""
from __future__ import annotations

from typing import Any


class RFSNV11CacheInjector:
    """Wrapper to make RFSN v11 compression compatible with
    minimal_decode_loop.

    This is NOT a global monkey-patch. It only affects the cache object
    passed into minimal_decode_loop.
    """

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 5,
        group_size: int = 64,
        use_wht: bool = True,
        dim: int = 128,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.use_wht = use_wht
        self.dim = dim
        self._compressor: Any | None = None

    def _get_compressor(self) -> Any:
        if self._compressor is None:
            from rfsn_v11.quant.kv_compressor import KVCompressor
            self._compressor = KVCompressor(
                k_bits=self.key_bits,
                v_bits=self.value_bits,
                group_size=self.group_size,
                dim=self.dim,
                use_wht=self.use_wht,
                skip_quality_gate=True,
            )
        return self._compressor

    def compress_kv(self, keys: Any, values: Any) -> Any:
        """Compress key and value arrays using RFSN v11."""
        compressor = self._get_compressor()
        return compressor.compress(keys, values)

    def decompress_kv(self, compressed: Any) -> tuple[Any, Any]:
        """Decompress to key and value arrays."""
        compressor = self._get_compressor()
        return compressor.decompress(compressed)

    @property
    def nbytes(self) -> int:
        """Return estimated bytes of the last compressed KV state.

        Raises NotImplementedError if memory accounting is not yet
        available.  Do not return fake 0 — that breaks promotion
        metrics that rely on honest memory reporting.
        """
        raise NotImplementedError(
            "RFSN v11 cache injector memory accounting is not implemented. "
            "This candidate remains OFFLINE_ONLY until real cache injection "
            "and memory proof exist."
        )
