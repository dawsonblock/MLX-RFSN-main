"""CUDA / Triton kernel backend — not yet implemented."""

from __future__ import annotations

from ._common import KernelRouteError


class CudaBackend:
    """CUDA/Triton kernel backend (stub)."""

    name = "cuda"

    @classmethod
    def available(cls) -> bool:
        try:
            import triton  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _raise():
        raise KernelRouteError(
            "CUDA backend is not yet implemented. "
            "Install triton>=3.0 and torch to enable."
        )

    @staticmethod
    def pack_bits(codes, bits: int) -> tuple:
        CudaBackend._raise()
        return None, 0  # type: ignore[return-value]

    @staticmethod
    def unpack_bits(packed, n_values: int, bits: int):
        CudaBackend._raise()
        return None  # type: ignore[return-value]

    @staticmethod
    def scaled_dot_product_attention(
        queries, keys, values, scale: float | None = None, causal: bool = False
    ):
        CudaBackend._raise()
        return None  # type: ignore[return-value]

    @staticmethod
    def packed_dequant(
        packed, scales, n_values: int, bits: int, group_size: int = 64
    ):
        CudaBackend._raise()
        return None  # type: ignore[return-value]

    @staticmethod
    def wht_transform(x):
        CudaBackend._raise()
        return None  # type: ignore[return-value]

    @staticmethod
    def apply_hash_signs(x, seed: int):
        CudaBackend._raise()
        return None  # type: ignore[return-value]

    @staticmethod
    def quantized_attention_decode(
        queries,
        packed_k,
        packed_v,
        scales_k,
        scales_v,
        n_keys: int,
        bits: int,
        group_size: int = 64,
        scale: float | None = None,
    ):
        CudaBackend._raise()
        return None  # type: ignore[return-value]
