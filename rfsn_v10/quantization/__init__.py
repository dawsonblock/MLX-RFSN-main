"""RFSN v10 experimental quantization package.

Exports for experimental quantizers and packed data structures.
MLX-dependent classes are loaded lazily via __getattr__ so that
pure-Python modules (e.g. layer_policy) can be imported on systems
without mlx installed.
"""
from __future__ import annotations

from .layer_policy import (  # noqa: I001
    KNOWN_MODES,
    LayerPolicy,
    load_policy,
    validate_layer_policy,
)

__all__ = [
    "LayerPolicy",
    "load_policy",
    "validate_layer_policy",
    "KNOWN_MODES",
    "PackedCodeBuffer",
    "PackedPolarCodes",
    "PolarPacked",
    "PolarQuantizer",
    "UniformQuantMeta",
    "GroupQuantMeta",
    "PackedCartesianCodes",
    "CartesianPacked",
    "GroupedCartesianQuantizer",
    "HybridPacked",
    "HybridPolarCartesianQuantizer",
    "TurboPolarPacked",
    "TurboPolarQuantizer",
    "QJLSketch",
    "QJLScoreCorrector",
    "QuantizedKVPacket",
    "QuantizedKVManager",
    "TurboPolarKVManager",
    "iterative_hierarchical_polar_forward",
    "iterative_hierarchical_polar_inverse",
    "quantize_uniform_fixed_range",
    "dequantize_uniform_fixed_range",
    "quantize_group_unsigned",
    "dequantize_group_unsigned",
    "unpack_code_blocks",
    "unpack_blocks",
    "dequantize_full",
    "dequantize_k_blocks",
    "dequantize_v_blocks",
]

_LAZY_IMPORT_MAP = {
    "PackedCodeBuffer": ("polar_quant", "PackedCodeBuffer"),
    "PackedPolarCodes": ("polar_quant", "PackedPolarCodes"),
    "PolarPacked": ("polar_quant", "PolarPacked"),
    "PolarQuantizer": ("polar_quant", "PolarQuantizer"),
    "UniformQuantMeta": ("polar_quant", "UniformQuantMeta"),
    "GroupQuantMeta": ("polar_quant", "GroupQuantMeta"),
    "iterative_hierarchical_polar_forward": (
        "polar_quant",
        "iterative_hierarchical_polar_forward",
    ),
    "iterative_hierarchical_polar_inverse": (
        "polar_quant",
        "iterative_hierarchical_polar_inverse",
    ),
    "quantize_uniform_fixed_range": (
        "polar_quant",
        "quantize_uniform_fixed_range",
    ),
    "dequantize_uniform_fixed_range": (
        "polar_quant",
        "dequantize_uniform_fixed_range",
    ),
    "quantize_group_unsigned": ("polar_quant", "quantize_group_unsigned"),
    "dequantize_group_unsigned": ("polar_quant", "dequantize_group_unsigned"),
    "PackedCartesianCodes": ("grouped_cartesian", "PackedCartesianCodes"),
    "CartesianPacked": ("grouped_cartesian", "CartesianPacked"),
    "GroupedCartesianQuantizer": (
        "grouped_cartesian",
        "GroupedCartesianQuantizer",
    ),
    "HybridPacked": ("hybrid_polar_cartesian", "HybridPacked"),
    "HybridPolarCartesianQuantizer": (
        "hybrid_polar_cartesian",
        "HybridPolarCartesianQuantizer",
    ),
    "TurboPolarPacked": ("turbo_polar_quant", "TurboPolarPacked"),
    "TurboPolarQuantizer": ("turbo_polar_quant", "TurboPolarQuantizer"),
    "QJLSketch": ("qjl_score_correction", "QJLSketch"),
    "QJLScoreCorrector": ("qjl_score_correction", "QJLScoreCorrector"),
    "QuantizedKVPacket": ("kv_quant_manager", "QuantizedKVPacket"),
    "QuantizedKVManager": ("kv_quant_manager", "QuantizedKVManager"),
    "TurboPolarKVManager": ("turbo_polar_kv_manager", "TurboPolarKVManager"),
    "unpack_code_blocks": ("block_unpack", "unpack_code_blocks"),
    "unpack_blocks": ("block_unpack", "unpack_blocks"),
    "dequantize_full": ("block_unpack", "dequantize_full"),
    "dequantize_k_blocks": ("block_unpack", "dequantize_k_blocks"),
    "dequantize_v_blocks": ("block_unpack", "dequantize_v_blocks"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        module_name, attr_name = _LAZY_IMPORT_MAP[name]
        module = __import__(
            f"rfsn_v10.quantization.{module_name}",
            fromlist=[attr_name],
        )
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
