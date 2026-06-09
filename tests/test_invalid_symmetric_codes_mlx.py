#!/usr/bin/env python3
"""Main12 invalid symmetric code parity tests for Metal and reference routes."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def _corrupt_packed_3bit_with_code7(n_values: int = 64) -> tuple[mx.array, mx.array, tuple[int, ...]]:
    shape = (1, 1, 1, n_values)
    codes = mx.zeros((n_values,), dtype=mx.uint32)
    codes = codes + mx.array(7, dtype=mx.uint32)
    packed, _ = BitPackedQuantizer.pack(codes, bits=3)
    scales = mx.ones((1,), dtype=mx.float32)
    return packed, scales, shape


def test_reference_path_rejects_invalid_symmetric_code(tmp_path) -> None:
    packed, scales, shape = _corrupt_packed_3bit_with_code7()

    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=3,
        v_bits=3,
        use_wht=False,
        use_incoherent_signs=False,
        prefer_metal_kernels=False,
    )

    with pytest.raises(ValueError, match="Invalid symmetric quant code"):
        manager._reconstruct_cached_tensor(
            packed=packed,
            scales=scales,
            n_values=shape[-1],
            shape=shape,
            bits=3,
            seed=0,
            use_wht=False,
            use_incoherent_signs=False,
            out_dtype=mx.float32,
        )


def test_strict_metal_path_rejects_invalid_symmetric_code(tmp_path) -> None:
    packed, scales, shape = _corrupt_packed_3bit_with_code7()

    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=3,
        v_bits=3,
        use_wht=False,
        use_incoherent_signs=False,
        prefer_metal_kernels=True,
        strict_metal=True,
    )

    with pytest.raises(ValueError, match="Invalid symmetric quant code"):
        manager._reconstruct_cached_tensor(
            packed=packed,
            scales=scales,
            n_values=shape[-1],
            shape=shape,
            bits=3,
            seed=0,
            use_wht=False,
            use_incoherent_signs=False,
            out_dtype=mx.float32,
        )


def test_non_strict_metal_behavior_documented(tmp_path) -> None:
    packed, scales, shape = _corrupt_packed_3bit_with_code7()

    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=3,
        v_bits=3,
        use_wht=False,
        use_incoherent_signs=False,
        prefer_metal_kernels=True,
        strict_metal=False,
        validate_metal_codes=False,
    )

    out = manager._reconstruct_cached_tensor(
        packed=packed,
        scales=scales,
        n_values=shape[-1],
        shape=shape,
        bits=3,
        seed=0,
        use_wht=False,
        use_incoherent_signs=False,
        out_dtype=mx.float32,
    )
    mx.eval(out)

    assert manager.last_reconstruction_kernel in {
        "metal_multikernel_dequant",
        "metal_failed_fallback_reference",
    }
