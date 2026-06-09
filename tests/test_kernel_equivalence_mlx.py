#!/usr/bin/env python3
"""Main12 metal reconstruction route equivalence tests."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kernels import apply_hash_signs_metal
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager

REQUIRED_SHAPES = [
    (1, 4, 128, 64),
    (1, 8, 512, 64),
    (1, 8, 1024, 128),
]


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def test_metal_hash_sign_is_self_inverse() -> None:
    mx.random.seed(42)
    x = mx.random.normal((1, 8, 128, 64))
    seed = 12345
    y = apply_hash_signs_metal(x, seed)
    z = apply_hash_signs_metal(y, seed)
    mx.eval(z)
    diff = mx.max(mx.abs(x - z)).item()
    assert diff < 1e-5


@pytest.mark.parametrize(
    ("use_wht", "use_signs", "expected_label"),
    [
        (False, False, "metal_multikernel_dequant"),
        (True, False, "metal_multikernel_dequant_wht"),
        (False, True, "metal_multikernel_dequant_sign"),
        (True, True, "metal_multikernel_dequant_wht_sign"),
    ],
)
@pytest.mark.parametrize("shape", REQUIRED_SHAPES)
def test_metal_reconstruction_matches_reference(
    use_wht,
    use_signs,
    expected_label,
    shape,
    tmp_path,
):
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=use_wht,
        use_incoherent_signs=use_signs,
        prefer_metal_kernels=True,
        prefer_fused_kernel=False,
        strict_metal=False,
    )

    mx.random.seed(7)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)
    manager.store("kernel_eq", keys, values, token_count=shape[2])

    metal_k, metal_v = manager.retrieve("kernel_eq", out_dtype=mx.float32)
    assert manager.last_reconstruction_kernel == expected_label

    cache = manager.active_caches["kernel_eq"]
    if cache.num_blocks > 0:
        ref_k = manager._reconstruct_all_blocks(
            cache=cache, is_key=True, out_dtype=mx.float32,
        )
        ref_v = manager._reconstruct_all_blocks(
            cache=cache, is_key=False, out_dtype=mx.float32,
        )
    else:
        ref_k = manager._reconstruct_packed_dequant_wht(
            packed=cache.k_packed,
            scales=cache.k_scales,
            n_values=cache.k_n_values,
            shape=cache.shape,
            bits=cache.k_bits,
            seed=cache.seed,
            use_wht=cache.use_wht,
            use_incoherent_signs=cache.use_incoherent_signs,
            out_dtype=mx.float32,
        )
        ref_v = manager._reconstruct_packed_dequant_wht(
            packed=cache.v_packed,
            scales=cache.v_scales,
            n_values=cache.v_n_values,
            shape=cache.shape,
            bits=cache.v_bits,
            seed=cache.seed,
            use_wht=cache.use_wht,
            use_incoherent_signs=cache.use_incoherent_signs,
            out_dtype=mx.float32,
        )

    mx.eval(metal_k, metal_v, ref_k, ref_v)
    assert cosine_similarity(metal_k, ref_k) > 0.999
    assert cosine_similarity(metal_v, ref_v) > 0.999
    assert mx.max(mx.abs(metal_k - ref_k)).item() < 1e-3
    assert mx.max(mx.abs(metal_v - ref_v)).item() < 1e-3
    assert metal_k.shape == ref_k.shape
    assert metal_v.shape == ref_v.shape
    assert metal_k.dtype == mx.float32
    assert metal_v.dtype == mx.float32


@pytest.mark.parametrize(
    ("use_wht", "use_signs", "expected_label"),
    [
        (False, False, "metal_multikernel_dequant"),
        (True, False, "metal_multikernel_dequant_wht"),
        (False, True, "metal_multikernel_dequant_sign"),
        (True, True, "metal_multikernel_dequant_wht_sign"),
    ],
)
@pytest.mark.parametrize("shape", REQUIRED_SHAPES)
def test_strict_metal_uses_metal_route(
    use_wht,
    use_signs,
    expected_label,
    shape,
    tmp_path,
):
    mx.random.seed(1)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)

    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path / "strict"),
        k_bits=8,
        v_bits=3,
        use_wht=use_wht,
        use_incoherent_signs=use_signs,
        prefer_metal_kernels=True,
        prefer_fused_kernel=False,
        strict_metal=True,
    )

    key = f"strict_metal_{shape}_{use_wht}_{use_signs}"
    manager.store(key, keys, values, token_count=shape[2])
    k_rec, v_rec = manager.retrieve(key, out_dtype=mx.float32)

    cache = manager.active_caches[key]
    if cache.num_blocks > 0:
        ref_k = manager._reconstruct_all_blocks(
            cache=cache, is_key=True, out_dtype=mx.float32,
        )
        ref_v = manager._reconstruct_all_blocks(
            cache=cache, is_key=False, out_dtype=mx.float32,
        )
    else:
        ref_k = manager._reconstruct_packed_dequant_wht(
            packed=cache.k_packed,
            scales=cache.k_scales,
            n_values=cache.k_n_values,
            shape=cache.shape,
            bits=cache.k_bits,
            seed=cache.seed,
            use_wht=cache.use_wht,
            use_incoherent_signs=cache.use_incoherent_signs,
            out_dtype=mx.float32,
        )
        ref_v = manager._reconstruct_packed_dequant_wht(
            packed=cache.v_packed,
            scales=cache.v_scales,
            n_values=cache.v_n_values,
            shape=cache.shape,
            bits=cache.v_bits,
            seed=cache.seed,
            use_wht=cache.use_wht,
            use_incoherent_signs=cache.use_incoherent_signs,
            out_dtype=mx.float32,
        )
    mx.eval(k_rec, v_rec, ref_k, ref_v)

    assert manager.last_reconstruction_kernel == expected_label
    assert (
        manager.last_reconstruction_kernel != "metal_failed_fallback_reference"
    )
    assert manager.last_reconstruction_kernel.startswith("metal_")
    assert cosine_similarity(k_rec, ref_k) > 0.999
    assert cosine_similarity(v_rec, ref_v) > 0.999
    assert mx.max(mx.abs(k_rec - ref_k)).item() < 1e-3
    assert mx.max(mx.abs(v_rec - ref_v)).item() < 1e-3
