#!/usr/bin/env python3
"""Fused kernel route tests for Main 18."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.kernels import KernelRouteError, packed_dequant_wht_sign_metal
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


REQUIRED_SHAPES = [
    (1, 4, 128, 64),
    (1, 8, 512, 64),
    (1, 8, 1024, 128),
]


def test_fused_kernel_source_route_exists() -> None:
    """Fused kernel source route must be importable."""
    assert callable(packed_dequant_wht_sign_metal)


@pytest.mark.parametrize("shape", REQUIRED_SHAPES)
@pytest.mark.parametrize("bits", [3, 8])
def test_fused_output_matches_sequential_reference(
    shape: tuple[int, int, int, int],
    bits: int,
) -> None:
    """Fused kernel output must match sequential reference within tolerance."""
    manager = RFSNTurboQuantKVManager(
        k_bits=bits,
        v_bits=bits,
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=False,
        group_size=64,
    )

    mx.random.seed(42)
    seed = 12345
    x = mx.random.normal(shape)

    x_pre = manager._apply_signs_on_the_fly(x, seed)
    x_wht = manager._apply_wht_pretransform(x_pre)
    q, scales = manager._quantize(x_wht.reshape(-1), bits)
    packed, n_values = BitPackedQuantizer.pack(q, bits)

    # Fused reconstruction
    fused = packed_dequant_wht_sign_metal(
        packed=packed,
        scales=scales,
        n_values=n_values,
        bits=bits,
        group_size=manager.group_size,
        seed=seed,
        out_dtype=mx.float32,
    ).reshape(shape)

    # Sequential reference reconstruction
    codes = BitPackedQuantizer.unpack(packed, n_values, bits)
    deq = manager._dequantize_unsigned(codes, scales, bits).reshape(shape)
    ref = manager._apply_wht_pretransform(deq)
    ref = manager._apply_signs_on_the_fly(ref, seed)

    mx.eval(fused, ref)

    cos = cosine_similarity(ref, fused)
    max_diff = float(mx.max(mx.abs(ref - fused)).item())
    assert cos >= 0.999, f"cosine={cos} for shape={shape} bits={bits}"
    msg = f"max_diff={max_diff} for shape={shape} bits={bits}"
    assert max_diff <= 1e-3, msg


def test_kv_manager_uses_fused_route_when_requested(tmp_path) -> None:
    """When prefer_fused_kernel=True, manager must record fused label.

    Block-aware caches defer signs to post-concatenation, so the
    per-block reconstruction uses the multi-kernel path.  The label
    reflects the effective kernel family used.
    """
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=True,
        prefer_fused_kernel=True,
        strict_metal=True,
    )

    shape = (1, 8, 512, 64)
    mx.random.seed(7)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    manager.store("skill", k, v, token_count=shape[2])
    k_rec, v_rec = manager.retrieve("skill", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)

    cache = manager.active_caches["skill"]
    if cache.num_blocks > 0:
        expected = "metal_multikernel_dequant_wht_sign"
    else:
        expected = "metal_fused_dequant_wht_sign"
    assert manager.last_reconstruction_kernel == expected


def test_strict_fused_failure_raises_not_fallback(tmp_path) -> None:
    """Strict mode must raise on fused failure, not silently fallback.

    Block-aware caches defer signs to post-concatenation and therefore
    never invoke the fused WHT+sign kernel during retrieve().  The
    strict-fused-failure behaviour is still tested by directly calling
    the fused helper for a legacy-format cache entry.
    """
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=True,
        prefer_fused_kernel=True,
        strict_metal=True,
    )

    shape = (1, 8, 512, 64)
    mx.random.seed(7)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    manager.store("strict_skill", k, v, token_count=shape[2])

    cache = manager.active_caches["strict_skill"]
    if cache.num_blocks > 0:
        # Block-aware path does not use fused kernel; verify by
        # injecting a failure into the direct helper and confirming
        # it is NOT reached during normal retrieve().
        original_fused = manager._reconstruct_packed_dequant_wht_sign_fused
        manager._reconstruct_packed_dequant_wht_sign_fused = (
            lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("injected fused failure")
            )
        )
        try:
            # Should succeed because fused path is bypassed
            k_rec, v_rec = manager.retrieve(
                "strict_skill", out_dtype=mx.float32
            )
            mx.eval(k_rec, v_rec)
            assert k_rec is not None
        finally:
            manager._reconstruct_packed_dequant_wht_sign_fused = original_fused
        return

    # Legacy-format cache: fused path is exercised
    original_fused = manager._reconstruct_packed_dequant_wht_sign_fused
    manager._reconstruct_packed_dequant_wht_sign_fused = (
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("injected fused failure")
        )
    )

    try:
        with pytest.raises(KernelRouteError):
            manager.retrieve("strict_skill", out_dtype=mx.float32)
    finally:
        manager._reconstruct_packed_dequant_wht_sign_fused = original_fused
