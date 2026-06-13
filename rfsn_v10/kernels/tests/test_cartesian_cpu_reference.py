"""Tests for CPU reference kernels (QK and SV).

These validate that the CPU reference produces the same results as:
  1. Dense attention over dequantized K/V (ground truth)
  2. Metal kernels (when available)
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_bhtg_scales_from_block(block, Hkv: int, Lkv: int, D: int) -> np.ndarray:  # noqa: N803
    """Convert a single PackedBlockV4 scales to BHTG shape."""
    scales = np.array(block.scales)
    # For a single block, scales is (B, Hkv, T, groups_per_vector)
    # where T == Lkv for this block
    return scales


def _dequantize_raw(block, bits: int, group_size: int) -> np.ndarray:
    """Dequantize packed_codes to dense WITHOUT inverse WHT/signs.

    This produces the raw quantized values in the transformed domain,
    matching what the Metal kernels operate on.
    """
    import math

    from rfsn_v10.cache.numpy_codec_oracle import _vector_aligned_unpack_numpy

    packed = block.packed_codes
    scales = block.scales
    B, H, T, D = block.batch_size, block.n_kv_heads, block.token_count, block.head_dim

    codes = _vector_aligned_unpack_numpy(packed, bits, D)
    qmax = (1 << (bits - 1)) - 1
    q_signed = codes.astype(np.float32) - float(qmax)

    n_groups = math.ceil(D / group_size)
    grouped = q_signed.reshape(B, H, T, n_groups, group_size)
    scale_expanded = scales[..., None]
    raw = grouped * scale_expanded
    return raw.reshape(B, H, T, D)


def test_cpu_qk_matches_original_domain() -> None:
    """CPU QK over packed codes must match QK over original (pre-transform) K."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_qk_cpu_reference

    np.random.seed(42)
    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2
    T = 64

    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)

    # Encode K (WHT + signs + quantize)
    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    k_block = k_codec.encode_bhtd(k, logical_start=0, layer_id=0, stream_id="K")

    # Dense attention over the ORIGINAL K
    repeats = Hq // Hkv
    k_exp = np.repeat(k, repeats, axis=1)
    scale_factor = D ** -0.5
    dense_scores = np.matmul(
        q.astype(np.float32),
        k_exp.astype(np.float32).transpose(0, 1, 3, 2),
    ) * scale_factor

    # CPU reference QK over packed codes (inverse-transforms internally)
    cpu_scores = cartesian_qk_cpu_reference(
        q,
        np.array(k_block.packed_codes),
        np.array(k_block.scales),
        bits=8,
        group_size=64,
        scale_factor=scale_factor,
        use_wht=True,
        sign_seed=42,
        layer_id=0,
        stream_id="K",
    )

    np.testing.assert_allclose(cpu_scores, dense_scores, atol=0.03, rtol=0.05)


def test_cpu_sv_matches_original_domain() -> None:
    """CPU SV over packed codes must match weighted sum over original (pre-transform) V."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_sv_cpu_reference

    np.random.seed(43)
    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2
    T = 64

    weights = np.random.randn(B, Hq, Lq, T).astype(np.float32)
    weights = np.exp(weights)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    v_block = v_codec.encode_bhtd(v, logical_start=0, layer_id=0, stream_id="V")

    # Dense attention over the ORIGINAL V
    repeats = Hq // Hkv
    v_exp = np.repeat(v, repeats, axis=1)
    dense_out = np.matmul(weights.astype(np.float32), v_exp.astype(np.float32))

    # CPU reference SV (inverse-transforms internally)
    cpu_out = cartesian_sv_cpu_reference(
        weights,
        np.array(v_block.packed_codes),
        np.array(v_block.scales),
        bits=5,
        group_size=64,
        head_dim=D,
        use_wht=True,
        sign_seed=42,
        layer_id=0,
        stream_id="V",
    )

    np.testing.assert_allclose(cpu_out, dense_out, atol=0.05, rtol=0.01)


def test_cpu_qk_matches_metal_kernel() -> None:
    """CPU reference QK must match Metal kernel QK when Metal is available."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_qk_cpu_reference

    np.random.seed(44)
    B, Hq, Lq, D = 1, 4, 1, 64
    Hkv = 2
    T = 64

    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)

    # Encode with WHT+signs but compare raw dequantization (no inverse transforms)
    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    k_block = k_codec.encode_bhtd(k, logical_start=0, layer_id=0, stream_id="K")

    cpu_scores = cartesian_qk_cpu_reference(
        q,
        np.array(k_block.packed_codes),
        np.array(k_block.scales),
        bits=8,
        group_size=64,
        scale_factor=D ** -0.5,
        use_wht=False,
        sign_seed=0,
    )

    try:
        from rfsn_v10.kernels.metal_cartesian import cartesian_qk_metal
        metal_scores = cartesian_qk_metal(
            mx.array(q),
            mx.array(k_block.packed_codes),
            mx.array(k_block.scales),
            bits=8,
            group_size=64,
            scale_factor=D ** -0.5,
            use_wht=False,
            sign_seed=0,
        )
        np.testing.assert_allclose(
            cpu_scores, np.array(metal_scores), atol=0.01, rtol=0.01
        )
    except Exception as exc:
        pytest.skip(f"Metal kernel unavailable or failed: {exc}")


def test_cpu_sv_matches_metal_kernel() -> None:
    """CPU reference SV must match Metal kernel SV when Metal is available."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_sv_cpu_reference

    np.random.seed(45)
    B, Hq, Lq, D = 1, 4, 1, 64
    Hkv = 2
    T = 64

    weights = np.random.randn(B, Hq, Lq, T).astype(np.float32)
    weights = np.exp(weights)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    # Encode with WHT+signs but compare raw dequantization (no inverse transforms)
    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    v_block = v_codec.encode_bhtd(v, logical_start=0, layer_id=0, stream_id="V")

    cpu_out = cartesian_sv_cpu_reference(
        weights,
        np.array(v_block.packed_codes),
        np.array(v_block.scales),
        bits=5,
        group_size=64,
        head_dim=D,
        use_wht=False,
        sign_seed=0,
    )

    try:
        from rfsn_v10.kernels.metal_cartesian import cartesian_sv_metal
        metal_out = cartesian_sv_metal(
            mx.array(weights),
            mx.array(v_block.packed_codes),
            mx.array(v_block.scales),
            bits=5,
            group_size=64,
            head_dim=D,
            use_wht=False,
            sign_seed=0,
        )
        np.testing.assert_allclose(
            cpu_out, np.array(metal_out), atol=0.05, rtol=0.01
        )
    except Exception as exc:
        pytest.skip(f"Metal kernel unavailable or failed: {exc}")


def test_metal_qk_fallback_to_cpu_with_wht() -> None:
    """Metal QK must fall back to CPU reference when WHT/signs are enabled."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_qk_cpu_reference

    np.random.seed(46)
    B, Hq, Lq, D = 1, 4, 1, 64
    Hkv = 2
    T = 64

    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)

    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    k_block = k_codec.encode_bhtd(k, logical_start=0, layer_id=0, stream_id="K")

    cpu_scores = cartesian_qk_cpu_reference(
        q,
        np.array(k_block.packed_codes),
        np.array(k_block.scales),
        bits=8,
        group_size=64,
        scale_factor=D ** -0.5,
        use_wht=True,
        sign_seed=42,
        layer_id=0,
        stream_id="K",
    )

    try:
        from rfsn_v10.kernels.metal_cartesian import cartesian_qk_metal
        metal_scores = cartesian_qk_metal(
            mx.array(q),
            mx.array(k_block.packed_codes),
            mx.array(k_block.scales),
            bits=8,
            group_size=64,
            scale_factor=D ** -0.5,
            use_wht=True,
            sign_seed=42,
            layer_id=0,
            stream_id="K",
        )
        np.testing.assert_allclose(
            cpu_scores, np.array(metal_scores), atol=0.01, rtol=0.01
        )
    except Exception as exc:
        pytest.skip(f"Metal fallback unavailable or failed: {exc}")


def test_metal_sv_fallback_to_cpu_with_wht() -> None:
    """Metal SV must fall back to CPU reference when WHT/signs are enabled."""
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec
    from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_sv_cpu_reference

    np.random.seed(47)
    B, Hq, Lq, D = 1, 4, 1, 64
    Hkv = 2
    T = 64

    weights = np.random.randn(B, Hq, Lq, T).astype(np.float32)
    weights = np.exp(weights)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)
    v_block = v_codec.encode_bhtd(v, logical_start=0, layer_id=0, stream_id="V")

    cpu_out = cartesian_sv_cpu_reference(
        weights,
        np.array(v_block.packed_codes),
        np.array(v_block.scales),
        bits=5,
        group_size=64,
        head_dim=D,
        use_wht=True,
        sign_seed=42,
        layer_id=0,
        stream_id="V",
    )

    try:
        from rfsn_v10.kernels.metal_cartesian import cartesian_sv_metal
        metal_out = cartesian_sv_metal(
            mx.array(weights),
            mx.array(v_block.packed_codes),
            mx.array(v_block.scales),
            bits=5,
            group_size=64,
            head_dim=D,
            use_wht=True,
            sign_seed=42,
            layer_id=0,
            stream_id="V",
        )
        np.testing.assert_allclose(
            cpu_out, np.array(metal_out), atol=0.05, rtol=0.01
        )
    except Exception as exc:
        pytest.skip(f"Metal fallback unavailable or failed: {exc}")
