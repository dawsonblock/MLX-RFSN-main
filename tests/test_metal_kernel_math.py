#!/usr/bin/env python3
"""
RFSN v10 - Metal Kernel Math Verification.
Validates:
- Bitpacking correctness
- Bitpacking rejection behavior
- Symmetric signed quantization
- Zero-bias behavior
- Walsh-Hadamard self-inverse behavior
- Deterministic randomized-sign preconditioning
- Packed-dequant-WHT reconstruction equivalence in FP32 proof mode
- Store/retrieve quality bounds
- Cache pinning memory-budget enforcement
Run:
    pytest tests/test_metal_kernel_math.py -v -s
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


# ==============================================================================
# Metrics
# ==============================================================================
def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()
def mae(a: mx.array, b: mx.array) -> float:
    return mx.mean(mx.abs(a - b)).item()
def rel_mae(a: mx.array, b: mx.array) -> float:
    denom = mx.maximum(mx.mean(mx.abs(a)), mx.array(1e-8))
    return (mx.mean(mx.abs(a - b)) / denom).item()
# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def kv_manager(tmp_path) -> RFSNTurboQuantKVManager:
    return RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_incoherent=True,
        max_memory_gb=0.1,
        max_pinned_memory_gb=0.05,
        cache_dir=str(tmp_path),
    )
@pytest.fixture
def kv_manager_plain(tmp_path) -> RFSNTurboQuantKVManager:
    return RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_incoherent=False,
        max_memory_gb=0.1,
        max_pinned_memory_gb=0.05,
        cache_dir=str(tmp_path),
    )
# ==============================================================================
# Bitpack Tests
# ==============================================================================
@pytest.mark.parametrize("bits", [0, 1, 9, 16, -1])
def test_bitpack_rejects_invalid_widths(bits: int) -> None:
    x = mx.array([0, 1, 2], dtype=mx.uint32)
    with pytest.raises(ValueError):
        BitPackedQuantizer.pack(x, bits)
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("n_values", [2, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000])
def test_bitpack_uint_roundtrip(bits: int, n_values: int) -> None:
    max_val = (1 << bits) - 1
    edge = mx.array([0, max_val], dtype=mx.uint32)
    if n_values > 2:
        rand = mx.random.randint(
            0,
            max_val + 1,
            (n_values - 2,),
            dtype=mx.uint32,
        )
        original = mx.concatenate([edge, rand])
    else:
        original = edge[:n_values]
    packed, n_v = BitPackedQuantizer.pack(original, bits)
    unpacked = BitPackedQuantizer.unpack(packed, n_v, bits)
    mx.eval(unpacked)
    assert n_v == n_values
    assert mx.array_equal(original, unpacked).item()
def test_bitpack_rejects_fractional_float_codes() -> None:
    x = mx.array([0.0, 1.5, 2.0], dtype=mx.float32)
    with pytest.raises(ValueError, match="integer"):
        BitPackedQuantizer.pack(x, 3)
def test_bitpack_accepts_integer_valued_float_codes() -> None:
    x = mx.array([0.0, 1.0, 2.0, 7.0], dtype=mx.float32)
    packed, n = BitPackedQuantizer.pack(x, 3)
    unpacked = BitPackedQuantizer.unpack(packed, n, 3)
    mx.eval(unpacked)
    assert n == 4
    assert mx.array_equal(unpacked, mx.array([0, 1, 2, 7], dtype=mx.uint32)).item()
def test_bitpack_rejects_negative_values() -> None:
    x = mx.array([0, 1, -1], dtype=mx.int32)
    with pytest.raises(ValueError, match="negative"):
        BitPackedQuantizer.pack(x, 3)
def test_bitpack_rejects_out_of_range_values() -> None:
    x_bad = mx.array([0, 1, 8], dtype=mx.uint32)
    with pytest.raises(ValueError, match="exceed"):
        BitPackedQuantizer.pack(x_bad, 3)
def test_bitpack_unpack_rejects_too_small_buffer() -> None:
    packed = mx.array([0], dtype=mx.uint32)
    with pytest.raises(ValueError, match="too small"):
        BitPackedQuantizer.unpack(packed, n_values=100, bits=3)
def test_bitpack_unpack_rejects_empty_buffer() -> None:
    packed = mx.array([], dtype=mx.uint32)
    with pytest.raises(ValueError, match="empty"):
        BitPackedQuantizer.unpack(packed, n_values=10, bits=3)
def test_bitpack_unpack_rejects_bad_n_values() -> None:
    packed = mx.array([0], dtype=mx.uint32)
    with pytest.raises(ValueError, match="n_values"):
        BitPackedQuantizer.unpack(packed, n_values=0, bits=3)
# ==============================================================================
# Sign Preconditioning Tests
# ==============================================================================
def test_incoherent_signs_are_self_inverse(kv_manager: RFSNTurboQuantKVManager) -> None:
    mx.random.seed(42)
    x = mx.random.normal((1, 2, 128, 64))
    seed = 12345
    y = kv_manager._apply_signs_on_the_fly(x, seed)
    z = kv_manager._apply_signs_on_the_fly(y, seed)
    mx.eval(z)
    diff = mx.max(mx.abs(x - z)).item()
    assert diff < 1e-6, f"Deterministic sign transform failed self-inverse check. diff={diff}"
def test_incoherent_signs_different_seed_changes_pattern(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    mx.random.seed(42)
    x = mx.random.normal((1, 2, 128, 64))
    y1 = kv_manager._apply_signs_on_the_fly(x, 12345)
    y2 = kv_manager._apply_signs_on_the_fly(x, 54321)
    mx.eval(y1, y2)
    # Different seeds should usually produce different sign layouts.
    diff = mx.mean(mx.abs(y1 - y2)).item()
    assert diff > 0.01
# ==============================================================================
# Quantization / Dequantization Tests
# ==============================================================================
def test_quantize_dequant_contract_roundtrip_and_zero_bias(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    x = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=mx.float32)
    bits = 3
    q, s = kv_manager._quantize(x, bits)
    restored = kv_manager._dequantize_unsigned(q, s, bits)
    mx.eval(q, restored)
    error = mx.mean(mx.abs(x - restored)).item()
    # 3-bit symmetric quantization is coarse. This threshold catches
    # catastrophic contract mismatches while accepting expected quantization noise.
    assert error < 0.2, f"Quant/dequant contract broken. MAE={error}"
    # Exact zero should reconstruct as exactly zero in symmetric quantization.
    assert restored[2].item() == 0.0
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_quantize_output_codes_within_symmetric_range(
    kv_manager: RFSNTurboQuantKVManager,
    bits: int,
) -> None:
    mx.random.seed(123)
    x = mx.random.normal((129,))
    q, scales = kv_manager._quantize(x, bits)
    mx.eval(q, scales)
    qmax = (1 << (bits - 1)) - 1
    max_valid_code = 2 * qmax
    assert int(mx.min(q).item()) >= 0
    assert int(mx.max(q).item()) <= max_valid_code
    assert scales.size == (x.size + kv_manager.group_size - 1) // kv_manager.group_size
def test_dequant_rejects_invalid_symmetric_code(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    # For 3-bit symmetric quantization:
    # qmax = 3, valid shifted codes are 0..6.
    # Code 7 is valid raw bitpack data but invalid symmetric quant data.
    q = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint32)
    scales = mx.array([1.0], dtype=mx.float32)
    with pytest.raises(ValueError, match="Invalid symmetric quant code"):
        kv_manager._dequantize_unsigned(q, scales, 3)
def test_dequant_rejects_bad_scale_count(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    q = mx.array([0, 1, 2, 3, 4], dtype=mx.uint32)
    scales = mx.array([], dtype=mx.float32)
    with pytest.raises(ValueError, match="Scale count mismatch"):
        kv_manager._dequantize_unsigned(q, scales, 3)
def test_quantize_rejects_empty_tensor(kv_manager: RFSNTurboQuantKVManager) -> None:
    x = mx.array([], dtype=mx.float32)
    with pytest.raises(ValueError, match="empty"):
        kv_manager._quantize(x, 3)
# ==============================================================================
# Walsh-Hadamard Tests
# ==============================================================================
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_wht_is_self_inverse_or_normalized(
    kv_manager: RFSNTurboQuantKVManager,
    head_dim: int,
) -> None:
    mx.random.seed(42)
    shape = (1, 2, 128, head_dim)
    x = mx.random.normal(shape)
    y = kv_manager._apply_wht_pretransform(x)
    z = kv_manager._apply_wht_pretransform(y)
    mx.eval(z)
    diff = mx.max(mx.abs(x - z)).item()
    assert diff < 1e-4, f"WHT is not self-inverse for head_dim={head_dim}. diff={diff}"
def test_wht_rejects_bad_head_dim(kv_manager: RFSNTurboQuantKVManager) -> None:
    x = mx.random.normal((1, 2, 128, 80))
    with pytest.raises(ValueError, match="Last dimension"):
        kv_manager._apply_wht_pretransform(x)
def test_wht_rejects_non_64_block(kv_manager: RFSNTurboQuantKVManager) -> None:
    x = mx.random.normal((1, 2, 128, 64))
    with pytest.raises(ValueError, match="exactly 64"):
        kv_manager._apply_wht_pretransform(x, wht_block=32)
# ==============================================================================
# Packed-dequant-WHT Reconstruction Equivalence Tests
# ==============================================================================
@pytest.mark.parametrize("bits", [3, 8])
@pytest.mark.parametrize("use_incoherent", [False, True])
def test_reconstruct_wht_matches_discrete_math(
    kv_manager: RFSNTurboQuantKVManager,
    bits: int,
    use_incoherent: bool,
) -> None:
    mx.random.seed(42)
    shape = (1, 2, 128, 64)
    x_fp32 = mx.random.normal(shape)
    seed = 12345
    x_pre = (
        kv_manager._apply_signs_on_the_fly(x_fp32, seed)
        if use_incoherent
        else x_fp32
    )
    x_wht = kv_manager._apply_wht_pretransform(x_pre)
    q_x, s_x = kv_manager._quantize(x_wht, bits)
    packed_x, n_x = BitPackedQuantizer.pack(q_x, bits)
    unpacked_q = BitPackedQuantizer.unpack(packed_x, n_x, bits)
    dequant = kv_manager._dequantize_unsigned(unpacked_q, s_x, bits)
    sequential_restored = kv_manager._apply_wht_pretransform(
        dequant.reshape(shape)
    )
    if use_incoherent:
        sequential_restored = kv_manager._apply_signs_on_the_fly(
            sequential_restored,
            seed,
        )
    fused_restored = kv_manager._reconstruct_packed_dequant_wht(
        packed=packed_x,
        scales=s_x,
        n_values=n_x,
        shape=shape,
        bits=bits,
        seed=seed,
        use_incoherent=use_incoherent,
        out_dtype=mx.float32,
    )
    mx.eval(sequential_restored, fused_restored)
    diff = mx.max(mx.abs(sequential_restored - fused_restored)).item()
    assert diff < 1e-4, (
        f"Packed-dequant-WHT reconstruction diverges from sequential math "
        f"(bits={bits}, use_incoherent={use_incoherent}). diff={diff}"
    )
def test_reconstruct_packed_dequant_wht_rejects_bad_packed_size(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    packed = mx.array([0], dtype=mx.uint32)
    scales = mx.ones((2,), dtype=mx.float32)
    with pytest.raises(ValueError, match="Packed buffer too small"):
        kv_manager._reconstruct_packed_dequant_wht(
            packed=packed,
            scales=scales,
            n_values=100,
            shape=(1, 1, 100, 1),
            bits=3,
            seed=1,
            use_incoherent=False,
            out_dtype=mx.float32,
        )
def test_reconstruct_packed_dequant_wht_rejects_bad_scale_count(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    q = mx.array([0, 1, 2, 3, 4], dtype=mx.uint32)
    packed, n = BitPackedQuantizer.pack(q, 3)
    with pytest.raises(ValueError, match="Scale count mismatch"):
        kv_manager._reconstruct_packed_dequant_wht(
            packed=packed,
            scales=mx.array([], dtype=mx.float32),
            n_values=n,
            shape=(1, 1, 5, 1),
            bits=3,
            seed=1,
            use_incoherent=False,
            out_dtype=mx.float32,
        )
def test_reconstruct_packed_dequant_wht_rejects_bad_shape_product(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    q = mx.array([0, 1, 2, 3, 4], dtype=mx.uint32)
    packed, n = BitPackedQuantizer.pack(q, 3)
    scales = mx.ones((1,), dtype=mx.float32)
    with pytest.raises(ValueError, match="Shape product"):
        kv_manager._reconstruct_packed_dequant_wht(
            packed=packed,
            scales=scales,
            n_values=n,
            shape=(1, 1, 6, 1),
            bits=3,
            seed=1,
            use_incoherent=False,
            out_dtype=mx.float32,
        )
def test_reconstruct_packed_dequant_wht_rejects_bad_out_dtype(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    q = mx.array([0, 1, 2, 3, 4], dtype=mx.uint32)
    packed, n = BitPackedQuantizer.pack(q, 3)
    scales = mx.ones((1,), dtype=mx.float32)
    with pytest.raises(ValueError, match="out_dtype"):
        kv_manager._reconstruct_packed_dequant_wht(
            packed=packed,
            scales=scales,
            n_values=n,
            shape=(1, 1, 5, 1),
            bits=3,
            seed=1,
            use_incoherent=False,
            out_dtype=mx.int32,
        )
# ==============================================================================
# Store / Retrieve Tests
# ==============================================================================
def test_store_rejects_bad_head_dim(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128, 80))
    v = mx.random.normal((1, 8, 128, 80))
    with pytest.raises(ValueError, match="Head dimension"):
        kv_manager.store("bad_head_dim", k, v, 128)
def test_store_rejects_shape_mismatch(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 64, 64))
    with pytest.raises(ValueError, match="shape mismatch"):
        kv_manager.store("shape_mismatch", k, v, 128)
def test_store_rejects_bad_rank(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128))
    v = mx.random.normal((1, 8, 128))
    with pytest.raises(ValueError, match="Expected KV shape"):
        kv_manager.store("bad_rank", k, v, 128)
def test_store_rejects_bad_token_count(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    with pytest.raises(ValueError, match="token_count"):
        kv_manager.store("bad_token_count", k, v, 0)
def test_store_retrieve_roundtrip_quality_bounds(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    mx.random.seed(123)
    shape = (1, 8, 2048, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    kv_manager.store("test_skill", k, v, 2048)
    result = kv_manager.retrieve("test_skill", out_dtype=mx.float32)
    assert result is not None
    k_rec, v_rec = result
    mx.eval(k_rec, v_rec)
    assert k_rec.shape == shape
    assert v_rec.shape == shape
    # Keys are 8-bit and should keep high fidelity.
    assert cosine_similarity(k, k_rec) > 0.95
    assert rel_mae(k, k_rec) < 0.20
    # Values are 3-bit and are expected to degrade more.
    assert cosine_similarity(v, v_rec) > 0.75
    assert rel_mae(v, v_rec) < 0.60
def test_retrieve_returns_none_for_missing_cache(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    assert kv_manager.retrieve("missing") is None
def test_retrieve_rejects_format_version_mismatch(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    kv_manager.store("versioned", k, v, 128)
    kv_manager.active_caches["versioned"].format_version = "bad_format"
    with pytest.raises(ValueError, match="Unsupported cache format"):
        kv_manager.retrieve("versioned", out_dtype=mx.float32)
def test_retrieve_rejects_quant_metadata_mismatch(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    kv_manager.store("metadata", k, v, 128)
    kv_manager.active_caches["metadata"].k_bits = 4
    with pytest.raises(ValueError, match="metadata"):
        kv_manager.retrieve("metadata", out_dtype=mx.float32)
# ==============================================================================
# Memory Budget / Pinning Tests
# ==============================================================================
def test_pin_cache_budget_enforcement(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    shape = (1, 8, 2048, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    kv_manager.store("cache_1", k, v, 2048)
    cache_size_gb = (
        kv_manager._estimate_cache_bytes(kv_manager.active_caches["cache_1"])
        / (1024 ** 3)
    )
    # Allow one cache but reject two.
    kv_manager.max_pinned_memory_gb = cache_size_gb * 1.5
    assert kv_manager.pincache("cache_1") is True
    kv_manager.store("cache_2", k, v, 2048)
    with pytest.raises(MemoryError):
        kv_manager.pincache("cache_2")
def test_pin_cache_returns_false_for_missing_cache(
    kv_manager: RFSNTurboQuantKVManager,
) -> None:
    assert kv_manager.pincache("missing") is False
def test_store_raises_if_cache_too_large_for_active_budget(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_incoherent=True,
        max_memory_gb=1e-9,
        max_pinned_memory_gb=0.05,
        cache_dir=str(tmp_path),
    )
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    with pytest.raises(MemoryError):
        manager.store("too_big", k, v, 128)
