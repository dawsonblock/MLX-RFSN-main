"""Tests for the Metal dense attention over reconstructed KV kernel.

This tests the mx.fast.metal_kernel based implementation against the
MLX reference implementation for numerical correctness.

NOTE: These tests verify the transitional dense-attention Metal kernel,
NOT a true packed attention kernel. The current implementation decodes all
compressed blocks to dense tensors and runs dense attention. A true packed
Metal kernel that consumes packed codes inside the shader is future work.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_basic():
    """Test that the Metal kernel module can be imported and run."""
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        metal_dense_attention_over_reconstructed_kv,
    )

    B, Hq, Lq, D = 1, 2, 4, 8
    Hkv, Lkv = 1, 16

    queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
    keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    scale = D ** -0.5

    output = metal_dense_attention_over_reconstructed_kv(
        queries, keys, values, scale, causal=True
    )

    assert output.shape == (B, Hq, Lq, D)
    assert output.dtype == mx.float32
    assert float(mx.max(mx.abs(output))) > 0.0
    assert not mx.any(mx.isnan(output))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_vs_reference():
    """Compare Metal kernel output against MLX reference for numerical match."""
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        metal_dense_attention_over_reconstructed_kv,
    )

    B, Hq, Lq, D = 1, 2, 4, 8
    Hkv, Lkv = 1, 16

    queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
    keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    scale = D ** -0.5

    # Metal output
    metal_output = metal_dense_attention_over_reconstructed_kv(
        queries, keys, values, scale, causal=True
    )

    # Reference: MLX built-in causal attention with GQA repeat
    keys_rep = mx.repeat(keys, Hq // Hkv, axis=1)
    values_rep = mx.repeat(values, Hq // Hkv, axis=1)
    scores = mx.matmul(queries, keys_rep.transpose(0, 1, 3, 2)) * scale
    q_pos = mx.arange(Lq)[:, None]
    kv_pos = mx.arange(Lkv)[None, :]
    causal_mask = q_pos >= kv_pos
    causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (B, Hq, Lq, Lkv))
    scores = mx.where(causal_mask, scores, mx.array(-float("inf")))
    weights = mx.softmax(scores, axis=-1)
    ref_output = mx.matmul(weights, values_rep)

    max_error = float(mx.max(mx.abs(metal_output - ref_output)))
    mean_error = float(mx.mean(mx.abs(metal_output - ref_output)))

    print(f"Max error: {max_error:.8f}")
    print(f"Mean error: {mean_error:.8f}")

    # Should match within numerical tolerance
    assert max_error < 1e-3, f"Max error too large: {max_error}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_shape_variations():
    """Test Metal kernel with different shapes."""
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        metal_dense_attention_over_reconstructed_kv,
    )

    test_cases = [
        (1, 1, 2, 4, 1, 8),
        (1, 4, 4, 16, 2, 32),
        (1, 8, 2, 32, 2, 64),
    ]

    for B, Hq, Lq, D, Hkv, Lkv in test_cases:
        queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
        keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        scale = D ** -0.5

        output = metal_dense_attention_over_reconstructed_kv(
            queries, keys, values, scale, causal=True
        )

        assert output.shape == (B, Hq, Lq, D), f"Shape mismatch for case {(B, Hq, Lq, D, Hkv, Lkv)}"
        assert not mx.any(mx.isnan(output)), f"NaN detected for case {(B, Hq, Lq, D, Hkv, Lkv)}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_attend_metal_vs_reference():
    """Test attend_metal against reference attend using real quantized blocks."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.mlx_packed_attention_reference import attend
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.kernels.metal.packed_attention_metal import attend_metal

    # Create session with small staging to force block creation
    session = GenerationCacheSession(
        model_id="test",
        num_layers=1,
        key_codec=CartesianCodec(bits=8, group_size=64),
        value_codec=CartesianCodec(bits=8, group_size=64),
        staging_capacity=8,
        dense_residual_window=0,
    )

    lc = session.get_layer_cache(0)

    # Create fake K/V data
    B, H, T, D = 1, 2, 16, 64
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)

    # Append to cache (creates sealed blocks)
    lc.append(keys, values)

    # Create queries
    queries = mx.random.normal((B, H, 1, D)).astype(mx.float32)

    # Metal output
    metal_out, metal_scratch = attend_metal(queries, lc, scale=D**-0.5, causal=True)

    # Reset cache and re-append for reference
    lc.reset()
    lc.append(keys, values)

    # Reference output
    ref_out, ref_scratch = attend(queries, lc, scale=D**-0.5, causal=True)

    max_error = float(mx.max(mx.abs(metal_out - ref_out)))
    mean_error = float(mx.mean(mx.abs(metal_out - ref_out)))

    print(f"Max error: {max_error:.8f}")
    print(f"Mean error: {mean_error:.8f}")

    assert max_error < 1e-3, f"Max error too large: {max_error}"
    assert metal_scratch is not None


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_attend_metal_records_full_history_materialization():
    """Verify that attend_metal records full_history_materialization_calls."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.kernels.metal.packed_attention_metal import attend_metal

    session = GenerationCacheSession(
        model_id="test",
        num_layers=1,
        key_codec=CartesianCodec(bits=8, group_size=64),
        value_codec=CartesianCodec(bits=8, group_size=64),
        staging_capacity=8,
        dense_residual_window=0,
    )

    lc = session.get_layer_cache(0)

    B, H, T, D = 1, 2, 16, 64
    keys = mx.random.normal((B, H, T, D)).astype(mx.float16)
    values = mx.random.normal((B, H, T, D)).astype(mx.float16)
    lc.append(keys, values)

    queries = mx.random.normal((B, H, 1, D)).astype(mx.float32)

    # Record counters before
    before = session.runtime_counters.to_dict()

    attend_metal(queries, lc, scale=D**-0.5, causal=True)

    # Record counters after
    after = session.runtime_counters.to_dict()

    # Verify full_history_materialization_calls increased
    assert after["full_history_materialization_calls"] > before["full_history_materialization_calls"], \
        "full_history_materialization_calls should increase after Metal attention"
    # Verify packed_blocks_read increased
    assert after["packed_blocks_read"] > before["packed_blocks_read"], \
        "packed_blocks_read should increase after Metal attention"
    # Verify packed_bytes_read increased
    assert after["packed_bytes_read"] > before["packed_bytes_read"], \
        "packed_bytes_read should increase after Metal attention"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_attend_metal_strict_mode():
    """Verify that strict mode raises on failure instead of falling back."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        StrictPackedExecutionError,
        attend_metal,
    )

    session = GenerationCacheSession(
        model_id="test",
        num_layers=1,
        key_codec=CartesianCodec(bits=8, group_size=64),
        value_codec=CartesianCodec(bits=8, group_size=64),
        staging_capacity=8,
        dense_residual_window=0,
    )

    lc = session.get_layer_cache(0)

    # Empty cache with strict mode should raise
    queries = mx.random.normal((1, 2, 1, 64)).astype(mx.float32)

    # In strict mode with empty cache, it should raise
    with pytest.raises(StrictPackedExecutionError):
        attend_metal(queries, lc, scale=64**-0.5, causal=True, strict=True)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_imports():
    """Test that all Metal kernel modules can be imported."""
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        StrictPackedExecutionError,
        attend_metal,
        benchmark_metal_vs_reference,
        metal_available,
        metal_dense_attention_over_reconstructed_kv,
    )

    assert callable(metal_dense_attention_over_reconstructed_kv)
    assert callable(attend_metal)
    assert callable(benchmark_metal_vs_reference)
    assert callable(metal_available)
    assert issubclass(StrictPackedExecutionError, RuntimeError)
