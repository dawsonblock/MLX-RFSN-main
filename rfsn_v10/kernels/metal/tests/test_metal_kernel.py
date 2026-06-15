"""Tests for the Metal packed attention kernel.

This tests the new mx.fast.metal_kernel based implementation.
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
    from rfsn_v10.kernels.metal.packed_attention_metal import metal_packed_attention
    
    # Create simple test data
    B, Hq, Lq, D = 1, 2, 4, 8
    Hkv = 1
    Lkv = 16
    
    queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
    keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
    scale = D ** -0.5
    
    # Run Metal kernel
    output = metal_packed_attention(queries, keys, values, scale, causal=True)
    
    assert output.shape == (B, Hq, Lq, D)
    assert output.dtype == mx.float32
    
    # Verify output is not all zeros or NaN
    assert float(mx.max(mx.abs(output))) > 0.0
    assert not mx.any(mx.isnan(output))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_shape_variations():
    """Test Metal kernel with different shapes."""
    from rfsn_v10.kernels.metal.packed_attention_metal import metal_packed_attention
    
    test_cases = [
        # (B, Hq, Lq, D, Hkv, Lkv)
        (1, 1, 2, 4, 1, 8),
        (1, 4, 4, 16, 2, 32),
        (1, 8, 2, 32, 2, 64),
    ]
    
    for B, Hq, Lq, D, Hkv, Lkv in test_cases:
        queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
        keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        scale = D ** -0.5
        
        output = metal_packed_attention(queries, keys, values, scale, causal=True)
        
        assert output.shape == (B, Hq, Lq, D), f"Shape mismatch for case {(B, Hq, Lq, D, Hkv, Lkv)}"
        assert not mx.any(mx.isnan(output)), f"NaN detected for case {(B, Hq, Lq, D, Hkv, Lkv)}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_metal_kernel_imports():
    """Test that all Metal kernel modules can be imported."""
    from rfsn_v10.kernels.metal.packed_attention_metal import (
        metal_packed_attention,
        attend_metal,
    )
    
    assert callable(metal_packed_attention)
    assert callable(attend_metal)
