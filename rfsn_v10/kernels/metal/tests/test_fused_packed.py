"""Tests for fused Metal packed attention kernel scaffold.

P0 #7: This is a scaffold, not a functional implementation.

The Metal kernel is not yet implemented. These tests verify:
- Wrapper initialization
- CPU fallback behavior
- Comparison infrastructure (using CPU fallback)

Real Metal execution tests are skipped until the kernel is implemented.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_fused_packed_wrapper_init():
    """Test Metal kernel wrapper initialization."""
    from rfsn_v10.kernels.metal.fused_packed_wrapper import FusedPackedAttentionMetal
    
    wrapper = FusedPackedAttentionMetal(bits=8, group_size=64)
    assert wrapper.bits == 8
    assert wrapper.group_size == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_fused_packed_cpu_fallback():
    """Test CPU fallback when Metal is not available."""
    from rfsn_v10.kernels.metal.fused_packed_wrapper import FusedPackedAttentionMetal
    
    wrapper = FusedPackedAttentionMetal(bits=8, group_size=64)
    
    # Create dummy inputs
    B, H, T, D = 1, 2, 64, 64
    queries = mx.random.normal(shape=(B, H, T, D)).astype(mx.float32)
    packed_keys = mx.random.randint(0, 255, shape=(1, 2, 64, 64)).astype(mx.uint8)
    packed_values = mx.random.randint(0, 255, shape=(1, 2, 64, 64)).astype(mx.uint8)
    blocks = []
    
    # Should fallback to CPU reference without error
    output = wrapper(queries, packed_keys, packed_values, blocks)
    assert output.shape == (B, H, T, D)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_fused_packed_comparison():
    """Test comparison between Metal kernel and MLX reference.
    
    Phase 8.29: Compare Metal kernel against direct-packed MLX reference
    """
    from rfsn_v10.kernels.metal.fused_packed_wrapper import FusedPackedAttentionMetal
    
    wrapper = FusedPackedAttentionMetal(bits=8, group_size=64)
    
    # Create dummy inputs
    B, H, T, D = 1, 2, 64, 64
    queries = mx.random.normal(shape=(B, H, T, D)).astype(mx.float32)
    packed_keys = mx.random.randint(0, 255, shape=(1, 2, 64, 64)).astype(mx.uint8)
    packed_values = mx.random.randint(0, 255, shape=(1, 2, 64, 64)).astype(mx.uint8)
    blocks = []
    
    # Compare with reference (will use CPU fallback)
    metrics = wrapper.compare_with_reference(queries, packed_keys, packed_values, blocks)
    
    # When using CPU fallback, error should be zero
    assert metrics["max_abs_error"] == 0.0
    assert metrics["mean_abs_error"] == 0.0
    assert metrics["passed"] is True


@pytest.mark.skip(reason="P0 #7: Metal kernel is a scaffold, not implemented")
def test_fused_packed_metal_execution():
    """Test actual Metal kernel execution.
    
    P0 #7: This test is skipped because the Metal kernel is a scaffold.
    
    When the kernel is actually implemented, this test should:
    1. Load the Metal kernel
    2. Execute on GPU
    3. Compare against CPU reference
    4. Verify numerical accuracy
    
    A skipped test is not evidence of a working implementation.
    """
    pass
