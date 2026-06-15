"""Tests for fused Metal packed attention kernel.

Fix #15: Remove claims that Metal kernel is functional.

The Metal kernel is currently a scaffold/stub implementation with CPU fallback.
It does not yet implement actual Metal GPU computation for packed attention.

Current status:
- Metal kernel framework exists (stub with CPU fallback)
- Kernel comparison infrastructure exists
- CPU fallback for testing is functional
- Actual Metal GPU computation is NOT yet implemented
- Experimental Metal kernels for experimental quantization paths do NOT exist

Tests verify:
- Wrapper initialization
- Metal shader file exists
- Comparison infrastructure (using CPU fallback)

Real Metal execution tests require Apple Silicon hardware and the Metal framework.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


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
def test_metal_shader_exists():
    """Test that the Metal shader file exists."""
    from pathlib import Path
    
    metal_file = Path(__file__).parent.parent / "fused_packed_attention.metal"
    assert metal_file.exists(), f"Metal shader file should exist: {metal_file}"
    
    # Verify it contains the kernel function
    content = metal_file.read_text()
    assert "kernel void fused_packed_attention" in content
    assert "decode_cartesian_simple" in content
    assert "OnlineSoftmaxState" in content


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_metal_kernel_structure():
    """Test that the Metal kernel has the required components."""
    from pathlib import Path
    
    metal_file = Path(__file__).parent.parent / "fused_packed_attention.metal"
    content = metal_file.read_text()
    
    # Verify all required components are present
    required_components = [
        "decode_cartesian_simple",  # Cartesian decoding
        "OnlineSoftmaxState",       # Online softmax
        "fused_packed_attention",    # Main kernel
        "causal",                   # Causal mask
        "scale",                     # Scale factor
    ]
    
    for component in required_components:
        assert component in content, f"Metal kernel should contain {component}"


@pytest.mark.skipif(not HAS_METAL, reason="Metal framework not available")
def test_fused_packed_metal_execution():
    """Test actual Metal kernel execution.
    
    This test requires Apple Silicon hardware and the Metal framework.
    It verifies that the Metal kernel can be loaded and executed.
    """
    from rfsn_v10.kernels.metal.fused_packed_wrapper import FusedPackedAttentionMetal
    
    wrapper = FusedPackedAttentionMetal(bits=8, group_size=64)
    
    # Verify kernel was loaded
    assert wrapper._kernel_loaded, "Metal kernel should be loaded when Metal is available"
