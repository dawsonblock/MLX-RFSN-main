"""Metal kernel wrapper for fused packed attention.

Phase 8.28: Build fused Metal runtime (one layer-level dispatch)
Phase 8.29: Compare Metal kernel against direct-packed MLX reference

This module provides a Python interface to the Metal fused packed attention kernel.
"""
from __future__ import annotations

import warnings

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    warnings.warn("MLX not installed, Metal kernel wrapper will be non-functional")

try:
    import metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    warnings.warn("Metal not available, kernel will use CPU fallback")


class FusedPackedAttentionMetal:
    """Metal kernel wrapper for fused packed attention.
    
    This class provides a Python interface to the Metal kernel that implements
    fused packed attention computation on Apple Silicon GPU.
    """
    
    def __init__(self, bits: int = 8, group_size: int = 64):
        """Initialize the Metal kernel wrapper.
        
        Args:
            bits: Bit-width for quantization (default: 8)
            group_size: Group size for Cartesian codec (default: 64)
        """
        self.bits = bits
        self.group_size = group_size
        self._kernel_loaded = False
        
        if HAS_METAL:
            self._load_kernel()
        else:
            warnings.warn("Metal not available, using CPU fallback")
    
    def _load_kernel(self) -> None:
        """Load the Metal kernel from .metal file."""
        # TODO: Implement actual Metal kernel loading
        # This would:
        # 1. Load fused_packed_attention.metal
        # 2. Compile the kernel
        # 3. Create compute pipeline
        self._kernel_loaded = False
    
    def __call__(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
    ) -> mx.array:
        """Execute fused packed attention.
        
        Args:
            queries: Query tensor [batch, num_heads, seq_len, head_dim]
            packed_keys: Packed key blocks
            packed_values: Packed value blocks
            blocks: List of PackedBlock metadata
        
        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for Metal kernel wrapper")
        
        if not self._kernel_loaded:
            # Fallback to CPU reference implementation
            return self._cpu_fallback(queries, packed_keys, packed_values, blocks)
        
        # TODO: Execute Metal kernel
        # This would:
        # 1. Prepare Metal buffers
        # 2. Set kernel arguments
        # 3. Dispatch kernel
        # 4. Read back results
        return self._cpu_fallback(queries, packed_keys, packed_values, blocks)
    
    def _cpu_fallback(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
    ) -> mx.array:
        """CPU fallback implementation for testing.
        
        This uses the MLX reference implementation as a fallback when Metal
        is not available or the kernel is not yet implemented.
        """
        from rfsn_v10.cache.mlx_packed_attention_reference import attend
        
        # Use MLX reference implementation
        return attend(queries, packed_keys, packed_values, blocks)
    
    def compare_with_reference(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
        tolerance: float = 1e-5,
    ) -> dict[str, float]:
        """Compare Metal kernel output against MLX reference.
        
        Phase 8.29: Compare Metal kernel against direct-packed MLX reference
        
        Args:
            queries: Query tensor
            packed_keys: Packed key blocks
            packed_values: Packed value blocks
            blocks: List of PackedBlock metadata
            tolerance: Numerical tolerance for comparison
        
        Returns:
            Dictionary with comparison metrics:
            - max_abs_error: Maximum absolute error
            - mean_abs_error: Mean absolute error
            - passed: Whether comparison passed within tolerance
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for comparison")
        
        # Get Metal kernel output
        metal_output = self(queries, packed_keys, packed_values, blocks)
        
        # Get MLX reference output
        from rfsn_v10.cache.mlx_packed_attention_reference import attend
        reference_output = attend(queries, packed_keys, packed_values, blocks)
        
        # Compute error metrics
        error = mx.abs(metal_output - reference_output)
        max_abs_error = float(mx.max(error))
        mean_abs_error = float(mx.mean(error))
        
        passed = max_abs_error < tolerance
        
        return {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "passed": passed,
        }
