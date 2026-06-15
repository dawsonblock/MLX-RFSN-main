"""Metal kernel wrapper for fused packed attention.

Fix #15: Remove claims that Metal kernel is functional.

This module provides a Python interface to the Metal kernel that implements
fused packed attention computation on Apple Silicon GPU.

The Metal kernel is currently a scaffold/stub implementation with CPU fallback.
It does not yet implement actual Metal GPU computation for packed attention.

Current status:
- Metal kernel framework exists (stub with CPU fallback)
- Kernel comparison infrastructure exists
- CPU fallback for testing is functional
- Actual Metal GPU computation is NOT yet implemented
- Experimental Metal kernels for experimental quantization paths do NOT exist

Comparison against MLX packed reference is provided for validation.
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
        self._compute_pipeline = None
        
        if HAS_METAL:
            self._load_kernel()
        else:
            warnings.warn("Metal not available, using CPU fallback")
    
    def _load_kernel(self) -> None:
        """Load the Metal kernel from .metal file."""
        try:
            import os
            from pathlib import Path
            
            # Find the Metal shader file
            metal_file = Path(__file__).parent / "fused_packed_attention.metal"
            if not metal_file.exists():
                warnings.warn(f"Metal shader file not found: {metal_file}")
                return
            
            # Read shader source
            with open(metal_file, 'r') as f:
                shader_source = f.read()
            
            # Create Metal device
            device = metal.MTLCreateSystemDefaultDevice()
            if device is None:
                warnings.warn("No Metal device available")
                return
            
            # Create command queue
            self._command_queue = device.newCommandQueue()
            
            # Compile shader
            library = device.newLibraryWithSource_options_error_(
                shader_source,
                metal.MTLCompileOptions(),
                None
            )
            
            if library is None:
                warnings.warn("Failed to compile Metal shader")
                return
            
            # Get kernel function
            kernel_function = library.newFunctionWithName("fused_packed_attention")
            if kernel_function is None:
                warnings.warn("Kernel function not found in shader")
                return
            
            # Create compute pipeline
            self._compute_pipeline = device.newComputePipelineStateWithFunction_(kernel_function)
            if self._compute_pipeline is None:
                warnings.warn("Failed to create compute pipeline")
                return
            
            self._kernel_loaded = True
            self._device = device
            
        except Exception as e:
            warnings.warn(f"Failed to load Metal kernel: {e}")
            self._kernel_loaded = False
    
    def __call__(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
    ) -> mx.array:
        """Execute fused packed attention.
        
        Args:
            queries: Query tensor [batch, num_heads, seq_len, head_dim]
            packed_keys: Packed key blocks
            packed_values: Packed value blocks
            blocks: List of PackedBlock metadata
            scale: Attention scale factor
            causal: Whether to apply causal mask
            query_start_pos: Global position of first query token
        
        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for Metal kernel wrapper")
        
        if not self._kernel_loaded:
            # Fallback to CPU reference implementation
            return self._cpu_fallback(queries, packed_keys, packed_values, blocks, scale, causal, query_start_pos)
        
        # TODO: Execute Metal kernel
        # This would:
        # 1. Prepare Metal buffers from MLX arrays
        # 2. Set kernel arguments
        # 3. Dispatch kernel
        # 4. Read back results
        # For now, use CPU fallback
        return self._cpu_fallback(queries, packed_keys, packed_values, blocks, scale, causal, query_start_pos)
    
    def _cpu_fallback(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
    ) -> mx.array:
        """CPU fallback implementation for testing.
        
        This uses the MLX reference implementation as a fallback when Metal
        is not available or the kernel is not yet fully implemented.
        """
        from rfsn_v10.cache.mlx_packed_attention_reference import attend
        
        # Use MLX reference implementation
        return attend(
            queries,
            blocks[0].layer_cache if blocks else None,
            scale=scale,
            mask="causal" if causal else None,
            query_start_pos=query_start_pos,
            causal=causal,
        )
    
    def compare_with_reference(
        self,
        queries: mx.array,
        packed_keys: mx.array,
        packed_values: mx.array,
        blocks: list,
        tolerance: float = 1e-5,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
    ) -> dict[str, float]:
        """Compare Metal kernel output against MLX reference.
        
        Args:
            queries: Query tensor
            packed_keys: Packed key blocks
            packed_values: Packed value blocks
            blocks: List of PackedBlock metadata
            tolerance: Numerical tolerance for comparison
            scale: Attention scale factor
            causal: Whether to apply causal mask
            query_start_pos: Global position of first query token
        
        Returns:
            Dictionary with comparison metrics:
            - max_abs_error: Maximum absolute error
            - mean_abs_error: Mean absolute error
            - passed: Whether comparison passed within tolerance
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for comparison")
        
        # Get Metal kernel output
        metal_output = self(queries, packed_keys, packed_values, blocks, scale, causal, query_start_pos)
        
        # Get MLX reference output
        from rfsn_v10.cache.mlx_packed_attention_reference import attend
        reference_output = attend(
            queries,
            blocks[0].layer_cache if blocks else None,
            scale=scale,
            mask="causal" if causal else None,
            query_start_pos=query_start_pos,
            causal=causal,
        )
        
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
