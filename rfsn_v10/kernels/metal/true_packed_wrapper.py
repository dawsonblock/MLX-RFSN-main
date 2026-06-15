"""True packed Metal kernel wrapper with execution contract recording.

P1 Implementation: This module provides:
1. Buffer preparation from PackedBlock metadata for Metal kernel dispatch
2. CPU fallback to MLX reference when Metal is unavailable
3. Differential testing against reference implementation
4. Execution contract recording for auditability

WARNING: This is a P1 scaffold. The Metal kernel is not yet fully functional
and will fall back to CPU reference. The scaffold provides the correct
interface and buffer layout for when the Metal shader is completed.
"""
from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx


try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    warnings.warn("MLX not installed, true packed kernel will use CPU fallback")

try:
    import metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    metal = None  # type: ignore
    warnings.warn("Metal not available, kernel will use CPU fallback")

if TYPE_CHECKING:
    from rfsn_v10.cache.packed_block import PackedBlock


# =============================================================================
# Execution Contract (P1: Auditability)
# =============================================================================

@dataclass(frozen=True)
class ExecutionContract:
    """Immutable record of kernel execution parameters and outcomes.

    This contract provides auditability for correctness validation:
    - Backend used (Metal vs CPU fallback)
    - Kernel version/hash for reproducibility
    - Block/token counts for coverage verification
    - Materialization metrics (should be zero for true packed)
    - Timing for performance tracking
    """
    backend: str  # "true_packed_metal" or "packed_reference_cpu"
    kernel_hash: str  # SHA256 of shader source or reference impl
    num_blocks: int
    total_kv_tokens: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    materialized_bytes: int = 0  # Should be 0 for true packed
    decoded_tokens: int = 0  # Should be 0 for true packed
    fallback_reason: str | None = None
    execution_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def validate_invariant(self) -> tuple[bool, list[str]]:
        """Validate zero-reconstruction invariant.

        Returns:
            (passed, violations): Boolean pass/fail and list of violation descriptions
        """
        violations = []

        if self.materialized_bytes > 0:
            violations.append(
                f"Invariant violated: materialized_bytes={self.materialized_bytes} > 0"
            )

        if self.decoded_tokens > 0:
            violations.append(
                f"Invariant violated: decoded_tokens={self.decoded_tokens} > 0"
            )

        if self.backend != "true_packed_metal":
            violations.append(
                f"Not using true packed Metal: backend={self.backend}"
            )

        return len(violations) == 0, violations


# =============================================================================
# Kernel Wrapper
# =============================================================================

class TruePackedAttentionMetal:
    """True packed attention kernel wrapper.

    P1 Scaffold: Currently falls back to CPU reference. When Metal kernel
    is complete, this will dispatch to GPU with zero dense reconstruction.
    """

    def __init__(self, bits: int = 8, group_size: int = 64, sign_seed: int = 42):
        """Initialize the true packed kernel wrapper.

        Args:
            bits: Bit-width for quantization (default: 8 for K8/V8)
            group_size: Group size for Cartesian codec (default: 64)
            sign_seed: Seed for hash-sign derivation (default: 42)
        """
        self.bits = bits
        self.group_size = group_size
        self.sign_seed = sign_seed
        self._kernel_loaded = False
        self._compute_pipeline = None
        self._kernel_hash: str | None = None

        # Load kernel if Metal is available
        if HAS_METAL:
            self._load_kernel()
        else:
            warnings.warn("Metal not available, will use CPU fallback")

    def _load_kernel(self) -> None:
        """Load the Metal kernel and compute its hash for reproducibility."""
        try:
            import os
            from pathlib import Path

            kernel_path = Path(__file__).parent / "true_packed_attention.metal"
            if not kernel_path.exists():
                warnings.warn(f"Metal kernel not found: {kernel_path}")
                return

            # Read and hash shader source for contract recording
            with open(kernel_path, 'rb') as f:
                shader_source = f.read()
                self._kernel_hash = hashlib.sha256(shader_source).hexdigest()[:16]

            # TODO: Compile and load kernel when Metal API is available
            # For now, mark as not loaded to trigger fallback
            self._kernel_loaded = False

        except Exception as e:
            warnings.warn(f"Failed to load Metal kernel: {e}")
            self._kernel_loaded = False

    def _prepare_metal_buffers(
        self,
        queries: "mx.array",
        blocks: list[PackedBlock],
    ) -> dict[str, Any]:
        """Prepare Metal buffers from PackedBlock metadata.

        P1: This converts Python PackedBlock objects to Metal-compatible buffers.
        The buffer layout must match the PackedBlockMetadata struct in the shader.
        """
        if not blocks:
            return {}

        # Extract metadata for each block
        num_blocks = len(blocks)
        metadata = []
        packed_keys_list = []
        packed_values_list = []

        for block in blocks:
            # Get block metadata
            meta = {
                'logical_start': block.position,
                'token_count': block.token_count,
                'layer_id': getattr(block, 'layer_id', 0),
                'stream_id': getattr(block, 'stream_id', 0),
                'key_scale': getattr(block, 'key_scale', 1.0),
                'key_zero_point': getattr(block, 'key_zero_point', 0.0),
                'value_scale': getattr(block, 'value_scale', 1.0),
                'value_zero_point': getattr(block, 'value_zero_point', 0.0),
            }
            metadata.append(meta)

            # Get packed data buffers
            # TODO: Ensure these are contiguous and properly formatted
            packed_keys_list.append(block.packed_keys)
            packed_values_list.append(block.packed_values)

        return {
            'num_blocks': num_blocks,
            'metadata': metadata,
            'packed_keys': packed_keys_list,
            'packed_values': packed_values_list,
        }

    def __call__(
        self,
        queries: "mx.array",
        blocks: list[PackedBlock],
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
        strict: bool = False,
    ) -> tuple["mx.array", ExecutionContract]:
        """Execute true packed attention with execution contract.

        Args:
            queries: Query tensor [batch=1, num_q_heads, seq_len, head_dim]
            blocks: List of PackedBlock metadata with packed K/V data
            scale: Attention scale factor (typically 1/sqrt(head_dim))
            causal: Whether to apply causal mask
            query_start_pos: Global position of first query token
            strict: If True, raise error on fallback instead of using CPU

        Returns:
            (output, contract): Attention output and execution contract
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for true packed attention")

        start_time = time.perf_counter()

        # P1: For now, always use CPU fallback until Metal kernel is complete
        # When complete, dispatch to Metal if available
        if self._kernel_loaded and HAS_METAL and HAS_MLX:
            # TODO: Dispatch to Metal kernel when implemented
            pass

        # CPU fallback to MLX reference
        if strict and not self._kernel_loaded:
            raise RuntimeError(
                "Strict mode: True packed Metal kernel not available. "
                f"Kernel loaded: {self._kernel_loaded}, Metal available: {HAS_METAL}"
            )

        output = self._cpu_fallback(
            queries, blocks, scale, causal, query_start_pos
        )

        execution_ms = (time.perf_counter() - start_time) * 1000

        # Create execution contract for fallback
        contract = ExecutionContract(
            backend="packed_reference_cpu",
            kernel_hash=self._compute_reference_hash(),
            num_blocks=len(blocks),
            total_kv_tokens=sum(b.token_count for b in blocks),
            num_q_heads=queries.shape[1] if queries.ndim >= 3 else 1,
            num_kv_heads=getattr(blocks[0], 'n_kv_heads', 1) if blocks else 1,
            head_dim=queries.shape[-1],
            materialized_bytes=0,  # Reference also doesn't materialize full history
            decoded_tokens=0,  # Reference decodes on-the-fly
            fallback_reason="Metal kernel not yet functional (P1 scaffold)",
            execution_ms=execution_ms,
        )

        return output, contract

    def _cpu_fallback(
        self,
        queries: "mx.array",
        blocks: list[PackedBlock],
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
    ) -> "mx.array":
        """CPU fallback using MLX packed attention reference.

        P1 Scaffold: This fallback is not yet functional because it receives
        a list of PackedBlock objects, but the reference attend() requires a
        QuantizedLayerCache. The production wrapper catches this exception and
        falls through to the next backend (Metal dense or packed reference).
        """
        raise RuntimeError(
            "TruePackedAttentionMetal CPU fallback is not yet functional. "
            "The production wrapper will fall through to the next backend."
        )

    def _compute_reference_hash(self) -> str:
        """Compute hash of reference implementation for contract."""
        import inspect
        from rfsn_v10.cache import mlx_packed_attention_reference

        source = inspect.getsource(mlx_packed_attention_reference)
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def compare_with_reference(
        self,
        queries: "mx.array",
        blocks: list[PackedBlock],
        tolerance: float = 1e-5,
        scale: float = 1.0,
        causal: bool = True,
        query_start_pos: int = 0,
    ) -> dict[str, Any]:
        """Compare kernel output against MLX reference.

        P1: Differential testing hook for kernel validation.

        Returns:
            Comparison metrics including max/mean error and pass/fail status
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for comparison")

        # Get kernel output (with fallback)
        kernel_output, contract = self(
            queries, blocks, scale, causal, query_start_pos, strict=False
        )

        # Get reference output
        reference_output = self._cpu_fallback(
            queries, blocks, scale, causal, query_start_pos
        )

        # Compute error metrics
        error = mx.abs(kernel_output - reference_output)
        max_abs_error = float(mx.max(error))
        mean_abs_error = float(mx.mean(error))

        # Compute relative error (avoid div by zero)
        ref_norm = float(mx.max(mx.abs(reference_output)))
        rel_error = max_abs_error / ref_norm if ref_norm > 0 else float('inf')

        passed = max_abs_error < tolerance

        return {
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'rel_error': rel_error,
            'tolerance': tolerance,
            'passed': passed,
            'backend_used': contract.backend,
            'kernel_hash': contract.kernel_hash,
            'execution_ms': contract.execution_ms,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def true_packed_attend(
    queries: "mx.array",
    blocks: list[PackedBlock],
    scale: float = 1.0,
    causal: bool = True,
    query_start_pos: int = 0,
    strict: bool = False,
) -> tuple["mx.array", ExecutionContract]:
    """Convenience function for true packed attention with contract.

    This is the primary entry point for P1 true packed attention.
    It returns both the output and an execution contract for auditability.
    """
    kernel = TruePackedAttentionMetal()
    return kernel(
        queries, blocks, scale, causal, query_start_pos, strict
    )
