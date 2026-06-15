"""Differential tests for true packed Metal kernel against MLX reference.

P1 Testing Framework: These tests validate that the Metal kernel produces
numerically identical results to the MLX packed attention reference.

Test levels:
- Level 1: Unit tests for individual shader functions (decode, dot product)
- Level 2: Integration tests for full attention computation
- Level 3: Fidelity tests comparing against reference across random inputs
- Level 4: Stress tests for edge cases (causal boundaries, GQA, large contexts)
"""
from __future__ import annotations

import pytest
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Skip all tests if MLX not available
pytestmark = [
    pytest.mark.skipif(not HAS_MLX, reason="MLX not installed"),
    pytest.mark.mlx,
]


# =============================================================================
# Level 1: Unit Tests (Shader Components)
# =============================================================================

class TestCartesianDecode:
    """Test Cartesian codec decode functions."""

    def test_uniform_dequantize_8bit(self):
        """Test 8-bit uniform dequantization."""
        # P1: This will test the Metal decode function when available
        # For now, verify Python reference implementation
        from rfsn_v10.cache.cartesian_codec import _dequantize_uniform

        # Test 8-bit dequantization round-trip
        original = np.array([0.5, -0.5, 1.0, -1.0, 0.0], dtype=np.float32)
        quantized = _dequantize_uniform(
            np.array([191, 64, 255, 0, 127], dtype=np.uint8),
            8, 1.0, 0.0
        )

        # Check approximate reconstruction
        np.testing.assert_allclose(quantized, original, rtol=0.1, atol=0.1)


class TestVectorOperations:
    """Test vector dot product and accumulation."""

    def test_vector_dot_accuracy(self):
        """Test vector dot product matches numpy."""
        if not HAS_MLX:
            pytest.skip("MLX not available")

        # Test various dimensions
        for dim in [32, 64, 128]:
            a = mx.random.normal((dim,)).astype(mx.float32)
            b = mx.random.normal((dim,)).astype(mx.float32)

            mlx_result = mx.sum(a * b)
            np_result = np.dot(np.array(a), np.array(b))

            assert abs(float(mlx_result) - np_result) < 1e-5


# =============================================================================
# Level 2: Integration Tests
# =============================================================================

class TestBufferPreparation:
    """Test Metal buffer preparation from PackedBlock."""

    def test_metadata_extraction(self):
        """Test extraction of block metadata for Metal."""
        # P1: Test that PackedBlockMetadata is correctly extracted
        # This validates the Python-to-Metal data path
        pass


# =============================================================================
# Level 3: Fidelity Tests (vs Reference)
# =============================================================================

class TestAgainstReference:
    """Differential tests comparing kernel against MLX reference."""

    TOLERANCE = 1e-5  # Maximum acceptable difference

    @pytest.fixture
    def simple_attention_case(self):
        """Generate a simple test case for attention."""
        if not HAS_MLX:
            pytest.skip("MLX not available")

        B, Hq, Hkv, Lq, Lkv, D = 1, 4, 2, 4, 16, 32

        queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
        keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        scale = D ** -0.5

        return {
            'queries': queries,
            'keys': keys,
            'values': values,
            'scale': scale,
            'B': B, 'Hq': Hq, 'Hkv': Hkv, 'Lq': Lq, 'Lkv': Lkv, 'D': D
        }

    def test_metal_dense_matches_reference(self):
        """Test Metal dense attention matches MLX reference."""
        # P1: This tests the existing metal_dense_attention_over_reconstructed_kv
        # to establish a baseline for comparison
        if not HAS_MLX:
            pytest.skip("MLX not available")

        from rfsn_v10.kernels.metal.packed_attention_metal import (
            metal_dense_attention_over_reconstructed_kv,
            metal_available,
        )

        if not metal_available():
            pytest.skip("Metal not available")

        # Simple test case
        B, Hq, Hkv, Lq, Lkv, D = 1, 2, 1, 4, 16, 32
        queries = mx.random.normal((B, Hq, Lq, D)).astype(mx.float32)
        keys = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        values = mx.random.normal((B, Hkv, Lkv, D)).astype(mx.float32)
        scale = D ** -0.5

        # Metal output
        metal_output = metal_dense_attention_over_reconstructed_kv(
            queries, keys, values, scale, causal=True
        )

        # Reference: MLX built-in attention with GQA repeat
        keys_rep = mx.repeat(keys, Hq // Hkv, axis=1)
        values_rep = mx.repeat(values, Hq // Hkv, axis=1)

        # Manual attention computation
        scores = mx.matmul(queries, mx.transpose(keys_rep, (0, 1, 3, 2))) * scale

        # Causal mask
        q_pos = mx.arange(Lq)[:, None]
        k_pos = mx.arange(Lkv)[None, :]
        mask = q_pos >= k_pos
        scores = mx.where(mask, scores, -1e9)

        # Softmax
        weights = mx.softmax(scores, axis=-1)

        # Attention output
        ref_output = mx.matmul(weights, values_rep)

        # Compare
        max_error = float(mx.max(mx.abs(metal_output - ref_output)))
        mean_error = float(mx.mean(mx.abs(metal_output - ref_output)))

        assert max_error < self.TOLERANCE, (
            f"Metal dense attention mismatch: max_error={max_error}, "
            f"mean_error={mean_error}, tolerance={self.TOLERANCE}"
        )

    def test_true_packed_matches_reference(self):
        """Test true packed kernel matches MLX packed reference."""
        # P1: When true packed kernel is complete, this validates it
        # Currently skips since kernel is scaffold
        pytest.skip("True packed kernel not yet functional (P1 scaffold)")


# =============================================================================
# Level 4: Stress Tests
# =============================================================================

class TestEdgeCases:
    """Stress tests for edge cases and boundary conditions."""

    def test_causal_boundary(self):
        """Test attention at exact causal boundary positions."""
        # P1: Ensure no off-by-one errors in causal masking
        pass

    def test_gqa_various_ratios(self):
        """Test GQA with various query-to-KV head ratios."""
        # P1: Test 1:1, 2:1, 4:1, 8:1 ratios
        pass

    def test_block_boundaries(self):
        """Test attention across block boundaries."""
        # P1: Critical for packed attention correctness
        pass

    def test_large_context(self):
        """Test with context lengths up to 8K tokens."""
        # P1: Performance and correctness at scale
        pass


# =============================================================================
# Execution Contract Tests
# =============================================================================

class TestExecutionContract:
    """Test execution contract recording and validation."""

    def test_contract_creation(self):
        """Test that execution contracts are properly created."""
        from rfsn_v10.kernels.metal.true_packed_wrapper import ExecutionContract

        contract = ExecutionContract(
            backend="packed_reference_cpu",
            kernel_hash="abc123",
            num_blocks=4,
            total_kv_tokens=256,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
        )

        assert contract.backend == "packed_reference_cpu"
        assert contract.materialized_bytes == 0
        assert contract.decoded_tokens == 0

    def test_invariant_validation_pass(self):
        """Test invariant validation passes for valid contract."""
        from rfsn_v10.kernels.metal.true_packed_wrapper import ExecutionContract

        contract = ExecutionContract(
            backend="true_packed_metal",
            kernel_hash="abc123",
            num_blocks=4,
            total_kv_tokens=256,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            materialized_bytes=0,
            decoded_tokens=0,
        )

        passed, violations = contract.validate_invariant()
        assert passed
        assert len(violations) == 0

    def test_invariant_validation_fail(self):
        """Test invariant validation fails when materialization occurs."""
        from rfsn_v10.kernels.metal.true_packed_wrapper import ExecutionContract

        contract = ExecutionContract(
            backend="metal_dense_reconstruction_violates_invariant",
            kernel_hash="abc123",
            num_blocks=4,
            total_kv_tokens=256,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            materialized_bytes=1024,
            decoded_tokens=256,
        )

        passed, violations = contract.validate_invariant()
        assert not passed
        assert len(violations) == 3  # backend, materialized_bytes, decoded_tokens


# =============================================================================
# Kernel Comparison Utilities
# =============================================================================

def compare_attention_outputs(
    output_a: "mx.array",
    output_b: "mx.array",
    tolerance: float = 1e-5
) -> dict[str, float]:
    """Compare two attention outputs and return metrics.

    Args:
        output_a: First output tensor
        output_b: Second output tensor
        tolerance: Maximum acceptable absolute difference

    Returns:
        Dictionary with comparison metrics
    """
    error = mx.abs(output_a - output_b)

    return {
        'max_abs_error': float(mx.max(error)),
        'mean_abs_error': float(mx.mean(error)),
        'rmse': float(mx.sqrt(mx.mean(error ** 2))),
        'tolerance': tolerance,
        'passed': float(mx.max(error)) < tolerance,
    }
