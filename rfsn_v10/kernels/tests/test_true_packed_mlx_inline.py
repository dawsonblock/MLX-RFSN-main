"""Tests for MLX inline Metal true packed kernel.

These tests verify that the kernel:
1. Compiles successfully via MLX's metal_kernel API
2. Executes without errors on synthetic data
3. Produces valid output shapes
4. Records zero materialization in execution contract
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from rfsn_v10.kernels.metal.true_packed_mlx_inline import (
    TruePackedMLXInline,
    true_packed_attention_mlx,
    ExecutionContract,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


class MockBlock:
    """Mock PackedBlock for testing."""

    def __init__(
        self,
        position: int,
        token_count: int,
        head_dim: int = 64,
        num_kv_heads: int = 4,
        bits: int = 8,
        layer_id: int = 0,
        stream_id: int = 0,
    ):
        self.position = position
        self.token_count = token_count
        self.layer_id = layer_id
        self.stream_id = stream_id
        self.key_scale = 1.0
        self.key_zero_point = 0.0
        self.value_scale = 1.0
        self.value_zero_point = 0.0

        # Calculate packed size
        total_elements = num_kv_heads * token_count * head_dim
        packed_bytes = (total_elements * bits + 7) // 8

        # Random packed data (uint8)
        self.packed_keys = np.random.randint(0, 256, size=packed_bytes, dtype=np.uint8)
        self.packed_values = np.random.randint(0, 256, size=packed_bytes, dtype=np.uint8)


class TestMLXInlineKernel:
    """Test MLX inline Metal kernel compilation and execution."""

    def test_kernel_compiles(self):
        """Test that the kernel compiles successfully."""
        kernel = TruePackedMLXInline(bits=8)
        assert kernel._kernel_hash is not None
        assert len(kernel._kernel_hash) == 16

    def test_execution_synthetic_data(self):
        """Test kernel execution with synthetic data."""
        kernel = TruePackedMLXInline(bits=8)

        # Create test queries
        B, Hq, Lq, D = 1, 4, 2, 64
        queries = mx.random.normal((B, Hq, Lq, D), dtype=mx.float32)

        # Create test blocks
        blocks = [
            MockBlock(position=0, token_count=32, head_dim=D, num_kv_heads=Hq),
            MockBlock(position=32, token_count=32, head_dim=D, num_kv_heads=Hq),
        ]

        # Execute kernel
        output, contract = kernel(
            queries=queries,
            blocks=blocks,
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=0,
        )

        # Verify output shape
        assert output.shape == queries.shape

        # Verify contract
        assert contract.backend == "true_packed_metal_mlx_inline"
        assert contract.num_blocks == 2
        assert contract.total_kv_tokens == 64
        assert contract.materialized_bytes == 0
        assert contract.decoded_tokens == 0

    def test_execution_contract_zero_materialization(self):
        """Verify contract shows zero materialization."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.zeros((1, 2, 1, 32))
        blocks = [MockBlock(position=0, token_count=16, head_dim=32, num_kv_heads=2)]

        _, contract = kernel(queries, blocks, scale=1.0)

        passed, violations = contract.validate_invariant()
        assert passed, f"Invariant violations: {violations}"
        assert len(violations) == 0

    def test_strict_mode_passes(self):
        """Test that strict mode passes with zero materialization."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.zeros((1, 2, 1, 32))
        blocks = [MockBlock(position=0, token_count=16, head_dim=32, num_kv_heads=2)]

        # Should not raise
        output, contract = kernel(
            queries, blocks, scale=1.0, strict=True
        )
        assert output.shape == (1, 2, 1, 32)

    def test_various_bit_widths(self):
        """Test kernel with different bit widths."""
        for bits in [4, 6, 8]:
            kernel = TruePackedMLXInline(bits=bits)

            queries = mx.zeros((1, 2, 1, 32))
            blocks = [MockBlock(position=0, token_count=8, head_dim=32, num_kv_heads=2, bits=bits)]

            output, contract = kernel(queries, blocks, scale=1.0)
            assert output.shape == (1, 2, 1, 32)
            assert contract.bits == bits

    def test_single_block(self):
        """Test with a single block."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.random.normal((1, 4, 4, 64), dtype=mx.float32)
        blocks = [MockBlock(position=0, token_count=64, head_dim=64, num_kv_heads=4)]

        output, contract = kernel(queries, blocks, scale=1.0 / 8.0)
        assert output.shape == (1, 4, 4, 64)
        assert contract.num_blocks == 1

    def test_multiple_blocks(self):
        """Test with multiple blocks."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.random.normal((1, 4, 2, 64), dtype=mx.float32)
        blocks = [
            MockBlock(position=0, token_count=32, head_dim=64, num_kv_heads=4),
            MockBlock(position=32, token_count=32, head_dim=64, num_kv_heads=4),
            MockBlock(position=64, token_count=16, head_dim=64, num_kv_heads=4),
        ]

        output, contract = kernel(queries, blocks, scale=1.0 / 8.0, causal=True)
        assert output.shape == (1, 4, 2, 64)
        assert contract.num_blocks == 3

    def test_performance_timing(self):
        """Test that execution time is recorded."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.zeros((1, 2, 1, 32))
        blocks = [MockBlock(position=0, token_count=16, head_dim=32, num_kv_heads=2)]

        _, contract = kernel(queries, blocks, scale=1.0)
        assert contract.execution_ms >= 0.0
        assert isinstance(contract.execution_ms, float)

    def test_causal_vs_non_causal(self):
        """Test causal and non-causal modes."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.random.normal((1, 2, 2, 32), dtype=mx.float32)
        blocks = [MockBlock(position=0, token_count=32, head_dim=32, num_kv_heads=2)]

        # Causal
        out_causal, _ = kernel(queries, blocks, scale=1.0, causal=True)
        assert out_causal.shape == (1, 2, 2, 32)

        # Non-causal
        out_full, _ = kernel(queries, blocks, scale=1.0, causal=False)
        assert out_full.shape == (1, 2, 2, 32)

    def test_convenience_function(self):
        """Test the convenience function."""
        queries = mx.zeros((1, 2, 1, 32))
        blocks = [MockBlock(position=0, token_count=16, head_dim=32, num_kv_heads=2)]

        output, contract = true_packed_attention_mlx(
            queries=queries,
            blocks=blocks,
            scale=1.0,
            bits=8,
        )

        assert output.shape == (1, 2, 1, 32)
        assert contract.bits == 8


class TestExecutionContract:
    """Test execution contract validation."""

    def test_contract_passes_for_metal(self):
        """Test that Metal contract passes validation."""
        contract = ExecutionContract(
            backend="true_packed_metal_mlx_inline",
            kernel_hash="abc123",
            num_blocks=2,
            total_kv_tokens=64,
            num_q_heads=4,
            num_kv_heads=4,
            head_dim=64,
            bits=8,
            materialized_bytes=0,
            decoded_tokens=0,
        )

        passed, violations = contract.validate_invariant()
        assert passed
        assert len(violations) == 0

    def test_contract_fails_for_fallback(self):
        """Test that fallback contract fails validation."""
        contract = ExecutionContract(
            backend="packed_reference_cpu",
            kernel_hash="ref",
            num_blocks=1,
            total_kv_tokens=32,
            num_q_heads=4,
            num_kv_heads=4,
            head_dim=64,
            bits=8,
            materialized_bytes=1024,
            decoded_tokens=32,
        )

        passed, violations = contract.validate_invariant()
        assert not passed
        assert len(violations) == 3

    def test_contract_fails_for_wrong_backend(self):
        """Test that non-Metal backend fails."""
        contract = ExecutionContract(
            backend="dense_reconstruction",
            kernel_hash="bad",
            num_blocks=1,
            total_kv_tokens=32,
            num_q_heads=4,
            num_kv_heads=4,
            head_dim=64,
            bits=8,
            materialized_bytes=0,
            decoded_tokens=0,
        )

        passed, violations = contract.validate_invariant()
        assert not passed
        assert any("backend" in v for v in violations)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_blocks(self):
        """Test with empty blocks list."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.zeros((1, 2, 1, 32))
        blocks = []

        # Should handle empty blocks gracefully
        output, contract = kernel(queries, blocks, scale=1.0)
        assert output.shape == (1, 2, 1, 32)
        assert contract.total_kv_tokens == 0

    def test_large_head_dim(self):
        """Test with larger head dimension."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.random.normal((1, 4, 2, 128), dtype=mx.float32)
        blocks = [MockBlock(position=0, token_count=16, head_dim=128, num_kv_heads=4)]

        output, contract = kernel(queries, blocks, scale=1.0 / 11.3)
        assert output.shape == (1, 4, 2, 128)

    def test_different_layer_stream_ids(self):
        """Test with different layer and stream IDs."""
        kernel = TruePackedMLXInline(bits=8)

        queries = mx.zeros((1, 2, 1, 32))
        block = MockBlock(position=0, token_count=16, head_dim=32, num_kv_heads=2)
        block.layer_id = 5
        block.stream_id = 3

        output, contract = kernel(queries, [block], scale=1.0)
        assert output.shape == (1, 2, 1, 32)
