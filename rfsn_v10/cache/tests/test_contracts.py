"""Direct unit tests for rfsn_v10.cache.contracts.

Validates PackedBlock, CacheStats, AttentionScratch, and validate_block_positions
without requiring MLX or quantization.
"""
from __future__ import annotations

import pytest


class TestPackedBlock:
    def test_payload_bytes_with_none_codes(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=0,
            bits=8,
            group_size=64,
            n_values=0,
        )
        assert block.payload_bytes() == 0

    def test_validate_accepts_valid_block(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=64,
            bits=8,
            group_size=64,
            n_values=64,
            num_elements=64,
            format_version=3,
        )
        block.validate()  # should not raise

    def test_validate_rejects_invalid_bits(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=0,
            bits=9,
            group_size=64,
            n_values=0,
        )
        with pytest.raises(ValueError, match="bits"):
            block.validate()

    def test_validate_rejects_zero_group_size(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=0,
            bits=8,
            group_size=0,
            n_values=0,
        )
        with pytest.raises(ValueError, match="group_size"):
            block.validate()

    def test_validate_rejects_negative_vector_alignment(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=0,
            bits=8,
            group_size=64,
            n_values=0,
            vector_alignment=0,
        )
        with pytest.raises(ValueError, match="vector_alignment"):
            block.validate()

    def test_v3_defaults(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=0,
            bits=8,
            group_size=64,
            n_values=0,
        )
        assert block.format_version == 3
        assert block.vector_alignment == 64
        assert block.wht_applied is False
        assert block.sign_seed == 0


class TestValidateBlockPositions:
    def test_empty_list_passes(self) -> None:
        from rfsn_v10.cache.contracts import validate_block_positions

        validate_block_positions([])  # should not raise

    def test_single_block_passes(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock, validate_block_positions

        block = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=10,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=0,
        )
        validate_block_positions([block])

    def test_contiguous_blocks_pass(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock, validate_block_positions

        b1 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=0,
        )
        b2 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=32,
        )
        validate_block_positions([b1, b2])

    def test_gap_raises(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock, validate_block_positions

        b1 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=0,
        )
        b2 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=33,  # gap at 32
        )
        with pytest.raises(ValueError, match="gap"):
            validate_block_positions([b1, b2])

    def test_overlap_raises(self) -> None:
        from rfsn_v10.cache.contracts import PackedBlock, validate_block_positions

        b1 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=0,
        )
        b2 = PackedBlock(
            packed_codes=None,
            scales=None,
            token_count=32,
            bits=8,
            group_size=64,
            n_values=64,
            logical_start=31,  # overlaps with b1
        )
        with pytest.raises(ValueError, match="overlap"):
            validate_block_positions([b1, b2])


class TestCacheStats:
    def test_default_counters_are_zero(self) -> None:
        from rfsn_v10.cache.contracts import CacheStats

        stats = CacheStats()
        assert stats.tokens_encoded == 0
        assert stats.tokens_requantized == 0
        assert stats.sealed_blocks == 0
        assert stats.staged_tokens == 0
        assert stats.dense_residual_tokens == 0
        assert stats.payload_bytes == 0

    def test_custom_values(self) -> None:
        from rfsn_v10.cache.contracts import CacheStats

        stats = CacheStats(
            tokens_encoded=100,
            sealed_blocks=2,
            staged_tokens=12,
        )
        assert stats.tokens_encoded == 100
        assert stats.sealed_blocks == 2
        assert stats.staged_tokens == 12


class TestAttentionScratch:
    def test_defaults(self) -> None:
        from rfsn_v10.cache.contracts import AttentionScratch

        scratch = AttentionScratch()
        assert scratch.max_reconstructed_block_tokens == 0
        assert scratch.score_vector_bytes == 0
        assert scratch.output_accumulator_bytes == 0


class TestPackedBlockV4:
    def test_minimal_valid_block(self) -> None:
        import numpy as np

        from rfsn_v10.cache.contracts import (
            PackedBlockV4,
            PackingLayout,
            Preconditioner,
            ScaleLayout,
            TensorLayout,
        )

        block = PackedBlockV4(
            packed_codes=np.zeros((1, 2, 64, 16), dtype=np.uint32),
            scales=np.zeros((1, 2, 64, 1), dtype=np.float32),
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=Preconditioner.WHT64_HASH_SIGN_V1,
            batch_size=1,
            n_kv_heads=2,
            token_count=64,
            head_dim=64,
            logical_start=0,
            logical_end=64,
            bits=8,
            group_size=64,
            groups_per_vector=1,
            codes_per_word=4,
            words_per_vector=16,
            original_value_count=1 * 2 * 64 * 64,
            padded_value_count=1 * 2 * 64 * 64,
            original_dtype="float16",
            sign_seed=42,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=0,
            stream_id="K",
            codec_signature="rfsn-v4-8-64-wht",
        )
        block.validate()

    def test_rejects_non_v4_format(self) -> None:
        import numpy as np

        from rfsn_v10.cache.contracts import (
            PackedBlockV4,
            PackingLayout,
            Preconditioner,
            ScaleLayout,
            TensorLayout,
        )

        block = PackedBlockV4(
            packed_codes=np.zeros((1, 1, 1, 1), dtype=np.uint32),
            scales=np.zeros((1, 1, 1, 1), dtype=np.float32),
            format_version=3,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.GLOBAL_FLAT_V3,
            scale_layout=ScaleLayout.FLAT_GROUPS_V3,
            preconditioner=Preconditioner.NONE,
            batch_size=1,
            n_kv_heads=1,
            token_count=1,
            head_dim=64,
            logical_start=0,
            logical_end=1,
            bits=8,
            group_size=64,
            groups_per_vector=1,
            codes_per_word=6,
            words_per_vector=11,
            original_value_count=64,
            padded_value_count=64,
            original_dtype="float16",
            sign_seed=0,
            sign_algorithm="",
            layer_id=0,
            stream_id="",
            codec_signature="",
        )
        with pytest.raises(ValueError, match="unsupported PackedBlock format"):
            block.validate()

    def test_rejects_logical_range_mismatch(self) -> None:
        import numpy as np

        from rfsn_v10.cache.contracts import (
            PackedBlockV4,
            PackingLayout,
            Preconditioner,
            ScaleLayout,
            TensorLayout,
        )

        block = PackedBlockV4(
            packed_codes=np.zeros((1, 1, 64, 16), dtype=np.uint32),
            scales=np.zeros((1, 1, 64, 1), dtype=np.float32),
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=Preconditioner.WHT64_HASH_SIGN_V1,
            batch_size=1,
            n_kv_heads=1,
            token_count=64,
            head_dim=64,
            logical_start=0,
            logical_end=65,  # mismatch
            bits=8,
            group_size=64,
            groups_per_vector=1,
            codes_per_word=4,
            words_per_vector=16,
            original_value_count=1 * 1 * 64 * 64,
            padded_value_count=1 * 1 * 64 * 64,
            original_dtype="float16",
            sign_seed=42,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=0,
            stream_id="K",
            codec_signature="rfsn-v4-8-64-wht",
        )
        with pytest.raises(ValueError, match="logical range"):
            block.validate()

    def test_rejects_shape_mismatch(self) -> None:
        import numpy as np

        from rfsn_v10.cache.contracts import (
            PackedBlockV4,
            PackingLayout,
            Preconditioner,
            ScaleLayout,
            TensorLayout,
        )

        block = PackedBlockV4(
            packed_codes=np.zeros((1, 1, 64, 10), dtype=np.uint32),  # wrong words
            scales=np.zeros((1, 1, 64, 1), dtype=np.float32),
            format_version=4,
            tensor_layout=TensorLayout.BHTD,
            packing_layout=PackingLayout.VECTOR_ALIGNED_UINT32_V4,
            scale_layout=ScaleLayout.BHTG_V4,
            preconditioner=Preconditioner.WHT64_HASH_SIGN_V1,
            batch_size=1,
            n_kv_heads=1,
            token_count=64,
            head_dim=64,
            logical_start=0,
            logical_end=64,
            bits=8,
            group_size=64,
            groups_per_vector=1,
            codes_per_word=4,
            words_per_vector=16,
            original_value_count=1 * 1 * 64 * 64,
            padded_value_count=1 * 1 * 64 * 64,
            original_dtype="float16",
            sign_seed=42,
            sign_algorithm="murmur32-avalanche-v1",
            layer_id=0,
            stream_id="K",
            codec_signature="rfsn-v4-8-64-wht",
        )
        with pytest.raises(ValueError, match="packed_codes shape mismatch"):
            block.validate()
