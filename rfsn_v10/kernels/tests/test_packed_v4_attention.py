"""Differential tests for PackedV4AttentionKernel.

These tests verify that the canonical true-packed kernel produces
numerically identical output to the blockwise packed reference when
both operate on real ``PackedBlockV4`` blocks.

Requirements
------------
* MLX + Metal must be available.
* ``RFSN_ENABLE_TRUE_PACKED=1`` must be set in the environment.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.contracts import PackedBlock
from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
from rfsn_v10.cache.mlx_packed_attention_reference import attend as reference_attend
from rfsn_v10.cache.paged_arena import paged_view_from_blocks
from rfsn_v10.kernels.metal.packed_v4_attention import (
    HAS_TRUE_PACKED_KERNEL,
    PackedV4AttentionKernel,
    packed_v4_attention,
)

# P2: Hard-fail guard — if RFSN_ENABLE_TRUE_PACKED=1 is explicitly set,
# the kernel must be available.  Skipping would make CI green on a
# production kernel failure.
_env_enable = __import__("os").environ.get("RFSN_ENABLE_TRUE_PACKED", "")
if _env_enable == "1" and not HAS_TRUE_PACKED_KERNEL:
    pytest.fail(
        "RFSN_ENABLE_TRUE_PACKED=1 is set but HAS_TRUE_PACKED_KERNEL is False. "
        "The Metal self-test failed; this is a hard failure, not a skip.",
        pytrace=False,
    )

pytestmark = [
    pytest.mark.skipif(not HAS_MLX, reason="MLX not installed"),
    pytest.mark.skipif(
        not HAS_TRUE_PACKED_KERNEL and _env_enable != "1",
        reason="RFSN_ENABLE_TRUE_PACKED=1 not set or self-test failed",
    ),
]


def _make_layer_cache(key_blocks, value_blocks):
    """Build a minimal QuantizedLayerCache from paired blocks."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    cache = QuantizedLayerCache(
        key_codec=k_codec,
        value_codec=v_codec,
        staging_capacity=0,
        dense_residual_window=0,
        layer_id=0,
    )
    # Directly populate sealed blocks via the internal list
    cache._key_blocks = list(key_blocks)
    cache._value_blocks = list(value_blocks)
    return cache


def _encode_kv_tensors(k_bhtd, v_bhtd, *, logical_start=0, layer_id=0):
    """Encode dense K/V tensors into PackedBlockV4 lists."""
    k_codec = CartesianCodec(bits=8, group_size=64, sign_seed=42)
    v_codec = CartesianCodec(bits=8, group_size=64, sign_seed=42)
    k_block = k_codec.encode_bhtd(
        k_bhtd, logical_start=logical_start, layer_id=layer_id, stream_id="K"
    )
    v_block = v_codec.encode_bhtd(
        v_bhtd, logical_start=logical_start, layer_id=layer_id, stream_id="V"
    )
    return k_block, v_block


class TestPackedV4FormatCompatibility:
    """Verify the kernel actually reads PackedBlockV4 fields."""

    def test_rejects_sub_byte_bits(self):
        """Must gate out sub-byte variants in this release."""
        with pytest.raises(ValueError, match="only supports bits==8"):
            PackedV4AttentionKernel(bits=4)


class TestPackedV4AgainstReference:
    """Differential tests: PackedV4 kernel vs blockwise reference."""

    def _diff(self, a, b, rtol=1e-4, atol=1e-5):
        """Return max absolute and relative differences."""
        abs_diff = float(mx.max(mx.abs(a - b)).item())
        denom = float(mx.max(mx.abs(b)).item())
        rel_diff = abs_diff / (denom + 1e-8)
        return abs_diff, rel_diff

    def test_single_block_exact(self):
        """One block, causal, should match reference."""
        B, Hq, Hkv, T, D = 1, 4, 4, 16, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        cache = _make_layer_cache([k_block], [v_block])

        ref_out, _ = reference_attend(
            queries, cache, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=T,
        )

        kernel = PackedV4AttentionKernel()
        out, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )

        abs_diff, rel_diff = self._diff(out, ref_out)
        assert rel_diff < 1e-3, f"single block mismatch: rel={rel_diff}, abs={abs_diff}"
        assert contract.dense_kv_materialized_bytes == 0
        assert contract.decoded_dense_tokens == 0
        assert contract.packed_blocks_read == 1
        assert contract.packed_bytes_read > 0
        assert contract.backend == "true_packed_metal_v4_k8"

    def test_multiple_blocks_exact(self):
        """Multiple blocks should match reference."""
        B, Hq, Hkv, T, D = 1, 4, 4, 16, 64
        queries = mx.random.normal((B, Hq, 2, D), dtype=mx.float32)

        k_blocks = []
        v_blocks = []
        for i in range(2):
            keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            kb, vb = _encode_kv_tensors(keys, values, logical_start=i * T)
            k_blocks.append(kb)
            v_blocks.append(vb)

        cache = _make_layer_cache(k_blocks, v_blocks)
        ref_out, _ = reference_attend(
            queries, cache, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=2 * T,
        )

        kernel = PackedV4AttentionKernel()
        out, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks(k_blocks, v_blocks),
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=2 * T,
            layer_id=0,
        )

        abs_diff, rel_diff = self._diff(out, ref_out)
        assert rel_diff < 1e-3, f"multi-block mismatch: rel={rel_diff}, abs={abs_diff}"
        assert contract.num_key_blocks == 2
        assert contract.total_kv_tokens == 2 * T

    def test_gqa_exact(self):
        """GQA (Hq > Hkv) should match reference."""
        B, Hq, Hkv, T, D = 1, 8, 2, 16, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        cache = _make_layer_cache([k_block], [v_block])

        ref_out, _ = reference_attend(
            queries, cache, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=T,
        )

        kernel = PackedV4AttentionKernel()
        out, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )

        abs_diff, rel_diff = self._diff(out, ref_out)
        assert rel_diff < 1e-3, f"GQA mismatch: rel={rel_diff}, abs={abs_diff}"
        assert contract.num_q_heads == Hq
        assert contract.num_kv_heads == Hkv

    def test_causal_masking(self):
        """Causal masking must zero out future positions correctly."""
        B, Hq, Hkv, T, D = 1, 2, 2, 8, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        cache = _make_layer_cache([k_block], [v_block])

        ref_out, _ = reference_attend(
            queries, cache, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=T,
        )

        kernel = PackedV4AttentionKernel()
        out, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )

        abs_diff, rel_diff = self._diff(out, ref_out)
        assert rel_diff < 1e-3, f"causal mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_non_causal(self):
        """Non-causal mode should attend to all tokens."""
        B, Hq, Hkv, T, D = 1, 2, 2, 8, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        cache = _make_layer_cache([k_block], [v_block])

        ref_out, _ = reference_attend(
            queries, cache, scale=1.0 / np.sqrt(D), causal=False
        )

        kernel = PackedV4AttentionKernel()
        out, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0 / np.sqrt(D),
            causal=False,
            query_start_pos=0,
            layer_id=0,
        )

        abs_diff, rel_diff = self._diff(out, ref_out)
        assert rel_diff < 1e-3, f"non-causal mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_contract_zero_materialization(self):
        """Execution contract must assert zero materialisation."""
        B, Hq, Hkv, T, D = 1, 2, 2, 8, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)

        kernel = PackedV4AttentionKernel()
        _, _, _, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0,
            causal=True,
            query_start_pos=T,
            strict=True,
            layer_id=0,
        )

        passed, violations = contract.validate_invariant()
        assert passed, f"Invariant violated: {violations}"

    def test_strict_mode_rejects_batch_size_gt_1(self):
        """Current kernel only supports batch_size=1."""
        queries = mx.zeros((2, 2, 1, 64))
        codec = CartesianCodec(bits=8, group_size=64)
        block = codec.encode_bhtd(
            mx.random.normal((1, 1, 2, 64), dtype=mx.float32),
            logical_start=0, layer_id=0, stream_id="K"
        )

        kernel = PackedV4AttentionKernel()
        with pytest.raises(RuntimeError, match="batch_size=1"):
            kernel(
                queries=queries,
                paged_kv=paged_view_from_blocks([block], [block]),
                scale=1.0,
                layer_id=0,
            )


class TestConvenienceWrapper:
    def test_packed_v4_attention_function(self):
        """The top-level convenience function must work."""
        B, Hq, Hkv, T, D = 1, 2, 2, 8, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)

        out, _, _, contract = packed_v4_attention(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0,
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )

        assert out.shape == queries.shape
        assert contract.backend == "true_packed_metal_v4_k8"


class TestPackedV4SoftmaxStats:
    """Verify the kernel exports correct softmax statistics for region merging."""

    def test_returns_valid_stats(self):
        """Kernel must return running_max and running_sum with correct shapes."""
        B, Hq, Hkv, T, D = 1, 4, 4, 16, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        kernel = PackedV4AttentionKernel()
        out, max_arr, sum_arr, contract = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=1.0 / np.sqrt(D),
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )
        assert max_arr.shape == (Hq, 1)
        assert sum_arr.shape == (Hq, 1)
        assert float(mx.min(sum_arr).item()) > 0, "running_sum must be positive"
        assert contract.backend == "true_packed_metal_v4_k8"

    def test_merge_with_dense_region(self):
        """Merging packed + dense region must match full dense oracle."""
        B, Hq, Hkv, D = 1, 4, 4, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        scale = 1.0 / np.sqrt(D)

        # Sealed packed block (first 8 tokens)
        keys_packed = mx.random.normal((B, Hkv, 8, D), dtype=mx.float32)
        values_packed = mx.random.normal((B, Hkv, 8, D), dtype=mx.float32)
        k_block, v_block = _encode_kv_tensors(keys_packed, values_packed, logical_start=0)

        # Staging (next 8 tokens)
        keys_staging = mx.random.normal((B, Hkv, 8, D), dtype=mx.float32)
        values_staging = mx.random.normal((B, Hkv, 8, D), dtype=mx.float32)

        # Full dense oracle over all 16 tokens
        all_keys = mx.concatenate([keys_packed, keys_staging], axis=2)
        all_values = mx.concatenate([values_packed, values_staging], axis=2)
        scores = (queries @ all_keys.transpose(0, 1, 3, 2)) * scale
        max_score = mx.max(scores, axis=-1, keepdims=True)
        exp_scores = mx.exp(scores - max_score)
        sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
        oracle = (exp_scores @ all_values) / sum_exp

        # Packed region via kernel
        kernel = PackedV4AttentionKernel()
        packed_out, packed_max, packed_sum, _ = kernel(
            queries=queries,
            paged_kv=paged_view_from_blocks([k_block], [v_block]),
            scale=scale,
            causal=True,
            query_start_pos=16,
            layer_id=0,
        )

        # Staging region via dense helper
        from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
            _dense_attention_with_stats,
            _merge_attention_regions,
        )
        stage_out, stage_max, stage_sum = _dense_attention_with_stats(
            queries,
            keys_staging,
            values_staging,
            scale,
            query_start_pos=16,
            kv_start_pos=8,
        )

        # Merge
        merged = _merge_attention_regions([
            (packed_out, packed_max, packed_sum),
            (stage_out, stage_max, stage_sum),
        ])

        abs_diff = float(mx.max(mx.abs(merged - oracle)).item())
        rel_diff = abs_diff / (float(mx.max(mx.abs(oracle)).item()) + 1e-8)
        # Tolerance relaxed to 5e-2 because the packed region introduces
        # quantization error; the test verifies the merge math, not bit-exactness.
        assert rel_diff < 5e-2, f"merge mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_empty_packed_with_staging_matches_oracle(self):
        """When no packed blocks exist, staging-only must match dense oracle."""
        B, Hq, Hkv, T, D = 1, 4, 4, 8, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        scale = 1.0 / np.sqrt(D)

        keys_staging = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values_staging = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        # Oracle
        scores = (queries @ keys_staging.transpose(0, 1, 3, 2)) * scale
        max_score = mx.max(scores, axis=-1, keepdims=True)
        exp_scores = mx.exp(scores - max_score)
        sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
        oracle = (exp_scores @ values_staging) / sum_exp

        from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
            _dense_attention_with_stats,
            _merge_attention_regions,
        )
        stage_out, stage_max, stage_sum = _dense_attention_with_stats(
            queries,
            keys_staging,
            values_staging,
            scale,
            query_start_pos=T,
            kv_start_pos=0,
        )
        merged = _merge_attention_regions([
            (stage_out, stage_max, stage_sum),
        ])

        abs_diff = float(mx.max(mx.abs(merged - oracle)).item())
        rel_diff = abs_diff / (float(mx.max(mx.abs(oracle)).item()) + 1e-8)
        assert rel_diff < 1e-3, f"staging-only mismatch: rel={rel_diff}, abs={abs_diff}"


class TestPackedV4CompilationStability:
    """Verify that growing context does not trigger new kernel compiles."""

    def test_kernel_compilation_count_flat(self):
        """Same compiled kernel for 1 page and 2 pages."""
        B, Hq, Hkv, T, D = 1, 4, 4, 16, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)

        kernel = PackedV4AttentionKernel()
        assert kernel.compilation_count == 0

        k1, v1 = _encode_kv_tensors(
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
        )
        kernel(
            queries,
            paged_view_from_blocks([k1], [v1], max_pages=8),
            scale=1.0,
            causal=True,
            query_start_pos=T,
            layer_id=0,
        )
        assert kernel.compilation_count == 1

        k2, v2 = _encode_kv_tensors(
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            logical_start=T,
        )
        kernel(
            queries,
            paged_view_from_blocks([k1, k2], [v1, v2], max_pages=8),
            scale=1.0,
            causal=True,
            query_start_pos=2 * T,
            layer_id=0,
        )
        assert kernel.compilation_count == 1, (
            "Kernel recompiled when context grew — template must be stable"
        )

    def test_no_history_copy_invariant(self):
        """Arena must report zero history recopy bytes."""
        from rfsn_v10.cache.paged_arena import PagedKVArena

        arena = PagedKVArena(
            max_pages=8,
            page_tokens=16,
            n_kv_heads=2,
            k_words_per_vector=16,
            v_words_per_vector=16,
            k_groups_per_vector=1,
            v_groups_per_vector=1,
        )
        for i in range(4):
            kb = _make_block_for_arena(16, i * 16)
            vb = _make_block_for_arena(16, i * 16)
            arena.append(kb, vb)

        assert arena.history_recopy_bytes == 0
        assert arena.page_write_bytes > 0


def _make_block_for_arena(token_count: int, logical_start: int) -> PackedBlock:
    import mlx.core as mx
    return PackedBlock(
        packed_codes=mx.zeros((1, 2, token_count, 16), dtype=mx.uint32),
        scales=mx.zeros((1, 2, token_count, 1), dtype=mx.float16),
        token_count=token_count,
        bits=8,
        group_size=64,
        n_values=token_count * 16,
        logical_start=logical_start,
        head_dim=64,
        num_elements=token_count * 16,
        batch_size=1,
        n_kv_heads=2,
    )


class TestPackedV4TiledKernel:
    """Differential tests for the tiled KV path against the scalar path."""

    def _diff(self, a, b, rtol=1e-4, atol=1e-5):
        abs_diff = float(mx.max(mx.abs(a - b)).item())
        denom = float(mx.max(mx.abs(b)).item())
        rel_diff = abs_diff / (denom + 1e-8)
        return abs_diff, rel_diff

    def test_tiled_matches_scalar_single_block(self):
        """Tiled path must match scalar path for a single 32-token block."""
        B, Hq, Hkv, T, D = 1, 4, 4, 32, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)
        keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
        values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)

        k_block, v_block = _encode_kv_tensors(keys, values)
        paged = paged_view_from_blocks([k_block], [v_block], max_pages=4)

        scalar_kernel = PackedV4AttentionKernel()
        tiled_kernel = PackedV4AttentionKernel(kv_tile_size=16)

        scalar_out, _, _, _ = scalar_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=T, layer_id=0,
        )
        tiled_out, _, _, _ = tiled_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=T, layer_id=0,
        )

        abs_diff, rel_diff = self._diff(scalar_out, tiled_out)
        assert rel_diff < 1e-3, f"tiled vs scalar mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_tiled_matches_scalar_gqa(self):
        """Tiled path must match scalar path under GQA (Hq > Hkv)."""
        B, Hq, Hkv, T, D = 1, 8, 2, 32, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)

        k_blocks = []
        v_blocks = []
        for i in range(2):
            keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            kb, vb = _encode_kv_tensors(keys, values, logical_start=i * T)
            k_blocks.append(kb)
            v_blocks.append(vb)

        paged = paged_view_from_blocks(k_blocks, v_blocks, max_pages=4)

        scalar_kernel = PackedV4AttentionKernel()
        tiled_kernel = PackedV4AttentionKernel(kv_tile_size=16)

        scalar_out, _, _, _ = scalar_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=2 * T, layer_id=0,
        )
        tiled_out, _, _, _ = tiled_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=2 * T, layer_id=0,
        )

        abs_diff, rel_diff = self._diff(scalar_out, tiled_out)
        assert rel_diff < 1e-3, f"tiled GQA mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_tiled_matches_scalar_multi_block(self):
        """Tiled path must match scalar path for two 32-token blocks."""
        B, Hq, Hkv, T, D = 1, 4, 4, 32, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)

        k_blocks = []
        v_blocks = []
        for i in range(2):
            keys = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            values = mx.random.normal((B, Hkv, T, D), dtype=mx.float32)
            kb, vb = _encode_kv_tensors(keys, values, logical_start=i * T)
            k_blocks.append(kb)
            v_blocks.append(vb)

        paged = paged_view_from_blocks(k_blocks, v_blocks, max_pages=4)

        scalar_kernel = PackedV4AttentionKernel()
        tiled_kernel = PackedV4AttentionKernel(kv_tile_size=24)

        scalar_out, _, _, _ = scalar_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=2 * T, layer_id=0,
        )
        tiled_out, _, _, _ = tiled_kernel(
            queries, paged, scale=1.0 / np.sqrt(D), causal=True,
            query_start_pos=2 * T, layer_id=0,
        )

        abs_diff, rel_diff = self._diff(scalar_out, tiled_out)
        assert rel_diff < 1e-3, f"tiled vs scalar mismatch: rel={rel_diff}, abs={abs_diff}"

    def test_tiled_compilation_count_stable(self):
        """Tiled kernel compiles once per unique (geometry, MAX_KV_TILES).

        MAX_KV_TILES is derived from arena capacity (max_pages * page_tokens)
        and baked into the template.  active_kv_tiles is passed at runtime.
        Changing the active tile count without exceeding the cached max does
        NOT recompile; only a larger arena capacity triggers a new compile.
        """
        B, Hq, Hkv, T, D = 1, 4, 4, 32, 64
        queries = mx.random.normal((B, Hq, 1, D), dtype=mx.float32)

        tiled_kernel = PackedV4AttentionKernel(kv_tile_size=16)
        assert tiled_kernel.compilation_count == 0

        # 32 tokens / tile_size 16 = 2 active tiles.
        # Arena: max_pages=8, page_tokens=32 → capacity 256 tokens → MAX_KV_TILES=16.
        k1, v1 = _encode_kv_tensors(
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
        )
        tiled_kernel(
            queries,
            paged_view_from_blocks([k1], [v1], max_pages=8),
            scale=1.0, causal=True, query_start_pos=T, layer_id=0,
        )
        assert tiled_kernel.compilation_count == 1

        # Another 32-token block: 64 tokens / tile_size 16 = 4 active tiles.
        # Same arena capacity → MAX_KV_TILES still 16 → reuse cached kernel.
        k2, v2 = _encode_kv_tensors(
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            mx.random.normal((B, Hkv, T, D), dtype=mx.float32),
            logical_start=T,
        )
        tiled_kernel(
            queries,
            paged_view_from_blocks([k1, k2], [v1, v2], max_pages=8),
            scale=1.0, causal=True, query_start_pos=2 * T, layer_id=0,
        )
        # Only 1 compile because MAX_KV_TILES (16) already covers 4 active tiles.
        assert tiled_kernel.compilation_count == 1

        # Re-run with the SAME 2 blocks (4 active tiles) — must reuse kernel
        tiled_kernel(
            queries,
            paged_view_from_blocks([k1, k2], [v1, v2], max_pages=8),
            scale=1.0, causal=True, query_start_pos=2 * T, layer_id=0,
        )
        assert tiled_kernel.compilation_count == 1, (
            "Re-ran same tile count but kernel recompiled"
        )

        # Larger arena capacity forces MAX_KV_TILES to grow → new compile.
        # max_pages=16, page_tokens=32 → capacity 512 tokens → MAX_KV_TILES=32.
        tiled_kernel(
            queries,
            paged_view_from_blocks([k1, k2], [v1, v2], max_pages=16),
            scale=1.0, causal=True, query_start_pos=2 * T, layer_id=0,
        )
        assert tiled_kernel.compilation_count == 2, (
            "Larger arena should trigger recompile with bigger MAX_KV_TILES"
        )
