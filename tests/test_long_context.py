#!/usr/bin/env python3
"""
RFSN v10 - Long Context Smoke Tests.
Validates the runtime with progressively longer sequences.
Does NOT require loading a real LLM — uses synthetic tensors.
"""
from __future__ import annotations

import math
import tempfile

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.attention import AdaptiveBlockSparseAttention
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10 import RFSNRuntime


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


@pytest.mark.parametrize("seq_len", [512, 2048, 4096])
def test_long_context_runtime(seq_len: int):
    """Validate runtime handles long sequences without crashing."""
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=8, v_bits=3, use_incoherent=True,
            max_memory_gb=1.0, max_pinned_memory_gb=0.5, cache_dir=td,
        )
        runtime = RFSNRuntime(
            kv_manager=mgr, model_id="smoke_test", block_size=64,
            audit_mode=True, top_k_ratio=0.5,
        )

        mx.random.seed(42)
        shape_kv = (1, 8, seq_len, 64)
        shape_q = (1, 8, 1, 64)

        q = mx.random.normal(shape_q)
        k = mx.random.normal(shape_kv)
        v = mx.random.normal(shape_kv)

        output, info = runtime.execute_decode_step(
            skill_pattern="long_ctx", layer_id="l0", batch_id="b1",
            queries=q, keys=k, values=v, top_k_ratio=0.5,
        )

        assert output.shape == shape_q
        assert info["sparse_enabled"] is False
        assert info["sparse_gate_reason"] == "disabled_by_default"
        assert info["sparse_success"] is False
        assert info["total_latency_ms"] > 0


def test_long_context_dense_baseline_comparison():
    """Compare sparse vs dense output quality at various sequence lengths."""
    mx.random.seed(42)
    for seq_len in [512, 2048, 4096]:
        shape_q = (1, 8, 1, 64)
        shape_kv = (1, 8, seq_len, 64)

        q = mx.random.normal(shape_q)
        k = mx.random.normal(shape_kv)
        v = mx.random.normal(shape_kv)

        # Dense baseline
        scale = 1.0 / math.sqrt(64)
        dense_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(dense_out)

        # Sparse with all blocks (should match dense)
        sparse_out, n_active, mode = AdaptiveBlockSparseAttention.execute(
            q, k, v, top_k_ratio=1.0, block_size=64,
        )
        mx.eval(sparse_out)

        cos = cosine_similarity(dense_out, sparse_out)
        assert cos > 0.999, f"top_k=1.0 should match dense for seq_len={seq_len}, got {cos}"


def test_long_context_memory_scales_reasonably():
    """Verify that cache memory usage scales linearly with sequence length."""
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=8, v_bits=3, use_incoherent=False,
            max_memory_gb=2.0, max_pinned_memory_gb=1.0, cache_dir=td,
        )

        sizes_bytes = []
        for seq_len in [512, 1024, 2048]:
            mx.random.seed(42)
            shape = (1, 8, seq_len, 64)
            k = mx.random.normal(shape)
            v = mx.random.normal(shape)
            mgr.store(f"mem_{seq_len}", k, v, seq_len)
            cache = mgr.active_caches[f"mem_{seq_len}"]
            est = mgr._estimate_cache_bytes(cache)
            sizes_bytes.append(est)

        # Memory should roughly double when sequence length doubles
        ratio_1024_512 = sizes_bytes[1] / sizes_bytes[0] if sizes_bytes[0] > 0 else 0
        ratio_2048_1024 = sizes_bytes[2] / sizes_bytes[1] if sizes_bytes[1] > 0 else 0

        # Allow 20% tolerance for rounding/padding effects
        assert 1.8 < ratio_1024_512 < 2.2, f"512→1024 ratio={ratio_1024_512}"
        assert 1.8 < ratio_2048_1024 < 2.2, f"1024→2048 ratio={ratio_2048_1024}"
