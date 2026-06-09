"""Synthetic 7B-style slice integration test — Week 5.

Runs a full sparse attention pipeline on a tiny synthetic model with the
same *layer shapes* as Llama 2-7B (N=32 layers, H=32 heads, D=128) but
tiny sequence length so the test completes in <2 s.
"""
from __future__ import annotations

import pytest

from rfsn_v10.adaptive_sparsity import AdaptiveSparsityController
from rfsn_v10.attention import AdaptiveBlockSparseAttention

mx = pytest.importorskip("mlx.core")


class Test7BSynthetic:
    N_LAYERS = 4
    N_HEADS = 32
    D_HEAD = 128
    BLOCK_SIZE = 32
    T = 128

    def test_full_pipeline(self):
        """End-to-end sparse attention on synthetic weights."""
        controller = AdaptiveSparsityController()

        for layer_id in range(self.N_LAYERS):
            keys = mx.random.normal(
                (1, self.N_HEADS, self.T, self.D_HEAD)
            )
            values = mx.random.normal(
                (1, self.N_HEADS, self.T, self.D_HEAD)
            )

            # Prefill
            prefill_q = mx.random.normal(
                (1, self.N_HEADS, self.T, self.D_HEAD)
            )
            out_prefill, _, mode = AdaptiveBlockSparseAttention.execute(
                prefill_q,
                keys,
                values,
                top_k_ratio=0.5,
                block_size=self.BLOCK_SIZE,
            )
            assert out_prefill.shape == (1, self.N_HEADS, self.T, self.D_HEAD)
            assert mode == "dense_prefill"

            # Decode
            decode_q = mx.random.normal(
                (1, self.N_HEADS, 1, self.D_HEAD)
            )
            out_decode, _, mode = AdaptiveBlockSparseAttention.execute(
                decode_q,
                keys,
                values,
                top_k_ratio=0.5,
                block_size=self.BLOCK_SIZE,
            )
            assert out_decode.shape == (1, self.N_HEADS, 1, self.D_HEAD)

            # Sparsity controller decision
            decision = controller.get_decision(
                model_id="synthetic-7b",
                layer_id=str(layer_id),
            )
            assert 0.0 <= decision.top_k_ratio <= 1.0

        assert len(controller._profiles) > 0
