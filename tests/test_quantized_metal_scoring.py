"""RFSN v10 — Quantized Metal attention scoring mode tests."""

from __future__ import annotations

import pytest

from rfsn_v10.kernels import maybe_supports_metal_kernels
from rfsn_v10.runtime.scoring_modes import score_attention_quantized_metal

mx = pytest.importorskip("mlx.core")


class TestQuantizedMetalScoring:
    """Tests for the fused quantized Metal attention scoring path."""

    def test_importable(self):
        assert callable(score_attention_quantized_metal)

    def test_raises_without_metal(self):
        """If Metal kernels are unavailable the function must raise."""
        if maybe_supports_metal_kernels():
            pytest.skip("Metal kernels are available on this platform")

        queries = mx.zeros((1, 4, 1, 64))
        packed_k = mx.zeros((4, 16), dtype=mx.uint32)
        packed_v = mx.zeros((4, 16), dtype=mx.uint32)
        scales_k = mx.ones((4, 16))
        scales_v = mx.ones((4, 16))

        with pytest.raises(Exception):
            score_attention_quantized_metal(
                queries,
                packed_k=packed_k,
                packed_v=packed_v,
                scales_k=scales_k,
                scales_v=scales_v,
                n_keys=64,
                bits=4,
            )

    @pytest.mark.skipif(
        not maybe_supports_metal_kernels(),
        reason="Metal kernels not available",
    )
    def test_smoke_with_metal(self):
        """Basic smoke test when Metal is present."""
        queries = mx.zeros((1, 4, 1, 64))
        n_h = 4
        n_keys = 64
        d_head = 64
        bits = 4
        group_size = 64
        codes_per_word = 32 // bits
        n_values = n_h * n_keys * d_head
        required_words = (n_values + codes_per_word - 1) // codes_per_word
        required_scales = (n_values + group_size - 1) // group_size
        packed_k = mx.zeros(required_words, dtype=mx.uint32)
        packed_v = mx.zeros(required_words, dtype=mx.uint32)
        scales_k = mx.ones(required_scales)
        scales_v = mx.ones(required_scales)

        out = score_attention_quantized_metal(
            queries,
            packed_k=packed_k,
            packed_v=packed_v,
            scales_k=scales_k,
            scales_v=scales_v,
            n_keys=n_keys,
            bits=bits,
            group_size=group_size,
        )
        assert out.shape == (1, 4, 1, 64)
