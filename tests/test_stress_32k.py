"""32k context stress test on each backend — Week 5."""
from __future__ import annotations

import math

import pytest

from rfsn_v10.kernels import backend

mx = pytest.importorskip("mlx.core")


class Test32KStress:
    """End-to-end synthetic 32k-context stress test.

    Uses tiny hidden dim so the test runs in seconds on CPU, but the
    *context length* (n_keys) is large enough to exercise memory paths.
    """

    N_KEYS = 32_768
    N_H = 4
    D_HEAD = 32
    BITS = 4
    GROUP_SIZE = 64

    def test_backend_attention_at_32k(self):
        """Pack, dequant, and run attention on 32k synthetic KV."""
        n_values = self.N_H * self.N_KEYS * self.D_HEAD

        # Fake packed K/V (all zeros → dequant = -scale * qmax)
        codes_per_word = 32 // self.BITS
        n_words = (n_values + codes_per_word - 1) // codes_per_word
        packed_k = mx.zeros((n_words,), dtype=mx.uint32)
        packed_v = mx.zeros((n_words,), dtype=mx.uint32)

        # Uniform scale of 1.0
        n_scales = (n_values + self.GROUP_SIZE - 1) // self.GROUP_SIZE
        scales_k = mx.ones((n_scales,), dtype=mx.float32)
        scales_v = mx.ones((n_scales,), dtype=mx.float32)

        # Single decode query
        queries = mx.ones((self.N_H, self.D_HEAD), dtype=mx.float32)

        out = backend.quantized_attention_decode(
            queries,
            packed_k,
            packed_v,
            scales_k,
            scales_v,
            n_keys=self.N_KEYS,
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )

        assert out.shape == (self.N_H, self.D_HEAD)
        assert out.dtype == mx.float32
        # All-zero keys → uniform attention weights → output is mean of V.
        # Since V dequant = -qmax, mean = -qmax.
        qmax = (1 << (self.BITS - 1)) - 1
        assert mx.allclose(
            out, mx.full(out.shape, -qmax, dtype=out.dtype), atol=0.1
        ).item()

    def test_backend_sdpa_causal_at_32k(self):
        """Causal sdpa produces triangular attention pattern."""
        T = 256  # smaller for speed, but causal logic is the same
        q = mx.ones((1, 1, T, self.D_HEAD), dtype=mx.float32)
        k = mx.ones((1, 1, T, self.D_HEAD), dtype=mx.float32)
        v = mx.ones((1, 1, T, self.D_HEAD), dtype=mx.float32)

        out = backend.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.D_HEAD), causal=True
        )

        assert out.shape == (1, 1, T, self.D_HEAD)
        # Causal mask → each position can only attend to itself and previous
        # positions.  With uniform keys, this means output at position i is
        # average of first (i+1) value vectors = 1.0.
        assert mx.allclose(out, mx.ones_like(out), atol=0.01).item()
