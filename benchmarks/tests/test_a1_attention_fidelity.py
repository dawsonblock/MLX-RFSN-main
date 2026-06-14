"""Step 9: A1 attention fidelity tests.

Tests the quality of Q @ K_compressed^T vs Q @ K_dense^T, and the
downstream attention output quality when using compressed values.
No model download required — uses synthetic Q/K/V.

Metrics tested:
    attention_score_cosine  >= 0.995
    attention_score_mae     (reported)
    attention_top5_overlap  >= 0.95
    softmax_kl              <= 0.05

Run:
    pytest benchmarks/tests/test_a1_attention_fidelity.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed"),
    pytest.mark.mlx_required,
    pytest.mark.unit,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _attention_score_cosine(q: np.ndarray, k_dense: np.ndarray, k_comp: np.ndarray, scale: float) -> float:
    scores_d = q @ k_dense.T * scale  # (T_q, T_kv)
    scores_c = q @ k_comp.T * scale
    flat_d = scores_d.flatten()
    flat_c = scores_c.flatten()
    cosine = np.dot(flat_d, flat_c) / (np.linalg.norm(flat_d) * np.linalg.norm(flat_c) + 1e-12)
    return float(cosine)


def _attention_top5_overlap(q: np.ndarray, k_dense: np.ndarray, k_comp: np.ndarray, scale: float) -> float:
    scores_d = q @ k_dense.T * scale
    scores_c = q @ k_comp.T * scale
    T_q = scores_d.shape[0]
    overlaps = []
    for t in range(T_q):
        d5 = set(np.argsort(scores_d[t])[-5:])
        c5 = set(np.argsort(scores_c[t])[-5:])
        overlaps.append(len(d5 & c5) / 5.0)
    return float(np.mean(overlaps))


def _softmax_kl(q: np.ndarray, k_dense: np.ndarray, k_comp: np.ndarray, scale: float) -> float:
    scores_d = q @ k_dense.T * scale
    scores_c = q @ k_comp.T * scale
    p_d = _softmax(scores_d)
    p_c = _softmax(scores_c)
    kl = float(np.mean(np.sum(p_d * (np.log(p_d + 1e-12) - np.log(p_c + 1e-12)), axis=-1)))
    return kl


def _wht_roundtrip_kv(keys_np: np.ndarray, values_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply WHT compress/decompress (k8, v4) and return reconstructed K, V."""
    from rfsn_v11.quant.key_quant import KeyQuant
    kq_k = KeyQuant(bits=8, group_size=64, use_wht=True, use_incoherent_signs=False)
    kq_v = KeyQuant(bits=4, group_size=64, use_wht=True, use_incoherent_signs=False)

    k = mx.array(keys_np)
    v = mx.array(values_np)

    k_wht = kq_k._apply_wht_pretransform(k)
    w_q, s, b = mx.quantize(k_wht, group_size=64, bits=8)
    k_recon = kq_k._apply_wht_pretransform(mx.dequantize(w_q, s, b, group_size=64, bits=8))

    v_wht = kq_v._apply_wht_pretransform(v)
    w_qv, sv, bv = mx.quantize(v_wht, group_size=64, bits=4)
    v_recon = kq_v._apply_wht_pretransform(mx.dequantize(w_qv, sv, bv, group_size=64, bits=4))

    return np.array(k_recon), np.array(v_recon)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qkv_512():
    """Q, K, V for a 512-token context, head_dim=128."""
    rng = np.random.default_rng(0)
    T, D = 512, 128
    scale = 1.0 / math.sqrt(D)
    Q = (rng.standard_normal((T, D)) / math.sqrt(D)).astype(np.float32)
    K = rng.standard_normal((T, D)).astype(np.float32)
    V = rng.standard_normal((T, D)).astype(np.float32)
    return Q, K, V, scale


@pytest.fixture(scope="module")
def qkv_2048():
    rng = np.random.default_rng(1)
    T, D = 2048, 128
    scale = 1.0 / math.sqrt(D)
    Q = (rng.standard_normal((T, D)) / math.sqrt(D)).astype(np.float32)
    K = rng.standard_normal((T, D)).astype(np.float32)
    V = rng.standard_normal((T, D)).astype(np.float32)
    return Q, K, V, scale


# ---------------------------------------------------------------------------
# Tests: Attention score fidelity (Q @ K^T)
# ---------------------------------------------------------------------------

class TestAttentionScoreFidelity:
    """Q @ K_dense^T vs Q @ K_compressed^T."""

    def _get_compressed_k(self, K_np: np.ndarray) -> np.ndarray:
        from rfsn_v11.quant.key_quant import KeyQuant
        kq = KeyQuant(bits=8, group_size=64, use_wht=True, use_incoherent_signs=False)
        K = mx.array(K_np)
        K_wht = kq._apply_wht_pretransform(K)
        w_q, s, b = mx.quantize(K_wht, group_size=64, bits=8)
        K_recon = kq._apply_wht_pretransform(mx.dequantize(w_q, s, b, group_size=64, bits=8))
        return np.array(K_recon)

    def test_cosine_512(self, qkv_512):
        Q, K, V, scale = qkv_512
        K_comp = self._get_compressed_k(K)
        cosine = _attention_score_cosine(Q, K, K_comp, scale)
        print(f"\nattention_score_cosine (512): {cosine:.6f}")
        assert cosine >= 0.995, f"attention_score_cosine {cosine:.6f} < 0.995 at T=512"

    def test_cosine_2048(self, qkv_2048):
        Q, K, V, scale = qkv_2048
        K_comp = self._get_compressed_k(K)
        cosine = _attention_score_cosine(Q, K, K_comp, scale)
        print(f"\nattention_score_cosine (2048): {cosine:.6f}")
        assert cosine >= 0.995, f"attention_score_cosine {cosine:.6f} < 0.995 at T=2048"

    def test_top5_overlap_512(self, qkv_512):
        Q, K, V, scale = qkv_512
        K_comp = self._get_compressed_k(K)
        overlap = _attention_top5_overlap(Q, K, K_comp, scale)
        print(f"\nattention_top5_overlap (512): {overlap:.4f}")
        assert overlap >= 0.95, f"attention_top5_overlap {overlap:.4f} < 0.95"

    def test_softmax_kl_512(self, qkv_512):
        Q, K, V, scale = qkv_512
        K_comp = self._get_compressed_k(K)
        kl = _softmax_kl(Q, K, K_comp, scale)
        print(f"\nsoftmax_kl (512): {kl:.6f}")
        assert kl <= 0.05, f"softmax_kl {kl:.6f} > 0.05"

    def test_mae_512(self, qkv_512):
        Q, K, V, scale = qkv_512
        K_comp = self._get_compressed_k(K)
        scores_d = Q @ K.T * scale
        scores_c = Q @ K_comp.T * scale
        mae = float(np.mean(np.abs(scores_d - scores_c)))
        print(f"\nattention_score_mae (512): {mae:.6e}")
        # No hard threshold on MAE; just report

    def test_head_dim_64(self):
        """Works correctly for Qwen2.5-0.5B head_dim=64."""
        from rfsn_v11.quant.key_quant import KeyQuant
        kq = KeyQuant(bits=8, group_size=64, use_wht=True, use_incoherent_signs=False)
        rng = np.random.default_rng(5)
        T, D = 128, 64
        scale = 1.0 / math.sqrt(D)
        Q = (rng.standard_normal((T, D)) / math.sqrt(D)).astype(np.float32)
        K = rng.standard_normal((T, D)).astype(np.float32)
        K_mx = mx.array(K)
        K_wht = kq._apply_wht_pretransform(K_mx)
        w_q, s, b = mx.quantize(K_wht, group_size=64, bits=8)
        K_comp = np.array(kq._apply_wht_pretransform(mx.dequantize(w_q, s, b, group_size=64, bits=8)))
        cosine = _attention_score_cosine(Q, K, K_comp, scale)
        print(f"\nattention_score_cosine (head_dim=64, T=128): {cosine:.6f}")
        assert cosine >= 0.995


# ---------------------------------------------------------------------------
# Tests: Attention output fidelity (softmax(Q K^T / sqrt(D)) @ V)
# ---------------------------------------------------------------------------

class TestAttentionOutputFidelity:
    """Full attention output: dense vs compressed K/V."""

    def test_output_cosine_512(self, qkv_512):
        Q, K, V, scale = qkv_512
        K_comp, V_comp = _wht_roundtrip_kv(K, V)

        def _attn_output(q, k, v, s):
            scores = _softmax(q @ k.T * s)
            return scores @ v  # (T, D)

        out_dense = _attn_output(Q, K, V, scale)
        out_comp = _attn_output(Q, K_comp, V_comp, scale)

        flat_d = out_dense.flatten()
        flat_c = out_comp.flatten()
        cosine = float(np.dot(flat_d, flat_c) / (np.linalg.norm(flat_d) * np.linalg.norm(flat_c) + 1e-12))
        print(f"\nattention_output_cosine (512): {cosine:.6f}")
        assert cosine >= 0.990, f"attention output cosine {cosine:.6f} < 0.990"


# ---------------------------------------------------------------------------
# Tests: A1 KV Cache full round-trip through update_and_fetch
# ---------------------------------------------------------------------------

class TestA1CacheAttentionFidelity:
    """Tests the A1_WHT_GroupedKVCache.update_and_fetch path for attention fidelity."""

    def test_attention_score_cosine_via_cache(self):
        from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_GroupedKVCache
        cache = A1_WHT_GroupedKVCache(head_dim=128, key_bits=8, value_bits=4, group_size=64)

        rng = np.random.default_rng(42)
        B, H, T, D = 1, 4, 64, 128
        scale = 1.0 / math.sqrt(D)

        keys_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        values_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        q_np = (rng.standard_normal((B, H, 1, D)) / math.sqrt(D)).astype(np.float32)

        keys = mx.array(keys_np)
        values = mx.array(values_np)
        k_comp, _ = cache.update_and_fetch(keys, values)

        # Compare Q @ K_dense^T vs Q @ K_comp^T (for one head)
        q_vec = q_np[0, 0, 0]  # (D,)
        k_dense = keys_np[0, 0]  # (T, D)
        k_compr = np.array(k_comp[0, 0])  # (T, D)

        cosine = _attention_score_cosine(
            q_vec[None],  # (1, D)
            k_dense,       # (T, D)
            k_compr,       # (T, D)
            scale,
        )
        print(f"\nA1 cache attention_score_cosine: {cosine:.6f}")
        assert cosine >= 0.995, f"A1 cache attn cosine {cosine:.6f} < 0.995"
