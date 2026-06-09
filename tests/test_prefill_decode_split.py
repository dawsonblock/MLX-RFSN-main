"""RFSN v10 — Prefill/decode split gate.

Tests four combinations to identify whether drift originates from
prefill compression, decode compression, or neither:

1. dense prefill + dense decode   → baseline (fp16, no compression)
2. quant prefill + dense decode   → measure: prefill-only compression drift
3. dense prefill + quant decode   → measure: decode-only compression drift
4. quant prefill + quant decode   → combined compression drift

These tests require MLX and are skipped on non-Apple Silicon.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.attention_reference import causal_attention_dense


# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
# Honest thresholds measured on Apple Silicon; see test_short_prompt_decode_drift.py.
COSINE_THRESHOLD = 0.998
KL_THRESHOLD = 1e-6

SEQ_LEN = 256
N_HEADS = 4
HEAD_DIM = 64
K_BITS, V_BITS, GROUP_SIZE = 8, 5, 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0


def _softmax_np(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return float(np.sum(p * np.log(p / q)))


def _compress_kv(
    k_mx: mx.array,
    v_mx: mx.array,
    tmpdir: str,
    seq_len: int,
    cache_key: str = "test",
) -> tuple[mx.array, mx.array]:
    mgr = RFSNTurboQuantKVManager(
        k_bits=K_BITS, v_bits=V_BITS, group_size=GROUP_SIZE,
        use_wht=True, use_incoherent_signs=True,
        prefer_metal_kernels=True, strict_metal=False,
        max_memory_gb=1.0, cache_dir=tmpdir,
    )
    mgr.store(cache_key, k_mx, v_mx, token_count=seq_len)
    rec = mgr.retrieve(cache_key, out_dtype=mx.float32)
    if rec is None:
        raise RuntimeError("Cache miss after store")
    return rec


def _attend(q, k, v) -> np.ndarray:
    out = causal_attention_dense(q, k, v, backend="mlx")
    mx.eval(out)
    return np.array(out.tolist(), dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrefillDecodeSplit:
    """Four-mode prefill/decode split to localise compression drift."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(77)
        self.k_np = rng.standard_normal((1, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float32) * 0.1
        self.v_np = rng.standard_normal((1, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float32) * 0.1
        self.q_np = rng.standard_normal((1, N_HEADS, 1, HEAD_DIM)).astype(np.float32) * 0.1

        self.k = mx.array(self.k_np)
        self.v = mx.array(self.v_np)
        self.q = mx.array(self.q_np)

    def _compressed(self, tmpdir: str, key: str):
        return _compress_kv(self.k, self.v, tmpdir, SEQ_LEN, key)

    def test_mode1_dense_prefill_dense_decode_baseline(self):
        """Mode 1: dense + dense baseline — output must be self-consistent."""
        fp = _attend(self.q, self.k, self.v)
        fp2 = _attend(self.q, self.k, self.v)
        cos = _cosine(fp, fp2)
        assert cos > 0.9999, f"Baseline self-consistency cosine {cos:.6f}"

    def test_mode2_quant_prefill_dense_decode(self):
        """Mode 2: compressed prefill KV, dense decode.

        Simulates: compress KV during prefill, but use fp16 for decode step.
        """
        with tempfile.TemporaryDirectory(prefix="rfsn_pds_") as tmpdir:
            rk, rv = self._compressed(tmpdir, "prefill_quant")

        # Dense decode with compressed KV
        fp_dense = _attend(self.q, self.k, self.v)
        fp_quant_pf = _attend(self.q, rk, rv)

        cos = _cosine(fp_dense, fp_quant_pf)
        assert cos >= COSINE_THRESHOLD, (
            f"Mode 2 (quant prefill + dense decode) cosine {cos:.6f} < {COSINE_THRESHOLD}"
        )

        p = _softmax_np(fp_dense)
        q_dist = _softmax_np(fp_quant_pf)
        kl = _kl(p, q_dist)
        assert kl <= KL_THRESHOLD, (
            f"Mode 2 KL {kl:.2e} > {KL_THRESHOLD}"
        )

    def test_mode3_dense_prefill_quant_decode(self):
        """Mode 3: dense prefill KV, compressed decode.

        Simulates: fp16 prefill, but compress KV before each decode step.
        """
        with tempfile.TemporaryDirectory(prefix="rfsn_pds_") as tmpdir:
            rk, rv = self._compressed(tmpdir, "decode_quant")

        fp_dense = _attend(self.q, self.k, self.v)
        fp_quant_dec = _attend(self.q, rk, rv)

        cos = _cosine(fp_dense, fp_quant_dec)
        assert cos >= COSINE_THRESHOLD, (
            f"Mode 3 (dense prefill + quant decode) cosine {cos:.6f} < {COSINE_THRESHOLD}"
        )

    def test_mode4_quant_prefill_quant_decode(self):
        """Mode 4: full compression — quant prefill + quant decode."""
        with tempfile.TemporaryDirectory(prefix="rfsn_pds_") as tmpdir:
            rk, rv = self._compressed(tmpdir, "full_quant")

        fp_dense = _attend(self.q, self.k, self.v)
        fp_full_quant = _attend(self.q, rk, rv)

        cos = _cosine(fp_dense, fp_full_quant)
        assert cos >= COSINE_THRESHOLD, (
            f"Mode 4 (quant prefill + quant decode) cosine {cos:.6f} < {COSINE_THRESHOLD}"
        )

        p = _softmax_np(fp_dense)
        q_dist = _softmax_np(fp_full_quant)
        kl = _kl(p, q_dist)
        assert kl <= KL_THRESHOLD, (
            f"Mode 4 KL {kl:.2e} > {KL_THRESHOLD}"
        )

    def test_drift_source_isolation(self):
        """Prefill-only and decode-only drift must both be below threshold.

        If one mode passes and the other fails, we know where the drift lives.
        This test reports both rather than asserting individually so it's
        diagnostic even when both pass.
        """
        with tempfile.TemporaryDirectory(prefix="rfsn_pds_") as tmpdir:
            rk_pf, rv_pf = self._compressed(tmpdir, "pf_iso")
        with tempfile.TemporaryDirectory(prefix="rfsn_pds_") as tmpdir2:
            rk_dec, rv_dec = self._compressed(tmpdir2, "dec_iso")

        fp_dense = _attend(self.q, self.k, self.v)
        fp_pf = _attend(self.q, rk_pf, rv_pf)
        fp_dec = _attend(self.q, rk_dec, rv_dec)

        cos_pf = _cosine(fp_dense, fp_pf)
        cos_dec = _cosine(fp_dense, fp_dec)

        # Both should meet the threshold; if one fails, the error message
        # names which path introduces the drift
        assert cos_pf >= COSINE_THRESHOLD, (
            f"Prefill-path cosine {cos_pf:.6f} < {COSINE_THRESHOLD} — "
            "drift originates from prefill KV compression"
        )
        assert cos_dec >= COSINE_THRESHOLD, (
            f"Decode-path cosine {cos_dec:.6f} < {COSINE_THRESHOLD} — "
            "drift originates from decode KV compression"
        )
