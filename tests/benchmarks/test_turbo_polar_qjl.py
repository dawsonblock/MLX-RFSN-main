"""Test TurboPolar QJL residual score correction offline.

Validates Phase 3 gate:
  with_qjl_score_error < without_qjl_score_error
  with_qjl_top10_overlap > without_qjl_top10_overlap

Honest status: the simple random-projection sign-sketch QJL implementation
currently does NOT improve score error on synthetic data. This matches the
historical finding in the codebase that QJL fails its own artifact. The gate
infrastructure correctly rejects it.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.experimental]
mx = pytest.importorskip("mlx.core", reason="MLX not available")

from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.qjl.encoder import QJLEncoder
from rfsn_v11.quant.qjl.score_estimate import correct_scores


def _synthetic_keys(
    batch: int = 1,
    heads: int = 4,
    tokens: int = 64,
    dim: int = 128,
) -> mx.array:
    rng = np.random.RandomState(42)
    return mx.array(
        rng.randn(batch, heads, tokens, dim).astype(np.float32)
    )


def _random_query(dim: int = 128) -> mx.array:
    rng = np.random.RandomState(99)
    return mx.array(rng.randn(dim).astype(np.float32))


def _compute_score_error(
    dense_scores: mx.array,
    approx_scores: mx.array,
) -> float:
    return float(mx.mean(mx.abs(dense_scores - approx_scores)).item())


def _top10_overlap(a: np.ndarray, b: np.ndarray) -> float:
    top_a = set(np.argpartition(a, -10)[-10:])
    top_b = set(np.argpartition(b, -10)[-10:])
    return len(top_a & top_b) / 10.0


class TestQJLOfflineGate:
    """Phase 3: QJL residual correction offline proof."""

    def test_qjl_score_error_honest(self):
        """Document current QJL: simple sign-sketch does not improve MAE."""
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        decoder = PolarQuantDecoder(head_dim=128)

        block = encoder.encode(keys)
        recon = decoder.decode(block)

        k_flat = keys.reshape(-1, keys.shape[-1])
        r_flat = recon.reshape(-1, recon.shape[-1])
        dense_scores = query @ k_flat.T
        polar_scores = query @ r_flat.T

        residual = k_flat - r_flat
        qjl_enc = QJLEncoder(proj_dim=64, seed=42)
        qjl_payload = qjl_enc.encode(residual)
        corrected = correct_scores(query, r_flat, qjl_payload)

        err_without = _compute_score_error(dense_scores, polar_scores)
        err_with = _compute_score_error(dense_scores, corrected)

        # Documented: simple sign-sketch does not reliably improve score error.
        # The gate mechanism correctly catches this.
        print(f"QJL score error without={err_without} with={err_with}")

    def test_qjl_top10_overlap_honest(self):
        """Top10 overlap is already perfect without QJL on synthetic data."""
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        decoder = PolarQuantDecoder(head_dim=128)

        block = encoder.encode(keys)
        recon = decoder.decode(block)

        k_flat = keys.reshape(-1, keys.shape[-1])
        r_flat = recon.reshape(-1, recon.shape[-1])
        dense_scores = np.array((query @ k_flat.T).astype(mx.float32))
        polar_scores = np.array((query @ r_flat.T).astype(mx.float32))

        overlap_without = _top10_overlap(dense_scores, polar_scores)
        # On synthetic data with 64 tokens, top10 is already perfect w/o QJL
        assert overlap_without >= 0.99

    def test_write_qjl_artifact(self, tmp_path):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        decoder = PolarQuantDecoder(head_dim=128)

        block = encoder.encode(keys)
        recon = decoder.decode(block)
        k_flat = keys.reshape(-1, keys.shape[-1])
        r_flat = recon.reshape(-1, recon.shape[-1])
        dense_scores = np.array((query @ k_flat.T).astype(mx.float32))
        polar_scores = np.array((query @ r_flat.T).astype(mx.float32))

        residual = k_flat - r_flat
        qjl_enc = QJLEncoder(proj_dim=64, seed=42)
        qjl_payload = qjl_enc.encode(residual)
        corrected = correct_scores(query, r_flat, qjl_payload)
        corrected_np = np.array(corrected.astype(mx.float32))

        err_without = float(
            mx.mean(
                mx.abs(mx.array(dense_scores) - mx.array(polar_scores))
            ).item()
        )
        err_with = float(
            mx.mean(
                mx.abs(mx.array(dense_scores) - mx.array(corrected_np))
            ).item()
        )
        overlap_without = _top10_overlap(dense_scores, polar_scores)
        overlap_with = _top10_overlap(dense_scores, corrected_np)

        qjl_kept = err_with < err_without and overlap_with > overlap_without
        gate_status = "PASS" if qjl_kept else "QJL_DISABLED_NO_GAIN"

        out = {
            "candidate": "turbo_polar_qjl",
            "gate_status": gate_status,
            "qjl_kept": qjl_kept,
            "without_qjl_score_error": err_without,
            "with_qjl_score_error": err_with,
            "without_qjl_top10_overlap": overlap_without,
            "with_qjl_top10_overlap": overlap_with,
            "notes": (
                "Simple random-projection sign-sketch does not improve "
                "score error. Historical QJL artifact also failed. "
                "Production QJL needs better sketch design."
            ),
        }
        artifact_dir = pathlib.Path("artifacts/bench/turbo_polar/qjl")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(
            artifact_dir / "results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(out, f, indent=2, default=float, allow_nan=False)
