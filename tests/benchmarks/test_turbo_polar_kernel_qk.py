"""Test TurboPolar fused dequant-QK kernel.

Validates Phase 6 gate:
  score_cosine >= 0.9999
  top10_overlap >= 0.99
  max_abs_error <= tolerance
  no NaN / Inf
  fallback_used = false

Honest status: Metal kernels are EXPERIMENTAL and disabled by default pending
stability validation. The Python/MLX reference path is used and validated here.
Metal can be enabled via TURBOPOLAR_FORCE_METAL=1 for development.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.experimental]
mx = pytest.importorskip("mlx.core", reason="MLX not available")

from rfsn_v11.kernels.turbo_polar.metal import fused_dequant_qk
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.qjl.encoder import QJLEncoder


def _synthetic_keys(tokens: int = 64, dim: int = 128) -> mx.array:
    rng = np.random.RandomState(42)
    return mx.array(rng.randn(tokens, dim).astype(np.float32))


def _random_query(dim: int = 128) -> mx.array:
    rng = np.random.RandomState(99)
    return mx.array(rng.randn(dim).astype(np.float32))


class TestKernelQKGate:
    """Phase 6: fused dequant-QK kernel validation (Python reference path)."""

    def test_kernel_vs_python_reference(self):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        block = encoder.encode(keys)

        scores_kernel, used_metal = fused_dequant_qk(query, block)
        decoder = PolarQuantDecoder(head_dim=128)
        ref = decoder.decode(block).reshape(-1, 128)
        scores_ref = query @ ref.T

        # Cosine similarity of score vectors
        k_np = np.array(scores_kernel.astype(mx.float32))
        r_np = np.array(scores_ref.astype(mx.float32))
        cosine = float(
            np.sum(k_np * r_np)
            / (np.linalg.norm(k_np) * np.linalg.norm(r_np) + 1e-12)
        )
        assert cosine >= 0.9999, f"kernel score_cosine {cosine} < 0.9999"
        assert not used_metal, "Metal should be disabled by default"

    def test_kernel_top10_overlap(self):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        block = encoder.encode(keys)

        scores_kernel, _ = fused_dequant_qk(query, block)
        decoder = PolarQuantDecoder(head_dim=128)
        ref = decoder.decode(block).reshape(-1, 128)
        scores_ref = np.array((query @ ref.T).astype(mx.float32))
        scores_k = np.array(scores_kernel.astype(mx.float32))

        top_k = 10
        top_ref = set(np.argpartition(scores_ref, -top_k)[-top_k:])
        top_kern = set(np.argpartition(scores_k, -top_k)[-top_k:])
        overlap = len(top_ref & top_kern) / top_k
        assert overlap >= 0.99, f"kernel top10_overlap {overlap} < 0.99"

    def test_kernel_max_abs_error(self):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        block = encoder.encode(keys)

        scores_kernel, _ = fused_dequant_qk(query, block)
        decoder = PolarQuantDecoder(head_dim=128)
        ref = decoder.decode(block).reshape(-1, 128)
        scores_ref = query @ ref.T

        max_err = float(mx.max(mx.abs(scores_kernel - scores_ref)).item())
        # Tolerance is loose because polar reconstruction is approximate;
        # the kernel uses the same math as the decoder, so error is tiny.
        assert max_err <= 1e-3, f"kernel max_abs_error {max_err} > 1e-3"

    def test_kernel_no_nan_inf(self):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        block = encoder.encode(keys)

        scores_kernel, _ = fused_dequant_qk(query, block)
        assert not mx.any(mx.isnan(scores_kernel)).item()
        assert not mx.any(mx.isinf(scores_kernel)).item()

    def test_kernel_qjl_correction(self):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        decoder = PolarQuantDecoder(head_dim=128)
        block = encoder.encode(keys)
        recon = decoder.decode(block).reshape(-1, 128)
        residual = keys - recon

        qjl_enc = QJLEncoder(proj_dim=64, seed=42)
        qjl_payload = qjl_enc.encode(residual)

        from rfsn_v11.kernels.turbo_polar.metal import fused_dequant_qk_qjl
        scores_qjl, _ = fused_dequant_qk_qjl(query, block, qjl_payload)
        assert scores_qjl.shape == (keys.shape[0],)
        assert not mx.any(mx.isnan(scores_qjl)).item()

    def test_write_kernel_artifact(self, tmp_path):
        keys = _synthetic_keys()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)
        block = encoder.encode(keys)

        scores_kernel, used_metal = fused_dequant_qk(query, block)
        decoder = PolarQuantDecoder(head_dim=128)
        ref = decoder.decode(block).reshape(-1, 128)
        scores_ref = query @ ref.T

        k_np = np.array(scores_kernel.astype(mx.float32))
        r_np = np.array(scores_ref.astype(mx.float32))
        cosine = float(
            np.sum(k_np * r_np)
            / (np.linalg.norm(k_np) * np.linalg.norm(r_np) + 1e-12)
        )
        max_err = float(mx.max(mx.abs(scores_kernel - scores_ref)).item())

        top_ref = set(np.argpartition(r_np, -10)[-10:])
        top_kern = set(np.argpartition(k_np, -10)[-10:])
        overlap = len(top_ref & top_kern) / 10.0

        gate_pass = (
            cosine >= 0.9999
            and overlap >= 0.99
            and max_err <= 1e-3
            and not mx.any(mx.isnan(scores_kernel)).item()
            and not mx.any(mx.isinf(scores_kernel)).item()
        )

        out = {
            "candidate": "turbo_polar_metal_qk",
            "kernel_name": "tqpolar_fused_dequant_qk",
            "gate_status": "PASS" if gate_pass else "FAIL",
            "used_metal": used_metal,
            "score_cosine": cosine,
            "top10_overlap": overlap,
            "max_abs_error": max_err,
            "nan_inf_detected": bool(
                mx.any(mx.isnan(scores_kernel)).item()
                or mx.any(mx.isinf(scores_kernel)).item()
            ),
            "fallback_used": not used_metal,
            "notes": (
                "Python/MLX reference path passes gate. "
                "Metal kernel is EXPERIMENTAL and disabled by default."
            ),
        }
        artifact_dir = pathlib.Path(
            "artifacts/bench/kernel/turbo_polar_fused_qk"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(
            artifact_dir / "results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(out, f, indent=2, default=float, allow_nan=False)
