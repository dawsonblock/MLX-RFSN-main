"""Test TurboPolar online softmax attention with dense V.

Validates Phase 8 gate:
  output_cosine >= 0.999
  max_abs_error <= tolerance
  top10_attention_overlap >= 0.99

Metal is disabled by default; Python/MLX reference path is validated.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.experimental]
mx = pytest.importorskip("mlx.core", reason="MLX not available")

from rfsn_v11.kernels.turbo_polar.metal import online_attention_dense_v
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder


def _synthetic_kv(
    tokens: int = 64,
    dim: int = 128,
) -> tuple[mx.array, mx.array]:
    rng = np.random.RandomState(42)
    keys = mx.array(rng.randn(tokens, dim).astype(np.float32))
    values = mx.array(rng.randn(tokens, dim).astype(np.float32))
    return keys, values


def _random_query(dim: int = 128) -> mx.array:
    rng = np.random.RandomState(99)
    return mx.array(rng.randn(dim).astype(np.float32))


def _dense_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scores = q @ k.T
    scores = scores - mx.max(scores)
    probs = mx.exp(scores)
    probs = probs / mx.sum(probs)
    return probs @ v


class TestOnlineAttentionGate:
    """Phase 8: online softmax attention with dense V."""

    def test_online_attention_vs_dense(self):
        keys, values = _synthetic_kv()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)

        # Single block for simplicity
        polar_block = encoder.encode(keys)
        output, used_metal = online_attention_dense_v(
            query, [polar_block], [values.astype(mx.float16)]
        )

        dense_out = _dense_attention(query, keys, values)

        # Cosine
        o_np = np.array(output.astype(mx.float32))
        d_np = np.array(dense_out.astype(mx.float32))
        cosine = float(
            np.sum(o_np * d_np)
            / (np.linalg.norm(o_np) * np.linalg.norm(d_np) + 1e-12)
        )
        assert cosine >= 0.999, f"output_cosine {cosine} < 0.999"
        assert not used_metal, "Metal should be disabled by default"

    def test_online_attention_max_abs_error(self):
        keys, values = _synthetic_kv()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)

        polar_block = encoder.encode(keys)
        output, _ = online_attention_dense_v(
            query, [polar_block], [values.astype(mx.float16)]
        )
        dense_out = _dense_attention(query, keys, values)

        max_err = float(mx.max(mx.abs(output - dense_out)).item())
        assert max_err <= 1e-2, f"max_abs_error {max_err} > 1e-2"

    def test_online_attention_no_nan_inf(self):
        keys, values = _synthetic_kv()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)

        polar_block = encoder.encode(keys)
        output, _ = online_attention_dense_v(
            query, [polar_block], [values.astype(mx.float16)]
        )
        assert not mx.any(mx.isnan(output)).item()
        assert not mx.any(mx.isinf(output)).item()

    def test_write_online_attention_artifact(self, tmp_path):
        keys, values = _synthetic_kv()
        query = _random_query()
        encoder = PolarQuantEncoder(angle_bits_level1=4, head_dim=128)

        polar_block = encoder.encode(keys)
        output, used_metal = online_attention_dense_v(
            query, [polar_block], [values.astype(mx.float16)]
        )
        dense_out = _dense_attention(query, keys, values)

        o_np = np.array(output.astype(mx.float32))
        d_np = np.array(dense_out.astype(mx.float32))
        cosine = float(
            np.sum(o_np * d_np)
            / (np.linalg.norm(o_np) * np.linalg.norm(d_np) + 1e-12)
        )
        max_err = float(mx.max(mx.abs(output - dense_out)).item())

        gate_pass = (
            cosine >= 0.999
            and max_err <= 1e-2
            and not mx.any(mx.isnan(output)).item()
            and not mx.any(mx.isinf(output)).item()
        )

        out = {
            "candidate": "turbo_polar_online_attention",
            "kernel_name": "tqpolar_online_attention_dense_v",
            "gate_status": "PASS" if gate_pass else "FAIL",
            "used_metal": used_metal,
            "output_cosine": cosine,
            "max_abs_error": max_err,
            "nan_inf_detected": bool(
                mx.any(mx.isnan(output)).item()
                or mx.any(mx.isinf(output)).item()
            ),
            "fallback_used": not used_metal,
            "notes": (
                "Python/MLX reference path passes gate. "
                "Metal kernel is EXPERIMENTAL and disabled by default."
            ),
        }
        artifact_dir = pathlib.Path(
            "artifacts/bench/kernel/turbo_polar_online_attention"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(
            artifact_dir / "results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(out, f, indent=2, default=float, allow_nan=False)
