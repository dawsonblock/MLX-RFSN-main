#!/usr/bin/env python3
"""
RFSN v10 — Experimental Quant Runtime acceptance tests.

Verifies mode validation, fallback selection, layer-policy loading,
and config parsing.  Does not require full MLX decode execution.
"""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=False)
def experimental_all_env(monkeypatch):
    """Enable all experimental flags for tests that require them."""
    monkeypatch.setenv("RFSN_EXPERIMENTAL_POLAR", "true")
    monkeypatch.setenv("RFSN_EXPERIMENTAL_QJL", "true")
    monkeypatch.setenv("RFSN_EXPERIMENTAL_ADAPTIVE", "true")
    import rfsn_v10.config as _cfg
    _cfg._config = None
    yield
    _cfg._config = None

from rfsn_v10.runtime.experimental_quant_runtime import (  # noqa: E402, I001
    ExperimentalQuantRuntime,
    LayerQuantPolicy,
    _adaptive_bits,
    _load_layer_policy,
    DEFAULT_QUANT_MODE,
)


# ------------------------------------------------------------------
# Mode validation / rejection tests
# ------------------------------------------------------------------

class TestModeValidation:
    def test_default_mode_is_stable(self):
        assert DEFAULT_QUANT_MODE == "stable_k8_v5_gs64"

    def test_rejects_invalid_quant_mode(self):
        with pytest.raises(ValueError, match="Unsupported quant_mode"):
            ExperimentalQuantRuntime(quant_mode="turbo_k8r8v6")

    def test_accepts_stable_mode(self, tmp_path):
        runtime = ExperimentalQuantRuntime(
            quant_mode="stable_k8_v5_gs64",
            telemetry_dir=str(tmp_path),
        )
        assert runtime.quant_mode == "stable_k8_v5_gs64"

    def test_accepts_adaptive_mode(self, tmp_path):
        runtime = ExperimentalQuantRuntime(
            quant_mode="adaptive",
            telemetry_dir=str(tmp_path),
        )
        assert runtime.quant_mode == "adaptive"

    def test_accepts_experimental_hybrid_mode(self, tmp_path, experimental_all_env):
        runtime = ExperimentalQuantRuntime(
            quant_mode="experimental_hybrid",
            telemetry_dir=str(tmp_path),
        )
        assert runtime.quant_mode == "experimental_hybrid"


# ------------------------------------------------------------------
# Adaptive bit heuristic tests
# ------------------------------------------------------------------

class TestAdaptiveBits:
    def test_short_sequence_uses_8_5_64(self):
        k, v, gs = _adaptive_bits(512)
        assert k == 8 and v == 5 and gs == 64

    def test_medium_sequence_uses_6_4_64(self):
        k, v, gs = _adaptive_bits(1024)
        assert k == 6 and v == 4 and gs == 64

    def test_long_sequence_uses_4_3_64(self):
        k, v, gs = _adaptive_bits(4096)
        assert k == 4 and v == 3 and gs == 64

    def test_zero_seq_len_raises(self):
        with pytest.raises(ValueError, match="seq_len must be positive"):
            _adaptive_bits(0)

    def test_negative_seq_len_raises(self):
        with pytest.raises(ValueError, match="seq_len must be positive"):
            _adaptive_bits(-1)

    def test_boundary_2048(self):
        k, v, gs = _adaptive_bits(2048)
        assert k == 6 and v == 4 and gs == 64

    def test_very_long_sequence_uses_4_3_64(self):
        k, v, gs = _adaptive_bits(100_000)
        assert k == 4 and v == 3 and gs == 64


# ------------------------------------------------------------------
# Layer policy loading tests
# ------------------------------------------------------------------

class TestLoadLayerPolicy:
    def test_load_valid_flat_policy(self, tmp_path):
        data = {
            "0": {"k_bits": 8, "v_bits": 5, "group_size": 64},
            "1": {"k_bits": 6, "v_bits": 4, "group_size": 32},
        }
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data))
        policy = _load_layer_policy(str(path))
        assert 0 in policy
        assert 1 in policy
        assert policy[0].k_bits == 8
        assert policy[1].v_bits == 4

    def test_load_structured_policy(self, tmp_path):
        data = {
            "default": {"mode": "cartesian"},
            "layers": {
                "0": {"k_bits": 8, "v_bits": 5, "group_size": 64},
            },
        }
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data))
        policy = _load_layer_policy(str(path))
        assert 0 in policy
        assert policy[0].k_bits == 8

    def test_none_path_returns_empty(self):
        assert _load_layer_policy(None) == {}

    def test_invalid_path_raises_jsondecodeerror(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            _load_layer_policy(str(bad))


# ------------------------------------------------------------------
# LayerQuantPolicy dataclass tests
# ------------------------------------------------------------------

class TestLayerQuantPolicy:
    def test_defaults(self):
        p = LayerQuantPolicy()
        assert p.k_bits == 8
        assert p.v_bits == 5
        assert p.group_size == 64
        assert p.use_wht is True
        assert p.quant_mode == "cartesian"

    def test_custom_values(self):
        p = LayerQuantPolicy(k_bits=6, v_bits=4, group_size=32)
        assert p.k_bits == 6
        assert p.v_bits == 4
        assert p.group_size == 32


# ------------------------------------------------------------------
# Telemetry / startup logging tests
# ------------------------------------------------------------------

class TestTelemetryLogging:
    def test_startup_event_logged(self, tmp_path):
        runtime = ExperimentalQuantRuntime(
            quant_mode="stable_k8_v5_gs64",
            telemetry_dir=str(tmp_path),
            model_id="test_model",
        )
        log_path = tmp_path / "experimental_quant_telemetry.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["event_type"] == "startup"
        assert record["model_id"] == "test_model"
        assert record["quant_mode"] == "stable_k8_v5_gs64"
        _ = runtime

    def test_layer_policy_layers_logged(self, tmp_path):
        policy_data = {
            "default": {"mode": "cartesian"},
            "layers": {"0": {"k_bits": 8, "v_bits": 5, "group_size": 64}},
        }
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(json.dumps(policy_data))
        runtime = ExperimentalQuantRuntime(
            quant_mode="stable_k8_v5_gs64",
            telemetry_dir=str(tmp_path),
            layer_policy_path=str(policy_path),
        )
        log_path = tmp_path / "experimental_quant_telemetry.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[0])
        assert record["layer_policy_layers"] == [0]
        _ = runtime


# ------------------------------------------------------------------
# Manager factory tests
# ------------------------------------------------------------------

class TestManagerFactory:
    def test_stable_manager_bits(self, tmp_path):
        runtime = ExperimentalQuantRuntime(
            quant_mode="stable_k8_v5_gs64",
            telemetry_dir=str(tmp_path),
        )
        mgr = runtime._make_manager_for_mode("stable_k8_v5_gs64")
        assert mgr.k_bits == 8
        assert mgr.v_bits == 5
        assert mgr.group_size == 64

    def test_adaptive_manager_defaults(self, tmp_path):
        runtime = ExperimentalQuantRuntime(
            quant_mode="adaptive",
            telemetry_dir=str(tmp_path),
        )
        mgr = runtime._make_manager_for_mode("adaptive")
        assert mgr.k_bits == 8
        assert mgr.v_bits == 5

    def test_hybrid_manager_uses_polar_cartesian(self, tmp_path, experimental_all_env):
        runtime = ExperimentalQuantRuntime(
            quant_mode="experimental_hybrid",
            telemetry_dir=str(tmp_path),
        )
        mgr = runtime._make_manager_for_mode("experimental_hybrid")
        assert mgr.quant_mode == "hybrid_polar_cartesian"
