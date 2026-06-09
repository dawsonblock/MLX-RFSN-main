"""Tests for experimental feature opt-in guards (Phase 4).

Verifies:
1. QJL cannot activate unless experimental.enable_qjl=true.
2. Polar/hybrid modes cannot activate unless experimental.enable_polar=true.
3. Adaptive cannot activate unless experimental.enable_adaptive=true.
4. Stable runtime never imports experimental modules at collection time.
5. require_experimental() emits a warning when a feature IS enabled.
"""
from __future__ import annotations

import os
import warnings

import pytest

from rfsn_v10.config import (
    ExperimentalConfig,
    RFSNConfig,
    require_experimental,
)


class TestExperimentalConfigDefaults:
    """All experimental flags must be disabled by default."""

    def test_qjl_disabled_by_default(self):
        cfg = RFSNConfig()
        assert cfg.experimental.enable_qjl is False

    def test_polar_disabled_by_default(self):
        cfg = RFSNConfig()
        assert cfg.experimental.enable_polar is False

    def test_adaptive_disabled_by_default(self):
        cfg = RFSNConfig()
        assert cfg.experimental.enable_adaptive is False


class TestRequireExperimental:
    """require_experimental() must block disabled features and warn on enabled ones."""

    def _cfg(self, **kwargs) -> RFSNConfig:
        return RFSNConfig(experimental=ExperimentalConfig(**kwargs))

    def test_qjl_blocked_when_disabled(self):
        cfg = self._cfg(enable_qjl=False)
        with pytest.raises(RuntimeError, match="'qjl' is disabled"):
            require_experimental("qjl", config=cfg)

    def test_polar_blocked_when_disabled(self):
        cfg = self._cfg(enable_polar=False)
        with pytest.raises(RuntimeError, match="'polar' is disabled"):
            require_experimental("polar", config=cfg)

    def test_adaptive_blocked_when_disabled(self):
        cfg = self._cfg(enable_adaptive=False)
        with pytest.raises(RuntimeError, match="'adaptive' is disabled"):
            require_experimental("adaptive", config=cfg)

    def test_qjl_allowed_when_enabled_emits_warning(self):
        cfg = self._cfg(enable_qjl=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            require_experimental("qjl", config=cfg)
        assert any("not validated" in str(x.message) for x in w), (
            "Expected a warning about experimental mode not being validated"
        )

    def test_polar_allowed_when_enabled_emits_warning(self):
        cfg = self._cfg(enable_polar=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            require_experimental("polar", config=cfg)
        assert any("not validated" in str(x.message) for x in w)

    def test_adaptive_allowed_when_enabled_emits_warning(self):
        cfg = self._cfg(enable_adaptive=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            require_experimental("adaptive", config=cfg)
        assert any("not validated" in str(x.message) for x in w)

    def test_unknown_feature_raises_value_error(self):
        cfg = self._cfg()
        with pytest.raises(ValueError, match="Unknown experimental feature"):
            require_experimental("nonexistent", config=cfg)


class TestExperimentalEnvOverride:
    """Experimental flags can be enabled via environment variables."""

    def test_qjl_from_env(self, monkeypatch):
        monkeypatch.setenv("RFSN_EXPERIMENTAL_QJL", "true")
        cfg = RFSNConfig.from_env()
        assert cfg.experimental.enable_qjl is True

    def test_polar_from_env(self, monkeypatch):
        monkeypatch.setenv("RFSN_EXPERIMENTAL_POLAR", "true")
        cfg = RFSNConfig.from_env()
        assert cfg.experimental.enable_polar is True

    def test_adaptive_from_env(self, monkeypatch):
        monkeypatch.setenv("RFSN_EXPERIMENTAL_ADAPTIVE", "true")
        cfg = RFSNConfig.from_env()
        assert cfg.experimental.enable_adaptive is True

    def test_flags_default_false_without_env(self, monkeypatch):
        monkeypatch.delenv("RFSN_EXPERIMENTAL_QJL", raising=False)
        monkeypatch.delenv("RFSN_EXPERIMENTAL_POLAR", raising=False)
        monkeypatch.delenv("RFSN_EXPERIMENTAL_ADAPTIVE", raising=False)
        cfg = RFSNConfig.from_env()
        assert cfg.experimental.enable_qjl is False
        assert cfg.experimental.enable_polar is False
        assert cfg.experimental.enable_adaptive is False


class TestKVManagerExperimentalGuard:
    """RFSNTurboQuantKVManager raises for experimental modes without opt-in."""

    def test_cartesian_mode_always_allowed(self):
        """Stable cartesian mode must never require experimental opt-in."""
        from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
        mgr = RFSNTurboQuantKVManager(quant_mode="cartesian")
        assert mgr.quant_mode == "cartesian"

    def test_hybrid_polar_requires_opt_in(self, monkeypatch):
        monkeypatch.delenv("RFSN_EXPERIMENTAL_POLAR", raising=False)
        from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
        with pytest.raises(RuntimeError, match="'polar' is disabled"):
            RFSNTurboQuantKVManager(quant_mode="hybrid_polar_cartesian")

    def test_isoquant_requires_opt_in(self, monkeypatch):
        monkeypatch.delenv("RFSN_EXPERIMENTAL_POLAR", raising=False)
        from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
        with pytest.raises(RuntimeError, match="'polar' is disabled"):
            RFSNTurboQuantKVManager(quant_mode="isoquant")

    def test_qjl_requires_opt_in(self, monkeypatch):
        monkeypatch.delenv("RFSN_EXPERIMENTAL_QJL", raising=False)
        from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
        with pytest.raises(RuntimeError, match="'qjl' is disabled"):
            RFSNTurboQuantKVManager(use_qjl_score_correction=True)


class TestStableRuntimeNoExperimentalImport:
    """Stable runtime import must not drag in experimental modules.

    These tests verify that experimental submodules are NOT imported eagerly
    by checking via a subprocess so test ordering doesn't pollute sys.modules.
    """

    def _run_python(self, code: str) -> tuple[int, str]:
        import subprocess
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode, result.stdout + result.stderr

    def test_import_rfsn_v10_does_not_import_qjl(self):
        code = (
            "import sys; "
            "import rfsn_v10; "
            "bad = 'rfsn_v10.quantization.qjl_score_correction' in sys.modules; "
            "print('FOUND' if bad else 'CLEAN')"
        )
        rc, out = self._run_python(code)
        assert rc == 0, f"Subprocess failed: {out}"
        assert "CLEAN" in out, (
            "qjl_score_correction should not be imported at package load time. "
            f"Got: {out}"
        )

    def test_import_rfsn_v10_does_not_import_polar_quant(self):
        code = (
            "import sys; "
            "import rfsn_v10; "
            "bad = 'rfsn_v10.quantization.polar_quant' in sys.modules; "
            "print('FOUND' if bad else 'CLEAN')"
        )
        rc, out = self._run_python(code)
        assert rc == 0, f"Subprocess failed: {out}"
        assert "CLEAN" in out, (
            "polar_quant should not be imported at package load time. "
            f"Got: {out}"
        )
