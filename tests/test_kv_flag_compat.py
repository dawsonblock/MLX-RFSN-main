"""Regression tests for KV compression flag naming and compat alias.

Covers:
- Canonical RFSN_ENABLE_KV_COMPRESSION env var
- Deprecated RFSN_ENABLE_QUANTIZED_KV alias emits DeprecationWarning
- New name takes precedence when both are set
"""
from __future__ import annotations

import warnings

import pytest


@pytest.mark.unit
def test_canonical_kv_flag_false(monkeypatch):
    monkeypatch.delenv("RFSN_ENABLE_KV_COMPRESSION", raising=False)
    monkeypatch.delenv("RFSN_ENABLE_QUANTIZED_KV", raising=False)
    from rfsn_v10.config import _resolve_kv_compression_env
    assert _resolve_kv_compression_env() is False


@pytest.mark.unit
def test_canonical_kv_flag_true(monkeypatch):
    monkeypatch.setenv("RFSN_ENABLE_KV_COMPRESSION", "true")
    monkeypatch.delenv("RFSN_ENABLE_QUANTIZED_KV", raising=False)
    from rfsn_v10.config import _resolve_kv_compression_env
    assert _resolve_kv_compression_env() is True


@pytest.mark.unit
def test_deprecated_alias_emits_warning(monkeypatch):
    monkeypatch.setenv("RFSN_ENABLE_QUANTIZED_KV", "true")
    monkeypatch.delenv("RFSN_ENABLE_KV_COMPRESSION", raising=False)
    from rfsn_v10.config import _resolve_kv_compression_env
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_kv_compression_env()
    assert result is True
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("RFSN_ENABLE_QUANTIZED_KV" in str(w.message) for w in caught)


@pytest.mark.unit
def test_new_name_takes_precedence_over_alias(monkeypatch):
    monkeypatch.setenv("RFSN_ENABLE_KV_COMPRESSION", "false")
    monkeypatch.setenv("RFSN_ENABLE_QUANTIZED_KV", "true")
    from rfsn_v10.config import _resolve_kv_compression_env
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_kv_compression_env()
    assert result is False
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.unit
def test_runtime_config_has_kv_field(monkeypatch):
    monkeypatch.delenv("RFSN_ENABLE_KV_COMPRESSION", raising=False)
    monkeypatch.delenv("RFSN_ENABLE_QUANTIZED_KV", raising=False)
    from rfsn_v10.config import RFSNConfig
    cfg = RFSNConfig.from_env()
    assert cfg.runtime.enable_kv_compression is False


@pytest.mark.unit
def test_runtime_config_kv_on(monkeypatch):
    monkeypatch.setenv("RFSN_ENABLE_KV_COMPRESSION", "true")
    monkeypatch.delenv("RFSN_ENABLE_QUANTIZED_KV", raising=False)
    from rfsn_v10.config import RFSNConfig
    cfg = RFSNConfig.from_env()
    assert cfg.runtime.enable_kv_compression is True
