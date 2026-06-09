#!/usr/bin/env python3
"""RFSN v10 — Configuration management tests.

Covers config validation, environment variable loading, YAML round-trip,
and default factory behaviour without requiring MLX.
"""
from __future__ import annotations

import pytest

from rfsn_v10.config import (
    CacheConfig,
    LoggingConfig,
    MemoryConfig,
    QuantizationConfig,
    RFSNConfig,
    SparseAttentionConfig,
    get_config,
    load_config,
    set_config,
)

# ------------------------------------------------------------------
# LoggingConfig
# ------------------------------------------------------------------

class TestLoggingConfig:
    def test_default_values(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "json"
        assert cfg.file is None

    def test_level_normalised_to_upper(self):
        cfg = LoggingConfig(level="debug")
        assert cfg.level == "DEBUG"

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Log level"):
            LoggingConfig(level="VERBOSE")

    def test_valid_levels_accepted(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert LoggingConfig(level=level).level == level


# ------------------------------------------------------------------
# MemoryConfig
# ------------------------------------------------------------------

class TestMemoryConfig:
    def test_default_values(self):
        cfg = MemoryConfig()
        assert cfg.max_gb == 8.0
        assert cfg.quota_gb == 10.0
        assert cfg.enable_leak_detection is True

    def test_negative_max_gb_rejected(self):
        with pytest.raises(ValueError):
            MemoryConfig(max_gb=-1.0)

    def test_zero_max_gb_rejected(self):
        with pytest.raises(ValueError):
            MemoryConfig(max_gb=0.0)


# ------------------------------------------------------------------
# CacheConfig
# ------------------------------------------------------------------

class TestCacheConfig:
    def test_default_values(self):
        cfg = CacheConfig()
        assert cfg.directory == "~/.cache/rfsn"
        assert cfg.enable_persistence is True
        assert cfg.enable_wal is True


# ------------------------------------------------------------------
# SparseAttentionConfig
# ------------------------------------------------------------------

class TestSparseAttentionConfig:
    def test_default_values(self):
        cfg = SparseAttentionConfig()
        assert cfg.default_top_k_ratio == 0.3
        assert cfg.block_size == 64
        assert cfg.enable_adaptive is True

    def test_top_k_ratio_bounds(self):
        with pytest.raises(ValueError):
            SparseAttentionConfig(default_top_k_ratio=1.5)
        with pytest.raises(ValueError):
            SparseAttentionConfig(default_top_k_ratio=-0.1)

    def test_zero_block_size_rejected(self):
        with pytest.raises(ValueError):
            SparseAttentionConfig(block_size=0)


# ------------------------------------------------------------------
# QuantizationConfig
# ------------------------------------------------------------------

class TestQuantizationConfig:
    def test_default_values(self):
        cfg = QuantizationConfig()
        assert cfg.default_bits == 8
        assert cfg.group_size == 64
        assert cfg.enable_wht is True

    def test_bits_bounds(self):
        with pytest.raises(ValueError):
            QuantizationConfig(default_bits=1)
        with pytest.raises(ValueError):
            QuantizationConfig(default_bits=9)

    def test_valid_bits_accepted(self):
        for bits in range(2, 9):
            assert QuantizationConfig(default_bits=bits).default_bits == bits


# ------------------------------------------------------------------
# RFSNConfig
# ------------------------------------------------------------------

class TestRFSNConfig:
    def test_defaults(self):
        cfg = RFSNConfig()
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.sparse_attention, SparseAttentionConfig)
        assert isinstance(cfg.quantization, QuantizationConfig)

    def test_from_env_with_defaults(self, monkeypatch):
        monkeypatch.delenv("RFSN_LOG_LEVEL", raising=False)
        monkeypatch.delenv("RFSN_LOG_FORMAT", raising=False)
        monkeypatch.delenv("RFSN_MAX_MEMORY_GB", raising=False)
        cfg = RFSNConfig.from_env()
        assert cfg.logging.level == "INFO"
        assert cfg.memory.max_gb == 8.0

    def test_from_env_with_overrides(self, monkeypatch):
        monkeypatch.setenv("RFSN_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("RFSN_MAX_MEMORY_GB", "16")
        monkeypatch.setenv("RFSN_ENABLE_PERSISTENCE", "false")
        cfg = RFSNConfig.from_env()
        assert cfg.logging.level == "DEBUG"
        assert cfg.memory.max_gb == 16.0
        assert cfg.cache.enable_persistence is False

    def test_yaml_roundtrip(self, tmp_path):
        path = tmp_path / "config.yaml"
        original = RFSNConfig(
            logging=LoggingConfig(level="ERROR", format="text"),
            memory=MemoryConfig(max_gb=4.0),
        )
        original.to_yaml(str(path))
        loaded = RFSNConfig.from_yaml(str(path))
        assert loaded.logging.level == "ERROR"
        assert loaded.logging.format == "text"
        assert loaded.memory.max_gb == 4.0

    def test_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RFSNConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))


# ------------------------------------------------------------------
# load_config / get_config / set_config
# ------------------------------------------------------------------

class TestGlobalConfig:
    def test_load_config_prefers_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RFSN_LOG_LEVEL", raising=False)
        path = tmp_path / "config.yaml"
        RFSNConfig(logging=LoggingConfig(level="CRITICAL")).to_yaml(str(path))
        cfg = load_config(str(path))
        assert cfg.logging.level == "CRITICAL"

    def test_load_config_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("RFSN_LOG_LEVEL", "WARNING")
        cfg = load_config(None)
        assert cfg.logging.level == "WARNING"

    def test_get_config_singleton(self):
        set_config(RFSNConfig(logging=LoggingConfig(level="DEBUG")))
        cfg = get_config()
        assert cfg.logging.level == "DEBUG"
