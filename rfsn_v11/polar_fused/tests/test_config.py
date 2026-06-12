"""Tests for PolarFusedConfig."""
from __future__ import annotations

import pytest

from rfsn_v11.polar_fused.config import PolarFusedConfig


def test_default_config_is_valid() -> None:
    cfg = PolarFusedConfig()
    assert cfg.key_bits == 4
    assert cfg.value_bits == 4
    assert cfg.head_dim == 128


def test_polar_safe_profile() -> None:
    cfg = PolarFusedConfig.polar_safe()
    assert cfg.key_bits == 4
    assert cfg.value_bits == 4
    assert cfg.boundary_layers == 2
    assert not cfg.enable_sparse_v


def test_polar_balanced_profile() -> None:
    cfg = PolarFusedConfig.polar_balanced()
    assert cfg.key_bits == 4
    assert cfg.value_bits == 3
    assert cfg.boundary_layers == 2


def test_polar_aggressive_profile() -> None:
    cfg = PolarFusedConfig.polar_aggressive()
    assert cfg.key_bits == 3
    assert cfg.value_bits == 3
    assert cfg.boundary_layers == 4


def test_invalid_key_bits() -> None:
    with pytest.raises(ValueError, match="key_bits must be"):
        PolarFusedConfig(key_bits=5)


def test_invalid_value_bits() -> None:
    with pytest.raises(ValueError, match="value_bits must be"):
        PolarFusedConfig(value_bits=1)


def test_invalid_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be"):
        PolarFusedConfig(head_dim=256)


def test_non_positive_allocation_block() -> None:
    with pytest.raises(ValueError, match="allocation_block_tokens must be positive"):
        PolarFusedConfig(allocation_block_tokens=0)


def test_negative_lazy_threshold() -> None:
    with pytest.raises(ValueError, match="lazy_quantization_tokens cannot be negative"):
        PolarFusedConfig(lazy_quantization_tokens=-1)


def test_sparse_v_rejected() -> None:
    with pytest.raises(ValueError, match="enable_sparse_v is not supported"):
        PolarFusedConfig(enable_sparse_v=True)


def test_rigidity_reuse_rejected() -> None:
    with pytest.raises(ValueError, match="enable_rigidity_reuse is not supported"):
        PolarFusedConfig(enable_rigidity_reuse=True)


def test_config_is_frozen() -> None:
    import dataclasses
    cfg = PolarFusedConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.key_bits = 3  # type: ignore[misc]
