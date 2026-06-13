"""Tests for canonical KV configuration."""
from __future__ import annotations

import pytest

from rfsn_v10.config import (
    CANONICAL_KV_CONFIG,
    CanonicalKVConfig,
    require_canonical_candidate,
)


def test_canonical_config_defaults() -> None:
    assert CANONICAL_KV_CONFIG.key_bits == 8
    assert CANONICAL_KV_CONFIG.value_bits == 5
    assert CANONICAL_KV_CONFIG.group_size == 64
    assert CANONICAL_KV_CONFIG.block_size == 64
    assert CANONICAL_KV_CONFIG.use_wht is True
    assert CANONICAL_KV_CONFIG.sign_seed == 42
    assert CANONICAL_KV_CONFIG.sign_algorithm == "splitmix64-v1"
    assert CANONICAL_KV_CONFIG.dense_residual_window == 0
    assert CANONICAL_KV_CONFIG.format_version == 4
    assert CANONICAL_KV_CONFIG.tensor_layout == "BHTD"
    assert CANONICAL_KV_CONFIG.packing_layout == "VECTOR_ALIGNED_UINT32"
    assert CANONICAL_KV_CONFIG.scale_layout == "BHTG"
    assert CANONICAL_KV_CONFIG.sparse_decode is False
    assert CANONICAL_KV_CONFIG.qjl_enabled is False
    assert CANONICAL_KV_CONFIG.polar_enabled is False


def test_require_canonical_accepts_exact() -> None:
    require_canonical_candidate(CANONICAL_KV_CONFIG)


def test_require_canonical_rejects_different() -> None:
    non_canonical = CanonicalKVConfig(key_bits=4)
    with pytest.raises(ValueError, match="Only the canonical"):
        require_canonical_candidate(non_canonical)
