"""Cache policy registry tests.

Ensures the policy registry correctly distinguishes control, baseline,
and promoted policies, and refuses unpromoted experimental candidates.
"""
from __future__ import annotations

import pytest

from rfsn_v11.integrations.cache_policy import (
    BASELINE_POLICIES,
    CONTROL_POLICIES,
    LEGACY_POLICIES,
    PROMOTED_POLICIES,
    create_cache_policy,
    is_promoted_policy,
    list_policies,
)


def test_control_policies_exist() -> None:
    assert "mlx_lm_fp16" in CONTROL_POLICIES
    assert "mlx_lm_quantized_kv" in CONTROL_POLICIES


def test_baseline_policies_contain_rfsn_v10() -> None:
    assert "rfsn_v10_k8_v5_gs64" in BASELINE_POLICIES
    assert "legacy_k8_v5_gs32" in LEGACY_POLICIES


def test_promoted_policies_empty_after_demotion() -> None:
    assert PROMOTED_POLICIES == {}


def test_turboquant_v2_not_in_promoted() -> None:
    assert "turboquant_v2_b4_gs64" not in PROMOTED_POLICIES
    assert "turboquant_v2_b4_gs64_rot" not in PROMOTED_POLICIES
    assert "turboquant_v2_b4_gs64_norot" not in PROMOTED_POLICIES


def test_rfsn_v11_not_in_promoted() -> None:
    assert "rfsn_v11_offline_asymmetric_kv_k8v4_gs64" not in PROMOTED_POLICIES


def test_create_known_control_policy() -> None:
    policy = create_cache_policy("mlx_lm_fp16")
    assert policy.name == "mlx_lm_fp16"
    assert policy.supports_real_generation is True


def test_create_known_baseline_policy() -> None:
    policy = create_cache_policy("rfsn_v10_k8_v5_gs64")
    assert policy.name == "rfsn_v10_k8_v5_gs64"
    assert policy.supports_real_generation is True


def test_create_legacy_policy() -> None:
    policy = create_cache_policy("legacy_k8_v5_gs32")
    assert policy.name == "legacy_k8_v5_gs32"
    assert policy.supports_real_generation is True


def test_create_unknown_policy_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_cache_policy("turboquant_v2_b4_gs64_norot")
    assert "Unknown cache policy" in str(exc_info.value)


def test_create_unknown_policy_with_allow_experimental() -> None:
    policy = create_cache_policy(
        "turboquant_v2_b4_gs64_norot",
        allow_experimental=True,
    )
    assert policy.name == "turboquant_v2_b4_gs64_norot"
    assert policy.supports_real_generation is True


def test_is_promoted_policy_false_for_rfsn_v10_after_demotion() -> None:
    assert is_promoted_policy("legacy_k8_v5_gs32") is False
    assert is_promoted_policy("rfsn_v10_k8_v5_gs64") is False


def test_is_promoted_policy_false_for_control() -> None:
    assert is_promoted_policy("mlx_lm_fp16") is False
    assert is_promoted_policy("mlx_lm_quantized_kv") is False


def test_is_promoted_policy_false_for_unknown() -> None:
    assert is_promoted_policy("turboquant_v2_b4_gs64_norot") is False


def test_list_policies_includes_control_and_promoted() -> None:
    policies = list_policies()
    assert "mlx_lm_fp16" in policies
    assert "mlx_lm_quantized_kv" in policies
    assert "legacy_k8_v5_gs32" in policies
    assert "rfsn_v10_k8_v5_gs64" in policies
