"""Cache policy abstraction for control, baseline, and promoted
KV-compression policies.

This module provides a clean internal abstraction so that candidate logic
does not leak into integration layers. Even if MLX-LM does not support
custom cache policies directly yet, this is the target interface.

Policies are split into three categories:
- CONTROL_POLICIES: maintained upstream paths (mlx_lm baseline / quantized)
- BASELINE_POLICIES: historically validated RFSN v10 configs
- PROMOTED_POLICIES: candidates that have passed full logit + memory gates

No candidate is added to PROMOTED_POLICIES until winner.json says so.

Example:
    from rfsn_v11.integrations.cache_policy import (
        CachePolicy, create_cache_policy,
    )

    policy = create_cache_policy("rfsn_v10_k8_v5_gs64")
    # Future: model.generate(prompt, cache_policy=policy)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachePolicy:
    """Describes a promoted cache policy for integration."""

    name: str
    candidate_name: str
    supports_real_generation: bool
    supports_prompt_cache: bool
    supports_streaming: bool
    supports_state_restore: bool
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.supports_real_generation:
            raise ValueError(
                f"CachePolicy '{self.name}' does not support real "
                "generation. Only promoted candidates with real cache "
                "injection can create a policy."
            )


# ---------------------------------------------------------------------------
# Policy registries
# ---------------------------------------------------------------------------

CONTROL_POLICIES: dict[str, dict[str, Any]] = {
    "mlx_lm_fp16": {
        "candidate_name": "mlx_lm_baseline",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {},
    },
    "mlx_lm_quantized_kv": {
        "candidate_name": "mlx_lm_quantized_kv_b8",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"kv_bits": 8, "kv_group_size": 64},
    },
}

BASELINE_POLICIES: dict[str, dict[str, Any]] = {
    "rfsn_v10_k8_v5_gs64": {
        "candidate_name": "rfsn_v10_k8_v5_gs64",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"default_bits": 8, "group_size": 64},
    },
}

LEGACY_POLICIES: dict[str, dict[str, Any]] = {
    "legacy_k8_v5_gs32": {
        "candidate_name": "legacy_k8_v5_gs32",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"default_bits": 8, "group_size": 32},
    },
}

PROMOTED_POLICIES: dict[str, dict[str, Any]] = {}

_KNOWN_POLICIES: dict[str, dict[str, Any]] = {
    **CONTROL_POLICIES,
    **BASELINE_POLICIES,
    **LEGACY_POLICIES,
    **PROMOTED_POLICIES,
}


def create_cache_policy(
    name: str, *, allow_experimental: bool = False, **overrides: Any
) -> CachePolicy:
    """Create a CachePolicy for a known candidate.

    Parameters
    ----------
    name
        Canonical policy name (e.g. "rfsn_v10_k8_v5_gs32").
    allow_experimental
        If True, allow creating a policy for an unpromoted experimental
        candidate. Default False — only control, baseline, and promoted
        policies are permitted.
    **overrides
        Optional overrides for policy fields.

    Raises
    ------
    ValueError
        If the policy name is unknown or the candidate does not support
        real generation.
    """
    if name not in _KNOWN_POLICIES:
        if not allow_experimental:
            raise ValueError(
                f"Unknown cache policy: {name!r}. "
                f"Known policies: {list(_KNOWN_POLICIES.keys())}. "
                f"Pass allow_experimental=True to allow unpromoted candidates."
            )
        # Experimental fallback — caller must provide full spec in overrides
        spec = dict(overrides)
        spec.setdefault("candidate_name", name)
        spec.setdefault("supports_real_generation", True)
        spec.setdefault("supports_prompt_cache", True)
        spec.setdefault("supports_streaming", True)
        spec.setdefault("supports_state_restore", False)
        spec.setdefault("config", {})
        return CachePolicy(name=name, **spec)

    spec = dict(_KNOWN_POLICIES[name])
    spec.update(overrides)
    return CachePolicy(name=name, **spec)


def list_policies() -> list[str]:
    """Return all known policy names."""
    return list(_KNOWN_POLICIES.keys())


def is_promoted_policy(name: str) -> bool:
    """Return True if the policy is in the promoted registry."""
    return name in PROMOTED_POLICIES
