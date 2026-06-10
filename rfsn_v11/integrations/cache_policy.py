"""Cache policy abstraction for promoted KV-compression candidates.

This module provides a clean internal abstraction so that candidate logic
does not leak into integration layers. Even if MLX-LM does not support
custom cache policies directly yet, this is the target interface.

Example:
    from rfsn_v11.integrations.cache_policy import CachePolicy, create_cache_policy

    policy = create_cache_policy("turboquant_v2_b4_gs64_rot")
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
                f"CachePolicy '{self.name}' does not support real generation. "
                "Only promoted candidates with real cache injection can create a policy."
            )


# ---------------------------------------------------------------------------
# Registry of known policies (populated as candidates are promoted)
# ---------------------------------------------------------------------------

_KNOWN_POLICIES: dict[str, dict[str, Any]] = {
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
    "rfsn_v10_k8_v5_gs32": {
        "candidate_name": "rfsn_v10_k8_v5_gs32",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"default_bits": 8, "group_size": 32},
    },
    "rfsn_v10_k8_v5_gs64": {
        "candidate_name": "rfsn_v10_k8_v5_gs64",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"default_bits": 8, "group_size": 64},
    },
    "turboquant_v2_b4_gs64_rot": {
        "candidate_name": "turboquant_v2_b4_gs64_rot",
        "supports_real_generation": True,
        "supports_prompt_cache": True,
        "supports_streaming": True,
        "supports_state_restore": False,
        "config": {"bits": 4, "group_size": 64, "use_rotation": True},
    },
}


def create_cache_policy(name: str, **overrides: Any) -> CachePolicy:
    """Create a CachePolicy for a known candidate.

    Parameters
    ----------
    name
        Canonical policy name (e.g. "turboquant_v2_b4_gs64_rot").
    **overrides
        Optional overrides for policy fields.

    Raises
    ------
    ValueError
        If the policy name is unknown or the candidate does not support
        real generation.
    """
    if name not in _KNOWN_POLICIES:
        raise ValueError(
            f"Unknown cache policy: {name!r}\n"
            f"Known policies: {list(_KNOWN_POLICIES.keys())}"
        )

    spec = dict(_KNOWN_POLICIES[name])
    spec.update(overrides)
    return CachePolicy(name=name, **spec)


def list_policies() -> list[str]:
    """Return all known policy names."""
    return list(_KNOWN_POLICIES.keys())
