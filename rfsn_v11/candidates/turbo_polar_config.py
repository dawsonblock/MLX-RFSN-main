"""TurboPolar candidate configuration.

Frozen dataclass with sensible defaults. All paths start disabled.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurboPolarConfig:
    """Immutable configuration for a TurboPolar candidate run.

    Defaults correspond to the first experimental config:
      k_angle_bits_level1 = 4
      k_angle_bits_deep   = 2  (deep levels disabled by default)
      v_bits              = 8  (dense fp16 values at start)
      block_size          = 64
      head_dim            = 128
      qjl_proj_dim        = 64
      use_qjl             = False
      use_metal           = False
      storage_mode        = "k_only_first"
    """

    k_angle_bits_level1: int = 4
    k_angle_bits_deep: int = 2
    v_bits: int = 8
    block_size: int = 64
    head_dim: int = 128
    qjl_proj_dim: int = 64
    use_qjl: bool = False
    use_metal: bool = False
    storage_mode: str = "k_only_first"

    # Candidate identity (set at construction time)
    candidate_name: str = "turbo_polar_k4_qjl64"
    candidate_status: str = "EXPERIMENTAL"
    promotion_eligible: bool = False
    gate_status: str = "PENDING_LOGIT_GATE"

    def with_qjl(self, enabled: bool = True) -> "TurboPolarConfig":
        """Return a new config with QJL toggled."""
        return TurboPolarConfig(
            k_angle_bits_level1=self.k_angle_bits_level1,
            k_angle_bits_deep=self.k_angle_bits_deep,
            v_bits=self.v_bits,
            block_size=self.block_size,
            head_dim=self.head_dim,
            qjl_proj_dim=self.qjl_proj_dim,
            use_qjl=enabled,
            use_metal=self.use_metal,
            storage_mode=self.storage_mode,
            candidate_name=self.candidate_name,
            candidate_status=self.candidate_status,
            promotion_eligible=self.promotion_eligible,
            gate_status=self.gate_status,
        )

    def with_metal(self, enabled: bool = True) -> "TurboPolarConfig":
        """Return a new config with Metal toggled."""
        return TurboPolarConfig(
            k_angle_bits_level1=self.k_angle_bits_level1,
            k_angle_bits_deep=self.k_angle_bits_deep,
            v_bits=self.v_bits,
            block_size=self.block_size,
            head_dim=self.head_dim,
            qjl_proj_dim=self.qjl_proj_dim,
            use_qjl=self.use_qjl,
            use_metal=enabled,
            storage_mode=self.storage_mode,
            candidate_name=self.candidate_name,
            candidate_status=self.candidate_status,
            promotion_eligible=self.promotion_eligible,
            gate_status=self.gate_status,
        )
