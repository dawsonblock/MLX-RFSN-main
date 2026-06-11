#!/usr/bin/env python3
"""Configuration management for RFSN v11.

Provides configuration schema, environment variable support,
and YAML file loading for production deployment.

Ported from rfsn_v10/config.py with ExperimentalConfig extended to include
three new gates: qjl_prod, sub4bit_small_head, isoquant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="json", description="Log format (json or text)"
    )
    file: str | None = Field(default=None, description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class MemoryConfig(BaseModel):
    """Memory management configuration."""

    model_config = ConfigDict(extra="forbid")

    max_gb: float = Field(
        default=8.0, ge=0.1, description="Maximum memory in GB"
    )
    quota_gb: float = Field(
        default=10.0, ge=0.1, description="Disk quota in GB"
    )
    enable_leak_detection: bool = Field(
        default=True, description="Enable leak detection"
    )


class CacheConfig(BaseModel):
    """Cache configuration."""

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(
        default="~/.cache/rfsn_v11", description="Cache directory"
    )
    enable_persistence: bool = Field(
        default=True, description="Enable disk persistence"
    )
    enable_wal: bool = Field(
        default=True, description="Enable write-ahead logging"
    )


class SparseAttentionConfig(BaseModel):
    """Sparse attention configuration."""

    model_config = ConfigDict(extra="forbid")

    default_top_k_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    block_size: int = Field(default=64, ge=1)
    enable_adaptive: bool = Field(default=True)


class QuantizationConfig(BaseModel):
    """Quantization configuration."""

    model_config = ConfigDict(extra="forbid")

    default_bits: int = Field(default=8, ge=2, le=8)
    group_size: int = Field(default=64, ge=1)
    enable_wht: bool = Field(default=True)
    enable_incoherent_signs: bool = Field(default=True)
    # Asymmetric K/V defaults
    k_bits: int = Field(default=8, ge=2, le=8, description="Key quantization bits")
    v_bits: int = Field(default=4, ge=2, le=8, description="Value quantization bits")


class BackendConfig(BaseModel):
    """Kernel backend configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["", "auto", "metal", "numpy", "mlx"] = Field(
        default="",
        description="Backend override (metal|numpy|mlx). "
        "Empty string or 'auto' lets the dispatcher choose.",
    )


class TelemetryConfig(BaseModel):
    """ClickHouse telemetry configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=8123, ge=1, le=65535)
    secure: bool = Field(default=True)
    auth_token: str = Field(default="")
    database: str = Field(default="default")


class ExperimentalConfig(BaseModel):
    """Opt-in gates for experimental / unvalidated features.

    All experimental paths are disabled by default.  Enabling any of them
    emits a loud warning at runtime because they have not been validated
    for production or quality-critical generation.

    v11 additions vs v10:
      - enable_qjl_prod: TurboQuant QJL as the primary bit-quantizer path
          (never enable by default — centroid resolution failure proven)
      - enable_sub4bit_small_head: v_bits < 4 on models with D < 64
          (cosine similarity degrades below safety threshold)
      - enable_isoquant: IsoQuant/hybrid-polar-cartesian paths from v10
          (not validated for v11 asymmetric K/V stack)
    """

    model_config = ConfigDict(extra="forbid")

    # v10 gates (preserved)
    enable_qjl: bool = Field(default=False)
    enable_polar: bool = Field(default=False)
    enable_adaptive: bool = Field(default=False)

    # v11 new gates
    enable_qjl_prod: bool = Field(
        default=False,
        description="TurboQuant QJL as primary quantizer (unsafe — off by default)",
    )
    enable_sub4bit_small_head: bool = Field(
        default=False,
        description="Allow v_bits < 4 on models with head_dim < 64",
    )
    enable_isoquant: bool = Field(
        default=False,
        description="IsoQuant/hybrid-polar-cartesian paths (unvalidated in v11)",
    )

    # Alpha 9 TurboPolar gates (all disabled by default — EXPERIMENTAL)
    enable_turbo_polar_offline: bool = Field(
        default=False,
        description="TurboPolar offline PolarQuant encoder/decoder (EXPERIMENTAL)",
    )
    enable_turbo_polar_qjl: bool = Field(
        default=False,
        description="TurboPolar QJL residual score correction (EXPERIMENTAL)",
    )
    enable_turbo_polar_metal: bool = Field(
        default=False,
        description="TurboPolar fused Metal dequant-QK + attention kernels (EXPERIMENTAL_METAL)",
    )


class RuntimeConfig(BaseModel):
    """Runtime flags."""

    model_config = ConfigDict(extra="forbid")

    default_quant_mode: str = Field(default="k8_v5_gs64")
    allow_experimental: bool = Field(default=False)
    qjl_enabled: bool = Field(default=False)
    sparse_decode_enabled: bool = Field(default=False)
    audit_enabled: bool = Field(default=True)


class RFSNConfig(BaseModel):
    """Main RFSN v11 configuration."""

    model_config = ConfigDict(extra="forbid")

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    sparse_attention: SparseAttentionConfig = Field(
        default_factory=SparseAttentionConfig
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig
    )
    backend: BackendConfig = Field(default_factory=BackendConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)

    @classmethod
    def from_env(cls) -> "RFSNConfig":
        """Load configuration from environment variables."""
        return cls(
            logging=LoggingConfig(
                level=os.getenv("RFSN_LOG_LEVEL", "INFO"),
                format=os.getenv("RFSN_LOG_FORMAT", "json"),
                file=os.getenv("RFSN_LOG_FILE"),
            ),
            memory=MemoryConfig(
                max_gb=float(os.getenv("RFSN_MAX_MEMORY_GB", "8.0")),
                quota_gb=float(os.getenv("RFSN_QUOTA_GB", "10.0")),
                enable_leak_detection=(
                    os.getenv("RFSN_ENABLE_LEAK_DETECTION", "true").lower()
                    == "true"
                ),
            ),
            cache=CacheConfig(
                directory=os.getenv("RFSN_CACHE_DIR", "~/.cache/rfsn_v11"),
                enable_persistence=(
                    os.getenv("RFSN_ENABLE_PERSISTENCE", "true").lower()
                    == "true"
                ),
                enable_wal=(
                    os.getenv("RFSN_ENABLE_WAL", "true").lower() == "true"
                ),
            ),
            backend=BackendConfig(
                name=os.getenv("RFSN_BACKEND", ""),
            ),
            telemetry=TelemetryConfig(
                host=os.getenv("RFSN_CLICKHOUSE_HOST", "localhost"),
                port=int(os.getenv("RFSN_CLICKHOUSE_PORT", "8123")),
                secure=(
                    os.getenv("RFSN_CLICKHOUSE_SECURE", "true").lower()
                    == "true"
                ),
                auth_token=os.getenv("RFSN_CLICKHOUSE_TOKEN", ""),
                database=os.getenv("RFSN_CLICKHOUSE_DB", "default"),
            ),
            runtime=RuntimeConfig(
                default_quant_mode=os.getenv(
                    "RFSN_DEFAULT_QUANT_MODE", "k8_v5_gs64"
                ),
                allow_experimental=(
                    os.getenv("RFSN_ALLOW_EXPERIMENTAL", "false").lower()
                    == "true"
                ),
                qjl_enabled=(
                    os.getenv("RFSN_QJL_ENABLED", "false").lower() == "true"
                ),
                sparse_decode_enabled=(
                    os.getenv("RFSN_SPARSE_DECODE_ENABLED", "false").lower()
                    == "true"
                ),
                audit_enabled=(
                    os.getenv("RFSN_AUDIT_ENABLED", "true").lower()
                    == "true"
                ),
            ),
            experimental=ExperimentalConfig(
                enable_qjl=(
                    os.getenv("RFSN_EXPERIMENTAL_QJL", "false").lower()
                    == "true"
                ),
                enable_polar=(
                    os.getenv("RFSN_EXPERIMENTAL_POLAR", "false").lower()
                    == "true"
                ),
                enable_adaptive=(
                    os.getenv("RFSN_EXPERIMENTAL_ADAPTIVE", "false").lower()
                    == "true"
                ),
                enable_qjl_prod=(
                    os.getenv("RFSN_EXPERIMENTAL_QJL_PROD", "false").lower()
                    == "true"
                ),
                enable_sub4bit_small_head=(
                    os.getenv(
                        "RFSN_EXPERIMENTAL_SUB4BIT_SMALL_HEAD", "false"
                    ).lower()
                    == "true"
                ),
                enable_isoquant=(
                    os.getenv("RFSN_EXPERIMENTAL_ISOQUANT", "false").lower()
                    == "true"
                ),
                enable_turbo_polar_offline=(
                    os.getenv("RFSN_EXPERIMENTAL_TURBO_POLAR_OFFLINE", "false").lower()
                    == "true"
                ),
                enable_turbo_polar_qjl=(
                    os.getenv("RFSN_EXPERIMENTAL_TURBO_POLAR_QJL", "false").lower()
                    == "true"
                ),
                enable_turbo_polar_metal=(
                    os.getenv("RFSN_EXPERIMENTAL_TURBO_POLAR_METAL", "false").lower()
                    == "true"
                ),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RFSNConfig":
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_config: RFSNConfig | None = None


def load_config(path: str | None = None) -> RFSNConfig:
    """Load configuration from YAML or environment."""
    if path:
        return RFSNConfig.from_yaml(path)
    return RFSNConfig.from_env()


def get_config() -> RFSNConfig:
    """Return the global configuration instance, loading from env if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: RFSNConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def require_experimental(feature: str, config: RFSNConfig | None = None) -> None:
    """Raise RuntimeError if *feature* is not enabled in experimental config.

    Call this at the top of any code path that activates an experimental
    feature so that it cannot silently activate on a stable runtime.

    Args:
        feature: One of the supported experimental feature names.
        config:  Config to check.  Uses the global config when *None*.

    Raises:
        RuntimeError: If the requested experimental feature is not enabled.
        ValueError: If *feature* is not a known gate name.
    """
    import warnings

    cfg = config or get_config()
    exp = cfg.experimental

    enabled = {
        # v10 gates
        "qjl": exp.enable_qjl,
        "polar": exp.enable_polar,
        "adaptive": exp.enable_adaptive,
        # v11 new gates
        "qjl_prod": exp.enable_qjl_prod,
        "sub4bit_small_head": exp.enable_sub4bit_small_head,
        "isoquant": exp.enable_isoquant,
        # Alpha 9 TurboPolar gates
        "turbo_polar_offline": exp.enable_turbo_polar_offline,
        "turbo_polar_qjl": exp.enable_turbo_polar_qjl,
        "turbo_polar_metal": exp.enable_turbo_polar_metal,
    }

    if feature not in enabled:
        raise ValueError(
            f"Unknown experimental feature {feature!r}.  "
            f"Valid values: {sorted(enabled)}"
        )

    if not enabled[feature]:
        raise RuntimeError(
            f"Experimental feature '{feature}' is disabled.  "
            f"Set experimental.enable_{feature}=true in config or "
            f"RFSN_EXPERIMENTAL_{feature.upper()}=true in environment to enable.  "
            "Warning: experimental features are not validated for production "
            "or quality-critical generation."
        )

    warnings.warn(
        f"Experimental mode enabled: '{feature}' is not validated for "
        "production or quality-critical generation.",
        stacklevel=3,
    )
