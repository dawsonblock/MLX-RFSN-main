#!/usr/bin/env python3
"""Configuration management for RFSN v10.

Provides configuration schema, environment variable support,
and YAML file loading for production deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal


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
        default="~/.cache/rfsn", description="Cache directory"
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


class BackendConfig(BaseModel):
    """Kernel backend configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["", "auto", "metal", "numpy", "mlx"] = Field(
        default="",
        description="Backend override (metal|numpy|mlx). "
        "Empty string or 'auto' lets the dispatcher choose. "
        "CUDA is not implemented.",
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
    """

    model_config = ConfigDict(extra="forbid")

    enable_qjl: bool = Field(default=False)
    enable_polar: bool = Field(default=False)
    enable_adaptive: bool = Field(default=False)


class RuntimeConfig(BaseModel):
    """Runtime flags matching default_runtime.yaml."""

    model_config = ConfigDict(extra="forbid")

    default_quant_mode: str = Field(default="k8_v5_gs64")
    allow_experimental: bool = Field(default=False)
    qjl_enabled: bool = Field(default=False)
    sparse_decode_enabled: bool = Field(default=False)
    audit_enabled: bool = Field(default=True)


class RFSNConfig(BaseModel):
    """Main RFSN configuration."""

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
    def from_env(cls) -> RFSNConfig:
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
                directory=os.getenv("RFSN_CACHE_DIR", "~/.cache/rfsn"),
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
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> RFSNConfig:
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        config_path = Path(path).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_config(path: str | None = None) -> RFSNConfig:
    """Load configuration from file or environment.

    Args:
        path: Optional path to YAML config file.  When *path* is provided the
              file **must** exist; a :exc:`FileNotFoundError` is raised if it
              does not.  When *path* is ``None`` the config is loaded from
              environment variables.

    Returns:
        RFSNConfig instance
    """
    if path:
        # Explicit path: require the file to exist.  Do not silently fall back
        # to environment variables when the caller specified a config file.
        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return RFSNConfig.from_yaml(path)
    return RFSNConfig.from_env()


# Global config instance
_config: RFSNConfig | None = None


def get_config() -> RFSNConfig:
    """Get the global configuration instance."""
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

    Call this at the top of any code path that activates QJL, polar, or
    adaptive features so that they cannot silently activate on a stable
    runtime.

    Args:
        feature: One of ``"qjl"``, ``"polar"``, or ``"adaptive"``.
        config:  Config to check.  Uses the global config when *None*.

    Raises:
        RuntimeError: If the requested experimental feature is not enabled.
    """
    import warnings

    cfg = config or get_config()
    exp = cfg.experimental

    enabled = {
        "qjl": exp.enable_qjl,
        "polar": exp.enable_polar,
        "adaptive": exp.enable_adaptive,
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
