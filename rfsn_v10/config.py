#!/usr/bin/env python3
"""Configuration management for RFSN v10.

Provides configuration schema, environment variable support,
and YAML file loading for production deployment.
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
    """Quantization configuration.

    Production configuration constrained to 2-8 bits.
    K16 diagnostic configurations should use KVCodecConfig with reference_only=True.
    """

    model_config = ConfigDict(extra="forbid")

    default_bits: int = Field(default=8, ge=2, le=8)  # Fix #7: Restore production cap to 8
    group_size: int = Field(default=64, ge=1)
    enable_wht: bool = Field(default=True)
    enable_incoherent_signs: bool = Field(default=True)


class KVCodecConfig(BaseModel):
    """Separate runtime key-codec bit width from legacy QuantizationConfig.

    This allows asymmetric K/V configurations (e.g., K16/V8) and diagnostic
    K16 configurations without conflicting with the legacy default_bits constraint.

    Fix #7: Use this for K16 diagnostic configurations with reference_only=True.
    """

    model_config = ConfigDict(extra="forbid")

    key_bits: int = Field(default=8, ge=2, le=16)
    value_bits: int = Field(default=8, ge=2, le=16)
    group_size: int = Field(default=64, ge=1)
    diagnostic_raw_codes: bool = Field(default=False)  # If True, classified as DIAGNOSTIC_REFERENCE_ONLY


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default="",
        description="Model identifier or local path",
    )


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


class ServerConfig(BaseModel):
    """FastAPI server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", description="Bind host (127.0.0.1 for local-only, 0.0.0.0 for LAN)")
    port: int = Field(default=8000, ge=1, le=65535)
    require_api_key: bool = Field(default=False, description="Require Authorization: Bearer <key>")
    api_key: str = Field(default="", description="API key (required when require_api_key=True)")
    max_prompt_chars: int = Field(default=24000, ge=1)
    max_tokens_limit: int = Field(default=4096, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    enable_dashboard: bool = Field(default=True, description="Serve /dashboard static HTML")
    enable_docs: bool = Field(default=True, description="Serve /docs and /redoc")
    max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        description="Max simultaneous generation requests (RFSN_MAX_CONCURRENT_REQUESTS)",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v, info):
        # Delay require_api_key check to model_post_init
        return v

    def model_post_init(self, __context) -> None:
        if self.require_api_key and not self.api_key:
            raise ValueError(
                "ServerConfig.api_key must be set when require_api_key=True. "
                "Set RFSN_API_KEY in environment."
            )
        if self.host == "0.0.0.0" and not self.require_api_key:
            raise ValueError(
                "LAN mode (host=0.0.0.0) requires API key enforcement. "
                "Pass --require-api-key and --api-key, or set "
                "RFSN_REQUIRE_API_KEY=true and RFSN_API_KEY=<key>."
            )


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
    enable_kv_compression: bool = Field(
        default=False,
        description="Enable v10 KV compression (RFSN_ENABLE_KV_COMPRESSION). "
        "Deprecated alias: RFSN_ENABLE_QUANTIZED_KV.",
    )
    packed_reference: bool = Field(
        default=False,
        description="Use direct packed-reference attention (no dense reconstruction).",
    )
    strict_packed_mode: bool = Field(
        default=False,
        description="Require packed cache and reject fallback to dense attention.",
    )


def _resolve_kv_compression_env() -> bool:
    """Resolve KV compression flag with compat alias.

    Canonical name: ``RFSN_ENABLE_KV_COMPRESSION``.
    Deprecated alias: ``RFSN_ENABLE_QUANTIZED_KV`` — accepted but emits a
    :class:`DeprecationWarning` so operators can migrate.
    """
    import warnings as _w

    new_val = os.getenv("RFSN_ENABLE_KV_COMPRESSION")
    old_val = os.getenv("RFSN_ENABLE_QUANTIZED_KV")

    if old_val is not None and new_val is None:
        _w.warn(
            "RFSN_ENABLE_QUANTIZED_KV is deprecated. "
            "Use RFSN_ENABLE_KV_COMPRESSION instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return old_val.lower() == "true"

    return (new_val or "false").lower() == "true"


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
    model: ModelConfig = Field(default_factory=ModelConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
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
            model=ModelConfig(
                id=os.getenv("RFSN_MODEL_ID", ""),
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
            server=ServerConfig(
                host=os.getenv("RFSN_HOST", "127.0.0.1"),
                port=int(os.getenv("RFSN_PORT", "8000")),
                require_api_key=(
                    os.getenv("RFSN_REQUIRE_API_KEY", "false").lower() == "true"
                ),
                api_key=os.getenv("RFSN_API_KEY", ""),
                max_prompt_chars=int(os.getenv("RFSN_MAX_PROMPT_CHARS", "24000")),
                max_tokens_limit=int(os.getenv("RFSN_MAX_TOKENS_LIMIT", "4096")),
                request_timeout_seconds=int(os.getenv("RFSN_REQUEST_TIMEOUT_SECONDS", "120")),
                enable_dashboard=(
                    os.getenv("RFSN_ENABLE_DASHBOARD", "true").lower() == "true"
                ),
                enable_docs=(
                    os.getenv("RFSN_ENABLE_DOCS", "true").lower() == "true"
                ),
                max_concurrent_requests=int(
                    os.getenv("RFSN_MAX_CONCURRENT_REQUESTS", "1")
                ),
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
                enable_kv_compression=_resolve_kv_compression_env(),
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


# ---------------------------------------------------------------------------
# Canonical KV configuration (promotion-eligible format)
# ---------------------------------------------------------------------------

class CanonicalKVConfig(BaseModel):
    """Canonical KV configuration for promotion-eligible runs.

    Only one candidate format is promotion-eligible:
      * K8 grouped symmetric keys
      * V5 grouped symmetric values
      * group_size = 64
      * block_size = 64 tokens
      * WHT-64 preconditioner + deterministic integer-hash signs
      * Vector-aligned uint32 packing (V4)
      * BHTG scales
    """

    model_config = ConfigDict(frozen=True)

    key_bits: int = Field(default=8)
    value_bits: int = Field(default=5)

    group_size: int = Field(default=64)
    block_size: int = Field(default=64)

    use_wht: bool = Field(default=True)
    sign_seed: int = Field(default=42)
    sign_algorithm: str = Field(default="splitmix64-v1")

    dense_residual_window: int = Field(default=0)

    format_version: int = Field(default=4)
    tensor_layout: str = Field(default="BHTD")
    packing_layout: str = Field(default="VECTOR_ALIGNED_UINT32")
    scale_layout: str = Field(default="BHTG")

    sparse_decode: bool = Field(default=False)
    qjl_enabled: bool = Field(default=False)
    polar_enabled: bool = Field(default=False)


CANONICAL_KV_CONFIG = CanonicalKVConfig()


def require_canonical_candidate(config: CanonicalKVConfig) -> None:
    if config != CANONICAL_KV_CONFIG:
        raise ValueError(
            "Only the canonical K8/V5/WHT-64 configuration is promotion eligible"
        )
