"""Backend state reporting for RFSN native execution.

Phase 3: Replace silent fallback with explicit, discoverable backend states.
Environment variables may override configuration for testing, but they
should not be the primary API.
"""
from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from typing import Any


class BackendState(StrEnum):
    """Exactly one state reported at startup.

    Ordered from least to most ready:
    - ABSENT: MLX not installed (portable CI, Linux CPU)
    - DISABLED_BY_CONFIGURATION: present but turned off in config
    - UNSUPPORTED_PLATFORM: MLX present but no Metal (Linux with MLX)
    - KERNEL_COMPILATION_FAILED: Metal present but shader compile error
    - SELF_TEST_FAILED: Compiled but numerical self-test failed
    - READY: Everything passes, kernel available for dispatch
    """

    ABSENT = "absent"
    DISABLED_BY_CONFIGURATION = "disabled_by_configuration"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    KERNEL_COMPILATION_FAILED = "kernel_compilation_failed"
    SELF_TEST_FAILED = "self_test_failed"
    READY = "ready"


@dataclass
class BackendReport:
    """Structured report of the native backend status."""

    state: BackendState
    backend_name: str = ""
    strict_mode: bool = False
    kernel_source_hash: str = ""
    kernel_configuration_hash: str = ""
    mlx_version: str = ""
    mlx_lm_version: str = ""
    macos_version: str = ""
    chip_model: str = ""
    memory_capacity_gb: float = 0.0
    reason: str = ""  # Human-readable detail when not READY
    # Provenance (audit fix: reproducibility)
    git_commit: str = ""
    benchmark_source_hash: str = ""
    prompt_token_hash: str = ""
    python_version: str = ""
    platform_machine: str = ""

    def is_ready(self) -> bool:
        return self.state == BackendState.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "backend_name": self.backend_name,
            "strict_mode": self.strict_mode,
            "kernel_source_hash": self.kernel_source_hash,
            "kernel_configuration_hash": self.kernel_configuration_hash,
            "mlx_version": self.mlx_version,
            "mlx_lm_version": self.mlx_lm_version,
            "macos_version": self.macos_version,
            "chip_model": self.chip_model,
            "memory_capacity_gb": self.memory_capacity_gb,
            "reason": self.reason,
            "git_commit": self.git_commit,
            "benchmark_source_hash": self.benchmark_source_hash,
            "prompt_token_hash": self.prompt_token_hash,
            "python_version": self.python_version,
            "platform_machine": self.platform_machine,
        }
