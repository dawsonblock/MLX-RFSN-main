"""RFSNRuntimeConfig — explicit API for native benchmark configuration.

Phase 3: Environment variables may override configuration for testing,
but they should not be the primary API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .backend_state import BackendReport, BackendState


@dataclass
class RFSNRuntimeConfig:
    """Canonical runtime configuration for RFSN native execution.

    Usage::

        cfg = RFSNRuntimeConfig(
            backend="metal_true_packed",
            strict_backend=True,
            model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        )
        report = cfg.probe_backend()
        if not report.is_ready():
            raise RuntimeError(report.reason)
    """

    backend: str = "metal_true_packed"
    strict_backend: bool = True
    model_id: str = ""
    context_lengths: list[int] = field(default_factory=lambda: [128, 512, 2048])
    output_tokens: int = 64
    key_bits: int = 8
    value_bits: int = 8
    group_size: int = 64
    staging_capacity: int = 64
    dense_residual_window: int = 0

    def probe_backend(self) -> BackendReport:
        """Probe the system and return a structured backend report.

        This is the single source of truth for whether native execution
        is possible.  It never silently falls back.
        """
        try:
            import mlx
            import mlx.core as mx
            _ = mx  # use it
        except ImportError:
            return BackendReport(
                state=BackendState.ABSENT,
                backend_name=self.backend,
                strict_mode=self.strict_backend,
                reason="MLX is not installed",
            )

        # Platform check: we need Metal on Apple Silicon
        try:
            has_metal = hasattr(mx, "metal") and mx.metal.is_available()
        except Exception:
            has_metal = False

        if not has_metal:
            return BackendReport(
                state=BackendState.UNSUPPORTED_PLATFORM,
                backend_name=self.backend,
                strict_mode=self.strict_backend,
                reason="MLX is installed but Metal is not available",
            )

        # Explicit opt-in check
        env_enable = os.environ.get("RFSN_ENABLE_TRUE_PACKED", "0") == "1"
        if not env_enable:
            return BackendReport(
                state=BackendState.DISABLED_BY_CONFIGURATION,
                backend_name=self.backend,
                strict_mode=self.strict_backend,
                reason="RFSN_ENABLE_TRUE_PACKED=1 is not set in environment",
            )

        # Delegate to the kernel module for compilation / self-test
        try:
            from rfsn_v10.kernels.metal.packed_v4_attention import (
                HAS_TRUE_PACKED_KERNEL,
                PackedV4AttentionKernel,
            )
        except Exception as exc:
            return BackendReport(
                state=BackendState.KERNEL_COMPILATION_FAILED,
                backend_name=self.backend,
                strict_mode=self.strict_backend,
                reason=f"Failed to import packed_v4_attention: {exc}",
            )

        if not HAS_TRUE_PACKED_KERNEL:
            return BackendReport(
                state=BackendState.SELF_TEST_FAILED,
                backend_name=self.backend,
                strict_mode=self.strict_backend,
                reason="PackedV4AttentionKernel self-test failed (compilation or numerical mismatch)",
            )

        # Gather provenance
        report = BackendReport(
            state=BackendState.READY,
            backend_name=self.backend,
            strict_mode=self.strict_backend,
        )

        # MLX version via importlib.metadata (mlx module has no __version__)
        try:
            from importlib.metadata import version as pkg_version
            report.mlx_version = pkg_version("mlx")
        except Exception:
            pass

        try:
            from importlib.metadata import version as pkg_version
            report.mlx_lm_version = pkg_version("mlx-lm")
        except Exception:
            pass

        try:
            import platform
            report.macos_version = platform.mac_ver()[0]
        except Exception:
            pass

        # Chip model and memory (peak_memory is allocated, not capacity)
        try:
            import mlx.core as mx
            dev = mx.metal.get_active_device()
            report.chip_model = str(dev) if dev is not None else ""
            report.memory_capacity_gb = round(
                mx.metal.get_peak_memory() / (1024 ** 3), 2
            )
        except Exception:
            pass

        # Kernel source hash — use the actual shader constant name
        try:
            from rfsn_v10.kernels.metal.packed_v4_attention import (
                _PACKED_V4_KERNEL_K8 as _src,
            )
            import hashlib
            report.kernel_source_hash = hashlib.sha256(
                _src.encode("utf-8")
            ).hexdigest()[:16]
        except Exception:
            pass

        # Configuration hash
        import hashlib
        config_str = (
            f"bits={self.key_bits}/{self.value_bits}"
            f"_gs={self.group_size}"
            f"_stage={self.staging_capacity}"
        )
        report.kernel_configuration_hash = hashlib.sha256(
            config_str.encode("utf-8")
        ).hexdigest()[:16]

        # Provenance
        try:
            import subprocess
            report.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True
            ).strip()
        except Exception:
            pass

        try:
            report.python_version = os.sys.version.split()[0]
        except Exception:
            pass

        try:
            import platform
            report.platform_machine = platform.machine()
        except Exception:
            pass

        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "strict_backend": self.strict_backend,
            "model_id": self.model_id,
            "context_lengths": self.context_lengths,
            "output_tokens": self.output_tokens,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "group_size": self.group_size,
            "staging_capacity": self.staging_capacity,
            "dense_residual_window": self.dense_residual_window,
        }
