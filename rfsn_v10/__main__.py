"""Entry point for ``python -m rfsn_v10``.

Subcommands
-----------
healthcheck         Run all health checks and report status.
version             Print package, Python, and backend information.
validate-config     Validate a YAML config file against the strict schema.

Usage examples::

    python -m rfsn_v10 healthcheck
    python -m rfsn_v10 version
    python -m rfsn_v10 validate-config --config configs/default_runtime.yaml
    RFSN_BACKEND=numpy python -m rfsn_v10 healthcheck
"""

from __future__ import annotations

import os
import sys


def _cmd_healthcheck() -> int:
    """Run all health checks and report status.

    Health semantics:
    - healthy   : requested backend available and all checks pass.  Exit 0.
    - degraded  : optional dependency missing, selected mode does not require it.
                  Exit 2.
    - unhealthy : selected backend unavailable or config invalid.  Exit 1.
    """
    from rfsn_v10.health import HealthStatus, get_health_checker
    from rfsn_v10.compat import MLX_AVAILABLE

    backend = os.environ.get("RFSN_BACKEND", "mlx").lower()

    checker = get_health_checker()
    checker.run_all_checks()
    report = checker.get_health_report()

    overall_status = report.get("overall_status", "unknown")
    timestamp = report.get("timestamp", 0.0)

    print(f"RFSN v10 healthcheck  (backend={backend})")
    print(f"  overall: {overall_status}")
    print(f"  timestamp: {timestamp:.3f}")
    for name, item in report.get("checks", {}).items():
        s = item.get("status", "unknown") if isinstance(item, dict) else str(item)
        msg = item.get("message", "") if isinstance(item, dict) else ""
        print(f"  {name}: {s}" + (f" — {msg}" if msg else ""))

    # Backend-aware exit code
    if backend == "mlx" and not MLX_AVAILABLE:
        print(
            "\nERROR: RFSN_BACKEND=mlx but MLX is not installed. "
            "Install mlx or set RFSN_BACKEND=numpy.",
            file=sys.stderr,
        )
        return 1

    if overall_status == HealthStatus.HEALTHY.value:
        return 0
    elif overall_status == HealthStatus.DEGRADED.value:
        if backend == "numpy":
            # numpy backend degraded (e.g. MLX unavailable) is acceptable
            return 0
        return 2
    else:
        return 1


def _cmd_version() -> int:
    """Print package, Python, and backend information."""
    import platform

    from rfsn_v10.compat import MLX_AVAILABLE

    try:
        from rfsn_v10._version import version as pkg_version
    except ImportError:
        pkg_version = "unknown (not installed from package)"

    backend = os.environ.get("RFSN_BACKEND", "mlx").lower()
    mlx_version = "not installed"
    if MLX_AVAILABLE:
        try:
            import mlx
            mlx_version = getattr(mlx, "__version__", "installed (version unknown)")
        except Exception:
            mlx_version = "installed (version unknown)"

    print(f"rfsn-v10 version   : {pkg_version}")
    print(f"Python             : {sys.version}")
    print(f"Platform           : {platform.platform()}")
    print(f"RFSN_BACKEND       : {backend}")
    print(f"MLX available      : {MLX_AVAILABLE}")
    print(f"MLX version        : {mlx_version}")
    return 0


def _cmd_validate_config(config_path: str | None) -> int:
    """Validate a YAML config file against the strict schema."""
    from rfsn_v10.config import load_config

    if config_path is None:
        print("ERROR: --config PATH is required for validate-config.", file=sys.stderr)
        return 1

    try:
        cfg = load_config(config_path)
        print(f"OK: config valid — {config_path}")
        print(f"  quant mode    : {cfg.runtime.default_quant_mode}")
        print(f"  experimental  : {cfg.runtime.allow_experimental}")
        print(f"  sparse decode : {cfg.runtime.sparse_decode_enabled}")
        print(f"  audit mode    : {cfg.runtime.audit_enabled}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Config validation failed — {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Dispatch CLI subcommands."""
    args = sys.argv[1:]

    if not args:
        # Default: healthcheck for backwards compatibility
        return _cmd_healthcheck()

    subcommand = args[0]

    if subcommand == "healthcheck":
        return _cmd_healthcheck()

    if subcommand == "version":
        return _cmd_version()

    if subcommand == "validate-config":
        config_path = None
        remaining = args[1:]
        for i, arg in enumerate(remaining):
            if arg == "--config" and i + 1 < len(remaining):
                config_path = remaining[i + 1]
                break
        return _cmd_validate_config(config_path)

    print(
        f"Unknown subcommand: {subcommand!r}\n"
        "Available subcommands: healthcheck, version, validate-config",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
