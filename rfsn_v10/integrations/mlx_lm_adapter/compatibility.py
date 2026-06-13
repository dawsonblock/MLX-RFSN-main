"""MLX-LM version compatibility checks.

The adapter is pinned to exact MLX and MLX-LM versions.
No version range is used — any mismatch is a hard error.
"""
from __future__ import annotations

PINNED_MLX_VERSION = "0.21.1"
PINNED_MLX_LM_VERSION = "0.20.6"


def _get_installed_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", None)
        if version is not None:
            return str(version)
        core = getattr(mod, "core", None)
        if core is not None:
            return str(getattr(core, "__version__", None))
        return None
    except Exception:
        return None


def check_mlx_lm_version() -> tuple[bool, str]:
    """Check if the installed mlx and mlx-lm versions match the pinned pair.

    Returns
    -------
    ok, msg
        ``ok`` is True if both versions match exactly, False otherwise.
        ``msg`` is a human-readable reason if not compatible.
    """
    mlx_version = _get_installed_version("mlx")
    mlx_lm_version = _get_installed_version("mlx_lm")

    if mlx_version is None:
        return False, "mlx is not installed"
    if mlx_lm_version is None:
        return False, "mlx-lm is not installed"

    if mlx_version != PINNED_MLX_VERSION:
        return (
            False,
            f"mlx {mlx_version} != pinned {PINNED_MLX_VERSION}"
        )
    if mlx_lm_version != PINNED_MLX_LM_VERSION:
        return (
            False,
            f"mlx-lm {mlx_lm_version} != pinned {PINNED_MLX_LM_VERSION}"
        )

    return True, (
        f"mlx {mlx_version} + mlx-lm {mlx_lm_version} pinned pair verified"
    )


def require_pinned_versions() -> None:
    """Raise RuntimeError if the installed versions do not match exactly."""
    ok, msg = check_mlx_lm_version()
    if not ok:
        raise RuntimeError(
            f"Unsupported MLX/MLX-LM version: {msg}. "
            f"This release requires mlx=={PINNED_MLX_VERSION} and "
            f"mlx-lm=={PINNED_MLX_LM_VERSION}."
        )
