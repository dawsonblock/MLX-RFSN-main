"""MLX-LM version compatibility checks.

The adapter is pinned to a specific MLX-LM version range.
Major version changes may break the cache interface.
"""
from __future__ import annotations

MIN_MLX_LM_VERSION = "0.21.0"


def check_mlx_lm_version() -> tuple[bool, str]:
    """Check if the installed mlx-lm version is compatible.

    Returns
    -------
    ok, msg
        ``ok`` is True if compatible, False otherwise.
        ``msg`` is a human-readable reason if not compatible.
    """
    try:
        import mlx_lm
    except ImportError:
        return False, "mlx-lm is not installed"
    try:
        from packaging import version
    except ImportError:
        return False, "packaging is required for version checks"

    installed = version.parse(mlx_lm.__version__)
    minimum = version.parse(MIN_MLX_LM_VERSION)

    if installed < minimum:
        return (
            False,
            f"mlx-lm {installed} < minimum {MIN_MLX_LM_VERSION}"
        )
    return True, f"mlx-lm {installed} is compatible"
