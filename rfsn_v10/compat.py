"""Compatibility helpers for optional MLX availability."""

from __future__ import annotations

try:
    import mlx.core as _mx

    MLX_AVAILABLE = True
    mx = _mx
except Exception:
    MLX_AVAILABLE = False

    class _MissingMLXModule:
        def __getattr__(self, name: str):
            raise AttributeError(name)

    mx = _MissingMLXModule()


def ensure_mlx_available() -> None:
    """Raise a clear error when MLX-only paths run without MLX installed."""
    if not MLX_AVAILABLE:
        raise ModuleNotFoundError(
            "mlx.core is required for this operation. Install MLX and run on a supported platform."
        )
