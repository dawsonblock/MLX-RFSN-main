"""Metal kernel directory for polar_fused.

Kernels are compiled from .metal source files at runtime via mlx.core.fast.
Until custom Metal compilation is available, the Python wrappers fall back
to reference MLX implementations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# MLX optional at import time
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False

_METAL_DIR = Path(__file__).parent


def _load_metal_source(name: str) -> str:
    """Load a .metal shader source file."""
    path = _METAL_DIR / f"{name}.metal"
    if path.exists():
        return path.read_text()
    return ""
