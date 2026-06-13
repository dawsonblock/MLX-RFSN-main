"""Central MLX-LM API adapter.

All MLX-LM imports for the RFSN integration must come through this module.
It enforces the pinned MLX/MLX-LM version pair and provides a single point
of control for future version migrations.
"""
from __future__ import annotations

from typing import Any

from .compatibility import require_pinned_versions

try:
    require_pinned_versions()
    _VERSIONS_OK = True
except RuntimeError as _version_exc:
    _VERSIONS_OK = False
    _VERSION_ERROR = str(_version_exc)


def _ensure_versions() -> None:
    if not _VERSIONS_OK:
        raise RuntimeError(_VERSION_ERROR)


# ------------------------------------------------------------------
# Lazy imports validated against the pinned version
# ------------------------------------------------------------------

def import_mlx_core() -> Any:
    """Import and return ``mlx.core`` after version validation."""
    _ensure_versions()
    import mlx.core as mx
    return mx


def import_mlx_lm_utils() -> Any:
    """Import and return ``mlx_lm.utils`` after version validation."""
    _ensure_versions()
    import mlx_lm.utils
    return mlx_lm.utils


def import_mlx_lm_generate() -> Any:
    """Import ``generate`` after version validation."""
    _ensure_versions()
    try:
        from mlx_lm import generate
    except ImportError:
        from mlx_lm.utils import generate
    return generate


def import_mlx_lm_generate_step() -> Any:
    """Import ``generate_step`` after version validation."""
    _ensure_versions()
    try:
        from mlx_lm import generate_step
    except ImportError:
        from mlx_lm.utils import generate_step
    return generate_step


def import_mlx_lm_stream_generate() -> Any:
    """Import ``stream_generate`` after version validation."""
    _ensure_versions()
    try:
        from mlx_lm import stream_generate
    except ImportError:
        from mlx_lm.utils import stream_generate
    return stream_generate


def import_mlx_lm_make_sampler() -> Any:
    """Import ``make_sampler`` after version validation."""
    _ensure_versions()
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        from mlx_lm.utils import make_sampler
    return make_sampler


def import_mlx_lm_make_logits_processors() -> Any:
    """Import ``make_logits_processors`` after version validation."""
    _ensure_versions()
    try:
        from mlx_lm.sample_utils import make_logits_processors
    except ImportError:
        from mlx_lm.utils import make_logits_processors
    return make_logits_processors
