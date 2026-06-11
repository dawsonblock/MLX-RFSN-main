"""Strict JSON serialization helpers.

Ensures benchmark artifacts can never contain NaN, Infinity, or -Infinity.
"""
from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["dump_json_strict", "dumps_json_strict"]


def _sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def dump_json_strict(
    obj: Any,
    fp: Any,
    indent: int = 2,
    default: Any = str,
) -> None:
    """Write *obj* to *fp* as JSON with ``allow_nan=False``.

    Before writing, any ``NaN``, ``Infinity``, or ``-Infinity`` values
    are recursively replaced with ``None`` so the dump cannot fail.
    """
    sanitized = _sanitize_non_finite(obj)
    json.dump(sanitized, fp, indent=indent, default=default, allow_nan=False)


def dumps_json_strict(
    obj: Any,
    indent: int = 2,
    default: Any = str,
) -> str:
    """Return a JSON string for *obj* with ``allow_nan=False``.

    Non-finite floats are recursively replaced with ``None``.
    """
    sanitized = _sanitize_non_finite(obj)
    return json.dumps(
        sanitized, indent=indent, default=default, allow_nan=False,
    )
