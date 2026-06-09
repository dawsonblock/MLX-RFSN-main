"""Shared symbols for the kernel backend package."""

from __future__ import annotations


class KernelRouteError(RuntimeError):
    """Raised when a requested reconstruction route cannot run."""
