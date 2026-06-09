"""IsoQuant quaternion preconditioner — experimental, not implemented in this build.

This module is a disabled stub. The IsoQuant preconditioning path requires the
``enable_polar`` experimental flag and is intentionally not implemented in the
alpha/beta builds.

Attempting to instantiate these classes will raise RuntimeError. This keeps the
module compilable and importable while preventing silent activation of unvalidated
experimental code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _ExperimentalNotImplemented(RuntimeError):
    """Raised when an unimplemented experimental feature is accessed."""


@dataclass
class IsoQuantMetadata:
    """Stub metadata container for IsoQuant preconditioned keys.

    Not implemented — raises on any use.
    """
    signs: Any = None
    rotation: Any = None

    def __post_init__(self):
        raise _ExperimentalNotImplemented(
            "IsoQuantMetadata is not implemented in this build. "
            "The IsoQuant preconditioner requires experimental opt-in "
            "(enable_polar=true) and is disabled by default."
        )


class IsoQuantPreconditioner:
    """Stub IsoQuant quaternion preconditioner.

    Not implemented — raises on any use.
    """

    def __init__(self, *args, **kwargs):
        raise _ExperimentalNotImplemented(
            "IsoQuantPreconditioner is not implemented in this build. "
            "Set experimental.enable_polar=true to opt in, but note that "
            "this feature is not validated for production use."
        )

    def forward(self, x, *args, **kwargs):
        raise _ExperimentalNotImplemented("IsoQuantPreconditioner.forward not implemented")

    def inverse(self, x, *args, **kwargs):
        raise _ExperimentalNotImplemented("IsoQuantPreconditioner.inverse not implemented")
