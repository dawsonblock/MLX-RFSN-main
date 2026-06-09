"""Fused IsoQuant+Polar quantizer — experimental, not implemented in this build.

This module is a disabled stub. The fused IsoQuant-Polar quantization path
requires the ``enable_polar`` experimental flag and is intentionally not
implemented in the alpha/beta builds.

Attempting to instantiate these classes will raise RuntimeError. This keeps the
module compilable and importable while preventing silent activation of unvalidated
experimental code.
"""
from __future__ import annotations

from typing import Any


class _ExperimentalNotImplemented(RuntimeError):
    """Raised when an unimplemented experimental feature is accessed."""


class FusedIsoQuantPolar:
    """Stub fused IsoQuant+Polar quantizer.

    Not implemented — raises on any use.
    """

    def __init__(self, *args, **kwargs):
        raise _ExperimentalNotImplemented(
            "FusedIsoQuantPolar is not implemented in this build. "
            "This experimental fused quantizer is disabled by default. "
            "Set experimental.enable_polar=true to opt in."
        )

    def quantize(self, x: Any, *args, **kwargs) -> Any:
        raise _ExperimentalNotImplemented("FusedIsoQuantPolar.quantize not implemented")

    def dequantize(self, x: Any, *args, **kwargs) -> Any:
        raise _ExperimentalNotImplemented("FusedIsoQuantPolar.dequantize not implemented")
