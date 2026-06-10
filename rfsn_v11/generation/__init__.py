"""Minimal benchmark-only generation loop for RFSN v11 cache injection.

This module provides a controlled decode loop that avoids monkey-patching
MLX-LM globally. It is intended for benchmark use only.
"""
from __future__ import annotations

from .minimal_decode import minimal_decode_loop
from .cache_injection import RFSNV11CacheInjector

__all__ = ["minimal_decode_loop", "RFSNV11CacheInjector"]
