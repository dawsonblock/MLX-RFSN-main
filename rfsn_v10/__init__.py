"""RFSN v10 package surface with lazy MLX-dependent imports."""

from __future__ import annotations

from importlib import import_module

try:
    from ._version import __version__
except Exception:  # pragma: no cover
    __version__ = "unknown"

from .compat import MLX_AVAILABLE

_LAZY_IMPORTS = {
    "BitPackedQuantizer": ".bitpack",
    "RFSNTurboQuantKVManager": ".kv_manager",
    "TurboQuantKVCache": ".kv_manager",
    "AdaptiveBlockSparseAttention": ".attention",
    "ExecutionMode": ".attention",
    "AdaptiveSparsityController": ".adaptive_sparsity",
    "AdaptiveDecision": ".adaptive_sparsity",
    "MemoryGuard": ".memory_guard",
    "AsyncWriter": ".async_writer",
    "ClickHouseClient": ".clickhouse_client",
    "KernelRouteError": ".kernels",
}

# RFSNRuntime and TelemetryEvent are exported from the runtime package.
_RUNTIME_IMPORTS = {
    "RFSNRuntime": ".runtime",
    "TelemetryEvent": ".runtime",
}

_ALL_LAZY = {**_LAZY_IMPORTS, **_RUNTIME_IMPORTS}

__all__ = ["MLX_AVAILABLE", "__version__", *_ALL_LAZY.keys()]


def __getattr__(name: str):
    module_name = _ALL_LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
