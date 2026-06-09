"""Custom exception hierarchy for rfsn_v11.

All public errors raised by the package are subclasses of :class:`RFSNError`
so callers can catch the base class if they want to handle any rfsn_v11
failure uniformly.

Exception tree::

    RFSNError
    ├── ConfigurationError       — missing/invalid config or env vars
    ├── ModelNotLoadedError      — generator used before model is loaded
    ├── BackendError             — mlx / transformers backend failure
    ├── QuantizationError        — KV quantization / compression failure
    │   └── CodebookError        — codebook look-up or data corruption
    ├── CacheError               — prefix / disk cache failure
    │   ├── CacheVersionError    — CACHE_VERSION mismatch on load
    │   └── CacheEvictionError   — page eviction failed unexpectedly
    ├── KernelError              — Metal kernel dispatch failure
    │   └── KernelDimError       — unsupported head dimension D
    ├── SparsityError            — sparse decode / block-mask failure
    └── TelemetryError           — non-fatal telemetry / ClickHouse error
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class RFSNError(Exception):
    """Base class for all rfsn_v11 exceptions."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(RFSNError):
    """Raised when a required environment variable or config field is absent
    or has an invalid value."""


# ---------------------------------------------------------------------------
# Model / generator lifecycle
# ---------------------------------------------------------------------------

class ModelNotLoadedError(RFSNError):
    """Raised when a generation method is called before the model singleton
    has been initialised (e.g. ``RFSN_MODEL_ID`` not set)."""


class BackendError(RFSNError):
    """Raised when the selected inference backend (mlx-lm or transformers)
    encounters an unrecoverable error during model load or generation."""


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class QuantizationError(RFSNError):
    """Raised when KV quantization or decompression fails."""


class CodebookError(QuantizationError):
    """Raised when codebook data is missing, corrupt, or produces an
    out-of-range index."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class CacheError(RFSNError):
    """Raised when the prefix cache or disk store encounters an error."""


class CacheVersionError(CacheError):
    """Raised when a serialised cache record has a ``__cache_version__``
    that does not match the current :data:`rfsn_v11.cache.version.CACHE_VERSION`."""


class CacheEvictionError(CacheError):
    """Raised when the paged cache manager fails to evict a page to meet
    a memory-pressure request."""


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

class KernelError(RFSNError):
    """Raised when a Metal kernel fails to compile or dispatch."""


class KernelDimError(KernelError):
    """Raised when the head dimension ``D`` is not supported by any compiled
    kernel variant (currently only D=128 and D=256 are supported)."""


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

class SparsityError(RFSNError):
    """Raised when the sparse decode path or block-mask logic encounters an
    unrecoverable error."""


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetryError(RFSNError):
    """Raised when a telemetry event fails to serialise or emit.

    This is intentionally *not* a subclass of any other error category; callers
    should treat telemetry failures as non-fatal and log them rather than
    propagating them to users."""
