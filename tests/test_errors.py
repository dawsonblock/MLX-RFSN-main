#!/usr/bin/env python3
"""RFSN v10 — Error handling tests.

Covers error codes, structured error creation, exception classes,
and error handler aggregation without external dependencies.
"""
from __future__ import annotations

from rfsn_v10.errors import (
    AttentionException,
    ErrorCode,
    ErrorHandler,
    ErrorSeverity,
    KernelException,
    KVCacheException,
    MemoryException,
    PersistenceException,
    RFSNError,
    RFSNException,
    ValidationException,
    get_error_handler,
    handle_error,
)

# ------------------------------------------------------------------
# ErrorCode & ErrorSeverity
# ------------------------------------------------------------------

class TestErrorCode:
    def test_all_codes_have_numeric_values(self):
        for code in ErrorCode:
            assert isinstance(code.value, int)
            assert code.value >= 1000

    def test_kv_codes_in_1xxx_range(self):
        assert ErrorCode.KV_QUANTIZATION_FAILED.value == 1001
        assert ErrorCode.KV_CACHE_FULL.value == 1003

    def test_unknown_error_is_9000(self):
        assert ErrorCode.UNKNOWN_ERROR.value == 9000


class TestErrorSeverity:
    def test_severity_values(self):
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.CRITICAL.value == "critical"


# ------------------------------------------------------------------
# RFSNError dataclass
# ------------------------------------------------------------------

class TestRFSNError:
    def test_basic_creation(self):
        err = RFSNError(
            code=ErrorCode.KV_CACHE_FULL,
            message="Cache is full",
        )
        assert err.code == ErrorCode.KV_CACHE_FULL
        assert err.message == "Cache is full"
        assert err.severity == ErrorSeverity.ERROR
        assert err.recoverable is True
        assert err.timestamp is not None
        assert err.context == {}

    def test_to_dict_structure(self):
        err = RFSNError(
            code=ErrorCode.ATTENTION_KERNEL_FAILED,
            message="Kernel failed",
            severity=ErrorSeverity.CRITICAL,
            context={"layer": 3},
            stack_trace="traceback",
            recoverable=False,
        )
        d = err.to_dict()
        assert d["code"] == 2002
        assert d["code_name"] == "ATTENTION_KERNEL_FAILED"
        assert d["severity"] == "critical"
        assert d["recoverable"] is False
        assert d["context"] == {"layer": 3}
        assert d["stack_trace"] == "traceback"
        assert "timestamp" in d

    def test_auto_timestamp(self):
        err1 = RFSNError(code=ErrorCode.UNKNOWN_ERROR, message="test")
        err2 = RFSNError(code=ErrorCode.UNKNOWN_ERROR, message="test")
        # Both should have timestamps, likely different (or same instant)
        assert err1.timestamp is not None
        assert err2.timestamp is not None

    def test_default_context_is_empty_dict(self):
        err = RFSNError(code=ErrorCode.TIMEOUT, message="timed out")
        assert err.context == {}
        assert isinstance(err.context, dict)


# ------------------------------------------------------------------
# ErrorHandler
# ------------------------------------------------------------------

class TestErrorHandler:
    def test_handle_creates_rfsn_error(self):
        handler = ErrorHandler()
        err = handler.handle(ValueError("bad value"), code=ErrorCode.VALIDATION_MODEL_MISMATCH)
        assert isinstance(err, RFSNError)
        assert err.code == ErrorCode.VALIDATION_MODEL_MISMATCH
        assert "bad value" in err.message
        assert err.stack_trace is not None

    def test_error_counts_increment(self):
        handler = ErrorHandler()
        handler.handle(RuntimeError("a"), code=ErrorCode.KV_CACHE_CORRUPTED)
        handler.handle(RuntimeError("b"), code=ErrorCode.KV_CACHE_CORRUPTED)
        handler.handle(RuntimeError("c"), code=ErrorCode.ATTENTION_INVALID_INPUT)
        summary = handler.get_error_summary()
        assert summary["error_counts"]["KV_CACHE_CORRUPTED"] == 2
        assert summary["error_counts"]["ATTENTION_INVALID_INPUT"] == 1

    def test_error_history_limited_in_summary(self):
        handler = ErrorHandler()
        for i in range(15):
            handler.handle(RuntimeError(str(i)), code=ErrorCode.UNKNOWN_ERROR)
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 15
        assert len(summary["recent_errors"]) == 10

    def test_reset_clears_history(self):
        handler = ErrorHandler()
        handler.handle(RuntimeError("x"), code=ErrorCode.UNKNOWN_ERROR)
        handler.reset()
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 0
        assert summary["error_counts"] == {}

    def test_recoverable_codes(self):
        handler = ErrorHandler()
        recoverable = handler.handle(
            RuntimeError("x"), code=ErrorCode.ATTENTION_FALLBACK_TRIGGERED
        )
        unrecoverable = handler.handle(
            RuntimeError("x"), code=ErrorCode.MEMORY_ALLOCATION_FAILED
        )
        assert recoverable.recoverable is True
        assert unrecoverable.recoverable is False


# ------------------------------------------------------------------
# Global helpers
# ------------------------------------------------------------------

class TestGlobalHelpers:
    def test_get_error_handler_is_singleton(self):
        h1 = get_error_handler()
        h2 = get_error_handler()
        assert h1 is h2

    def test_handle_error_returns_rfsn_error(self):
        err = handle_error(RuntimeError("oops"), code=ErrorCode.KERNEL_EXECUTION_FAILED)
        assert isinstance(err, RFSNError)
        assert err.code == ErrorCode.KERNEL_EXECUTION_FAILED


# ------------------------------------------------------------------
# Exception classes
# ------------------------------------------------------------------

class TestExceptions:
    def test_rfsn_exception_message_format(self):
        exc = RFSNException(
            code=ErrorCode.KV_QUANTIZATION_FAILED,
            message="quant failed",
        )
        assert "KV_QUANTIZATION_FAILED" in str(exc)
        assert "quant failed" in str(exc)

    def test_subclass_exceptions(self):
        assert issubclass(AttentionException, RFSNException)
        assert issubclass(KVCacheException, RFSNException)
        assert issubclass(MemoryException, RFSNException)
        assert issubclass(KernelException, RFSNException)
        assert issubclass(PersistenceException, RFSNException)
        assert issubclass(ValidationException, RFSNException)

    def test_attention_exception_with_context(self):
        exc = AttentionException(
            code=ErrorCode.ATTENTION_SPARSE_QUALITY_DEGRADED,
            message="quality low",
            context={"layer": 2, "cosine": 0.85},
        )
        assert exc.context == {"layer": 2, "cosine": 0.85}
        assert "quality low" in str(exc)
