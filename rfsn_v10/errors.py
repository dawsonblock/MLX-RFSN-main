#!/usr/bin/env python3
"""Structured error codes and handling for RFSN v10.

Provides comprehensive error classification, structured error codes,
and error handling utilities for production deployment.
"""

from __future__ import annotations

import enum
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class ErrorCode(enum.Enum):
    """Structured error codes for RFSN operations."""

    # KV Cache Errors (1xxx)
    KV_QUANTIZATION_FAILED = 1001
    KV_DEQUANTIZATION_FAILED = 1002
    KV_CACHE_FULL = 1003
    KV_CACHE_CORRUPTED = 1004
    KV_INVALID_SHAPE = 1005

    # Attention Errors (2xxx)
    ATTENTION_INVALID_INPUT = 2001
    ATTENTION_KERNEL_FAILED = 2002
    ATTENTION_FALLBACK_TRIGGERED = 2003
    ATTENTION_SPARSE_QUALITY_DEGRADED = 2004

    # Memory Errors (3xxx)
    MEMORY_ALLOCATION_FAILED = 3001
    MEMORY_QUOTA_EXCEEDED = 3002
    MEMORY_LEAK_DETECTED = 3003
    MEMORY_OOM = 3004

    # Kernel Errors (4xxx)
    KERNEL_COMPILATION_FAILED = 4001
    KERNEL_EXECUTION_FAILED = 4002
    KERNEL_UNSUPPORTED_HARDWARE = 4003

    # Persistence Errors (5xxx)
    PERSISTENCE_WRITE_FAILED = 5001
    PERSISTENCE_READ_FAILED = 5002
    PERSISTENCE_RECOVERY_FAILED = 5003
    PERSISTENCE_QUOTA_EXCEEDED = 5004

    # Validation Errors (6xxx)
    VALIDATION_QUALITY_THRESHOLD = 6001
    VALIDATION_REGRESSION = 6002
    VALIDATION_MODEL_MISMATCH = 6003

    # General Errors (9xxx)
    UNKNOWN_ERROR = 9000
    TIMEOUT = 9001
    CANCELLED = 9002


class ErrorSeverity(enum.Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RFSNError:
    """Structured error information."""

    code: ErrorCode
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: datetime = None
    context: dict[str, Any] = None
    stack_trace: str | None = None
    recoverable: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
        }


class ErrorHandler:
    """Centralized error handling and logging."""

    def __init__(self):
        self.error_history: list[RFSNError] = []
        self.error_counts: dict[ErrorCode, int] = {}

    def handle(
        self,
        error: Exception,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: dict[str, Any] | None = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> RFSNError:
        """Handle an exception and create structured error."""
        stack_trace = traceback.format_exc()

        rfsn_error = RFSNError(
            code=code,
            message=str(error),
            severity=severity,
            context=context or {},
            stack_trace=stack_trace,
            recoverable=self._is_recoverable(code),
        )

        self.error_history.append(rfsn_error)
        self.error_counts[code] = self.error_counts.get(code, 0) + 1

        return rfsn_error

    def _is_recoverable(self, code: ErrorCode) -> bool:
        """Determine if an error is recoverable."""
        recoverable_codes = {
            ErrorCode.ATTENTION_FALLBACK_TRIGGERED,
            ErrorCode.ATTENTION_SPARSE_QUALITY_DEGRADED,
            ErrorCode.KERNEL_UNSUPPORTED_HARDWARE,
            ErrorCode.MEMORY_QUOTA_EXCEEDED,
        }
        return code in recoverable_codes

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of recent errors."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": {code.name: count for code, count in self.error_counts.items()},
            "recent_errors": [e.to_dict() for e in self.error_history[-10:]],
        }

    def reset(self) -> None:
        """Reset error history."""
        self.error_history.clear()
        self.error_counts.clear()


# Singleton instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


def handle_error(
    error: Exception,
    code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    context: dict[str, Any] | None = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
) -> RFSNError:
    """Handle an error using the global error handler."""
    return get_error_handler().handle(error, code, context, severity)


class RFSNException(Exception):
    """Base exception for RFSN errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        context: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.context = context or {}
        super().__init__(f"[{code.name}] {message}")


class KVCacheException(RFSNException):
    """Exception for KV cache errors."""

    pass


class AttentionException(RFSNException):
    """Exception for attention errors."""

    pass


class MemoryException(RFSNException):
    """Exception for memory errors."""

    pass


class KernelException(RFSNException):
    """Exception for kernel errors."""

    pass


class PersistenceException(RFSNException):
    """Exception for persistence errors."""

    pass


class ValidationException(RFSNException):
    """Exception for validation errors."""

    pass
