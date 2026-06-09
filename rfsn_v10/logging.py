#!/usr/bin/env python3
"""Comprehensive logging for RFSN v10.

Provides structured logging with multiple levels, context tracking,
and integration with error handling.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "context"):
            log_entry["context"] = record.context

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class RFSNLogger:
    """Structured logger for RFSN operations."""

    def __init__(
        self,
        name: str = "rfsn",
        level: int = logging.INFO,
        log_file: str | None = None,
        enable_json: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )

        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def _log(
        self,
        level: int,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log a message with optional context."""
        if context:
            extra = {"context": context}
            self.logger.log(level, message, extra=extra)
        else:
            self.logger.log(level, message)

    def debug(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, context)

    def info(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, context)

    def warning(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, context)

    def error(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, context)

    def critical(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, context)

    def exception(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        exc_info: Any = None,
    ) -> None:
        """Log exception with traceback."""
        self.logger.error(message, exc_info=exc_info, extra={"context": context or {}})


# Singleton logger instances
_loggers: dict[str, RFSNLogger] = {}


def get_logger(
    name: str = "rfsn",
    level: int = logging.INFO,
    log_file: str | None = None,
    enable_json: bool = True,
) -> RFSNLogger:
    """Get or create a logger instance."""
    key = f"{name}_{level}_{log_file}_{enable_json}"
    if key not in _loggers:
        _loggers[key] = RFSNLogger(name, level, log_file, enable_json)
    return _loggers[key]


class MetricsLogger:
    """Logger for metrics and telemetry."""

    def __init__(self, logger: RFSNLogger):
        self.logger = logger

    def log_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a metric."""
        context = {"metric_name": name, "metric_value": value}
        if tags:
            context["tags"] = tags
        self.logger.info(f"Metric: {name} = {value}", context=context)

    def log_counter(
        self,
        name: str,
        increment: int = 1,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a counter increment."""
        context = {"counter_name": name, "increment": increment}
        if tags:
            context["tags"] = tags
        self.logger.info(f"Counter: {name} += {increment}", context=context)

    def log_histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a histogram value."""
        context = {"histogram_name": name, "value": value}
        if tags:
            context["tags"] = tags
        self.logger.info(f"Histogram: {name} = {value}", context=context)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    enable_json: bool = True,
) -> RFSNLogger:
    """Setup global logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    return get_logger("rfsn", log_level, log_file, enable_json)
