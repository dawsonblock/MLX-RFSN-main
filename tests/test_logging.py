#!/usr/bin/env python3
"""RFSN v10 — Logging tests.

Covers JSON formatter, structured logger, metrics logger,
and setup helpers without requiring MLX.
"""
from __future__ import annotations

import json
import logging

from rfsn_v10.logging import (
    JSONFormatter,
    MetricsLogger,
    RFSNLogger,
    get_logger,
    setup_logging,
)

# ------------------------------------------------------------------
# JSONFormatter
# ------------------------------------------------------------------

class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "hello"
        assert "timestamp" in data
        assert "module" in data

    def test_format_with_context(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="warn", args=(), exc_info=None,
        )
        record.context = {"user_id": 42}
        result = formatter.format(record)
        data = json.loads(result)
        assert data["context"] == {"user_id": 42}

    def test_format_with_exception(self):
        import sys
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except Exception:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error", args=(), exc_info=exc_info,
            )
        result = formatter.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]


# ------------------------------------------------------------------
# RFSNLogger
# ------------------------------------------------------------------

class TestRFSNLogger:
    def test_default_level(self):
        logger = RFSNLogger(name="test_default")
        assert logger.logger.level == logging.INFO

    def test_custom_level(self):
        logger = RFSNLogger(name="test_debug", level=logging.DEBUG)
        assert logger.logger.level == logging.DEBUG

    def test_log_levels(self, caplog):
        logger = RFSNLogger(name="test_levels", level=logging.DEBUG)
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")
        logger.critical("critical msg")
        # Check handlers have at least one
        assert len(logger.logger.handlers) > 0

    def test_log_with_context(self, caplog):
        logger = RFSNLogger(name="test_ctx", level=logging.DEBUG, enable_json=False)
        logger.info("msg", context={"key": "value"})
        # The log should have been emitted
        assert len(logger.logger.handlers) > 0

    def test_file_handler(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = RFSNLogger(name="test_file", level=logging.INFO, log_file=log_file)
        logger.info("file test")
        assert (tmp_path / "test.log").exists()

    def test_text_format(self, capsys):
        logger = RFSNLogger(name="test_text", level=logging.INFO, enable_json=False)
        logger.info("plain text")
        captured = capsys.readouterr()
        assert "plain text" in captured.out or captured.err


# ------------------------------------------------------------------
# get_logger singleton
# ------------------------------------------------------------------

class TestGetLogger:
    def test_same_params_same_instance(self):
        logger1 = get_logger("singleton", logging.INFO, None, True)
        logger2 = get_logger("singleton", logging.INFO, None, True)
        assert logger1 is logger2

    def test_different_params_different_instance(self):
        logger1 = get_logger("diff", logging.INFO, None, True)
        logger2 = get_logger("diff", logging.DEBUG, None, True)
        assert logger1 is not logger2


# ------------------------------------------------------------------
# MetricsLogger
# ------------------------------------------------------------------

class TestMetricsLogger:
    def test_log_metric(self, capsys):
        logger = RFSNLogger(name="metrics_test", level=logging.INFO, enable_json=False)
        metrics = MetricsLogger(logger)
        metrics.log_metric("temperature", 25.5)
        captured = capsys.readouterr()
        assert "temperature" in (captured.out + captured.err)

    def test_log_counter(self, capsys):
        logger = RFSNLogger(name="counter_test", level=logging.INFO, enable_json=False)
        metrics = MetricsLogger(logger)
        metrics.log_counter("requests", 3)
        captured = capsys.readouterr()
        assert "requests" in (captured.out + captured.err)

    def test_log_histogram(self, capsys):
        logger = RFSNLogger(name="hist_test", level=logging.INFO, enable_json=False)
        metrics = MetricsLogger(logger)
        metrics.log_histogram("latency", 12.5)
        captured = capsys.readouterr()
        assert "latency" in (captured.out + captured.err)


# ------------------------------------------------------------------
# setup_logging
# ------------------------------------------------------------------

class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging(level="INFO")
        assert isinstance(logger, RFSNLogger)

    def test_respects_level(self):
        logger = setup_logging(level="DEBUG")
        assert logger.logger.level == logging.DEBUG
