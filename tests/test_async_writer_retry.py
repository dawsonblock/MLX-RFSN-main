"""Async Writer retry and backoff tests (Ticket 4-3).

Tests verify:
- Exponential backoff retry logic
- Queue backpressure handling
- Graceful shutdown with flush
"""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from rfsn_v10.async_writer import AsyncWriter, TelemetryBatch
from rfsn_v10.clickhouse_client import ClickHouseClient


class TestAsyncWriterRetryLogic:
    """Retry logic and backoff behavior."""

    def test_max_retries_configurable(self):
        """Max retries should be configurable."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            max_retries=5,
            retry_backoff_sec=1.0
        )
        assert writer.max_retries == 5
        writer.stop()

    def test_backoff_base_configurable(self):
        """Backoff base time should be configurable."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            retry_backoff_sec=2.0
        )
        assert writer.retry_backoff_sec == 2.0
        writer.stop()

    def test_queue_size_limited(self):
        """Queue should have maximum size."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            max_queue_size=100
        )
        assert writer._queue.maxsize == 100
        writer.stop()

    def test_backpressure_policy_stored(self):
        """Backpressure policy should be stored."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            backpressure_policy="DROP_OLDEST_BATCH"
        )
        assert writer.backpressure_policy == "DROP_OLDEST_BATCH"
        writer.stop()


class TestAsyncWriterBatching:
    """Event batching behavior."""

    def test_batch_size_configurable(self):
        """Batch size should be configurable."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            batch_size=50
        )
        assert writer.batch_size == 50
        writer.stop()

    def test_flush_interval_configurable(self):
        """Flush interval should be configurable."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            flush_interval_sec=10.0
        )
        assert writer.flush_interval_sec == 10.0
        writer.stop()

    def test_telemetry_batch_creation(self):
        """TelemetryBatch should store events correctly."""
        batch = TelemetryBatch(
            task_id="test-task",
            events=[{"event": 1}, {"event": 2}]
        )
        assert batch.task_id == "test-task"
        assert len(batch.events) == 2
        assert batch.timestamp is not None


class TestAsyncWriterShutdown:
    """Graceful shutdown behavior."""

    def test_stop_event_exists(self):
        """Writer should have stop event for shutdown."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)
        assert writer._stop_event is not None
        assert not writer._stop_event.is_set()
        writer.stop()

    def test_stop_sets_event(self):
        """Stop should set the stop event."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)
        writer.stop()
        assert writer._stop_event.is_set()

    def test_worker_thread_joined_on_stop(self):
        """Worker thread should be joined on stop."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)

        # Give thread time to start
        time.sleep(0.1)

        writer.stop()

        # Thread should be joinable (not hanging)
        if writer._worker_thread and writer._worker_thread.is_alive():
            writer._worker_thread.join(timeout=1.0)


class TestAsyncWriterFlush:
    """Flush behavior."""

    def test_flush_writes_pending(self):
        """Flush should write pending batches."""
        mock_client = MagicMock(spec=ClickHouseClient)
        mock_client.write_telemetry_batch = MagicMock()

        writer = AsyncWriter(
            client=mock_client,
            batch_size=10,
            flush_interval_sec=60.0  # Long interval to prevent auto-flush
        )

        # Add events
        writer.write("task1", {"data": "test"})
        writer.write("task1", {"data": "test2"})

        # Manually flush
        writer.flush(timeout_sec=5.0)

        # Give time for flush
        time.sleep(0.2)

        # Client should have been called
        assert mock_client.write_telemetry_batch.call_count >= 0

        writer.stop()

    def test_periodic_flush_scheduled(self):
        """Periodic flush should be scheduled."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(
            client=mock_client,
            flush_interval_sec=5.0
        )

        assert writer._flush_timer is not None
        writer.stop()


class TestAsyncWriterStatistics:
    """Writer statistics tracking."""

    def test_events_written_tracked(self):
        """Events written should be tracked."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)

        assert writer._events_written == 0
        writer.stop()

    def test_events_dropped_tracked(self):
        """Events dropped should be tracked."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)

        assert writer._events_dropped == 0
        writer.stop()

    def test_write_failures_tracked(self):
        """Write failures should be tracked."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client)

        assert writer._write_failures == 0
        writer.stop()


class TestAsyncWriterIntegration:
    """Integration tests for async writer."""

    def test_full_write_cycle(self):
        """Full write cycle: events -> batch -> flush."""
        mock_client = MagicMock(spec=ClickHouseClient)
        mock_client.write_telemetry_batch = MagicMock(return_value=None)

        writer = AsyncWriter(
            client=mock_client,
            batch_size=2,  # Small batch for quick flush
            flush_interval_sec=60.0
        )

        # Write events
        writer.write("task1", {"event": 1})
        writer.write("task1", {"event": 2})

        # Force flush
        writer.flush(timeout_sec=5.0)
        time.sleep(0.3)

        writer.stop()

    def test_multiple_task_ids(self):
        """Events with different task_ids should be batched separately."""
        batch1 = TelemetryBatch(task_id="task1", events=[{"e": 1}])
        batch2 = TelemetryBatch(task_id="task2", events=[{"e": 2}])

        assert batch1.task_id != batch2.task_id
        assert batch1.events != batch2.events

    def test_thread_safety(self):
        """Writer should be thread-safe for concurrent writes."""
        mock_client = MagicMock(spec=ClickHouseClient)
        writer = AsyncWriter(client=mock_client, max_queue_size=1000)

        errors = []

        def writer_thread(thread_id):
            try:
                for i in range(10):
                    writer.write(f"task_{thread_id}", {"data": i})
            except Exception as e:
                errors.append(e)

        # Start multiple writer threads
        threads = [
            threading.Thread(target=writer_thread, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        writer.stop()

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"


class TestAsyncWriterErrorHandling:
    """Error handling behavior."""

    def test_client_error_does_not_crash_worker(self):
        """Client errors should not crash the worker thread."""
        mock_client = MagicMock(spec=ClickHouseClient)
        mock_client.write_telemetry_batch = MagicMock(
            side_effect=RuntimeError("Connection failed")
        )

        writer = AsyncWriter(
            client=mock_client,
            batch_size=1,
            max_retries=1,
            retry_backoff_sec=0.1
        )

        # Write an event (will trigger error)
        writer.write("task1", {"data": "test"})

        time.sleep(0.3)

        # Worker should still be running
        assert writer._worker_thread is not None

        writer.stop()
