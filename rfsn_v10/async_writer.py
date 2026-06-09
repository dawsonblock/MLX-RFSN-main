#!/usr/bin/env python3
"""
RFSN v10 - Async Telemetry Writer.

Background worker that batches and writes metrics to ClickHouse with retry logic,
backpressure handling, and ordered flushing per task_id.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .clickhouse_client import ClickHouseClient


@dataclass
class TelemetryBatch:
    """Batch of telemetry events to write together."""
    task_id: str
    events: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AsyncWriter:
    """
    Background telemetry writer with batching, retries, and backpressure handling.

    Features:
    - Batches writes by task_id for ordering
    - Retries transient errors with exponential backoff
    - Respects max queue size to prevent unbounded memory growth
    - Flushes on shutdown with timeout
    - Emits warnings on backpressure
    """

    def __init__(
        self,
        client: ClickHouseClient,
        batch_size: int = 100,
        flush_interval_sec: float = 5.0,
        max_queue_size: int = 10000,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
        backpressure_policy: str = "DISCARD_CURRENT_BATCH",
    ):
        """
        Args:
            client: ClickHouseClient instance for database writes.
            batch_size: Max events per batch before forcing flush.
            flush_interval_sec: Maximum time to wait before flushing batch.
            max_queue_size: Maximum number of batches in queue.
            max_retries: Number of retry attempts for failed writes.
            retry_backoff_sec: Base backoff time between retries (seconds).
            backpressure_policy: Behavior when queue is full. Supported values:
                "DISCARD_CURRENT_BATCH" (default), "DROP_OLDEST_BATCH".
        """
        self.client = client
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.backpressure_policy = backpressure_policy.upper()

        # Threading primitives
        self._queue: queue.Queue[TelemetryBatch | None] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._flush_timer: threading.Timer | None = None
        self._current_batch: TelemetryBatch | None = None
        self._lock = threading.RLock()

        # Statistics
        self._events_written = 0
        self._events_dropped = 0
        self._write_failures = 0
        self._last_warning_time = 0.0

        self._start()

    def _start(self) -> None:
        """Start the background worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncTelemetryWriter",
            daemon=True
        )
        self._worker_thread.start()
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Schedule the next periodic flush."""
        if self._stop_event.is_set():
            return

        self._flush_timer = threading.Timer(
            self.flush_interval_sec,
            self._flush_timeout
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush_timeout(self) -> None:
        """Called when flush interval expires."""
        self._flush_current_batch()
        self._schedule_flush()

    def _flush_current_batch(self) -> None:
        """Flush the current batch if it has events."""
        with self._lock:
            if self._current_batch and self._current_batch.events:
                batch_to_write = self._current_batch
                self._current_batch = TelemetryBatch(
                    task_id=batch_to_write.task_id,
                    events=[],
                    timestamp=batch_to_write.timestamp,
                )
                # Queue the batch for writing (non-blocking)
                try:
                    self._queue.put_nowait(batch_to_write)
                except queue.Full:
                    if self.backpressure_policy == "DROP_OLDEST_BATCH":
                        try:
                            dropped_batch = self._queue.get_nowait()
                            if dropped_batch is not None:
                                self._events_dropped += len(dropped_batch.events)
                            self._queue.put_nowait(batch_to_write)
                            self._warn_backpressure("queue full; dropped oldest batch")
                        except queue.Empty:
                            self._events_dropped += len(batch_to_write.events)
                            self._warn_backpressure("queue full during batch flush")
                        except queue.Full:
                            self._events_dropped += len(batch_to_write.events)
                            self._warn_backpressure("queue still full after dropping oldest batch")
                    else:
                        # Default policy: discard the current batch.
                        self._events_dropped += len(batch_to_write.events)
                        self._warn_backpressure("queue full during batch flush")

    def _warn_backpressure(self, reason: str) -> None:
        """Emit a backpressure warning, rate-limited to once per 10 seconds."""
        now = time.time()
        if now - self._last_warning_time > 10.0:
            import warnings
            warnings.warn(f"Telemetry backpressure: {reason}")
            self._last_warning_time = now

    def write(self, task_id: str, event: dict) -> None:
        """
        Write a telemetry event (non-blocking).

        Args:
            task_id: Identifier for ordering preservation.
            event: Telemetry event dictionary to write.
        """
        # Check if we need to start a new batch for this task_id
        with self._lock:
            if (self._current_batch is None or
                self._current_batch.task_id != task_id or
                len(self._current_batch.events) >= self.batch_size):
                self._flush_current_batch()
                self._current_batch = TelemetryBatch(
                    task_id=task_id,
                    events=[],
                    timestamp=datetime.now(timezone.utc),
                )

            self._current_batch.events.append(event)

            # If batch is full, flush it immediately
            if len(self._current_batch.events) >= self.batch_size:
                self._flush_current_batch()

    def _worker_loop(self) -> None:
        """Main worker loop: processes batches from the queue."""
        while True:
            try:
                # Get batch with timeout to allow checking stop_event when queue is empty
                batch = self._queue.get(timeout=0.1)
                if batch is None:  # Sentinel value for shutdown
                    break

                self._write_batch_with_retries(batch)
                self._queue.task_done()
            except queue.Empty:
                # If queue is empty and we're supposed to stop, exit
                if self._stop_event.is_set():
                    break
                continue
            except Exception as e:
                # Log but don't crash the worker thread
                import warnings
                warnings.warn(f"Telemetry worker error: {e}")

    def _write_batch_with_retries(self, batch: TelemetryBatch) -> None:
        """Write a batch with retry logic for transient errors."""
        if not batch.events:
            return

        delay = self.retry_backoff_sec
        for attempt in range(self.max_retries + 1):
            try:
                self.client.write_telemetry_batch(batch.events)
                self._events_written += len(batch.events)
                return  # Success
            except Exception as e:
                if attempt == self.max_retries:
                    # Final attempt failed
                    self._events_dropped += len(batch.events)
                    self._write_failures += 1
                    import warnings
                    warnings.warn(
                        f"Failed to write telemetry batch after {self.max_retries + 1} attempts: {e}"
                    )
                    return

                # Transient error - wait and retry
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    def flush(self, timeout_sec: float = 10.0) -> None:
        """
        Flush all pending telemetry and stop the writer.

        Args:
            timeout_sec: Maximum time to wait for flush completion.
        """
        self.stop(timeout_sec)

    def stop(self, timeout_sec: float = 10.0) -> None:
        """
        Stop the background worker and flush remaining events.

        Args:
            timeout_sec: Maximum time to wait for shutdown.
        """
        if self._stop_event.is_set():
            return

        deadline = time.monotonic() + timeout_sec
        self._stop_event.set()

        # Cancel flush timer
        if self._flush_timer:
            self._flush_timer.cancel()

        # Flush current batch
        with self._lock:
            if self._current_batch and self._current_batch.events:
                self._flush_current_batch()

        # Send sentinel to wake up worker after all pending batches are queued.
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))

        # Wait for worker to finish processing the sentinel and any queued items
        if self._worker_thread:
            remaining = max(0.0, deadline - time.monotonic())
            self._worker_thread.join(timeout=remaining)

        # Close client
        self.client.close()

    def get_stats(self) -> dict:
        """Return writer statistics."""
        with self._lock:
            queue_size = self._queue.qsize()
            current_batch_size = len(self._current_batch.events) if self._current_batch else 0

            return {
                "events_written": self._events_written,
                "events_dropped": self._events_dropped,
                "write_failures": self._write_failures,
                "queue_size": queue_size,
                "current_batch_size": current_batch_size,
                "is_alive": self._worker_thread.is_alive() if self._worker_thread else False,
            }
