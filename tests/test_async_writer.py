from __future__ import annotations

from rfsn_v10.async_writer import AsyncWriter


class DummyClient:
    def __init__(self):
        self.batches: list[list[dict]] = []
        self.closed = False

    def write_telemetry_batch(self, events):
        self.batches.append(list(events))

    def close(self):
        self.closed = True


def test_async_writer_stop_flushes_pending_events():
    client = DummyClient()
    writer = AsyncWriter(
        client=client,
        batch_size=10,
        flush_interval_sec=60.0,
        max_queue_size=100,
    )

    writer.write("task-1", {"id": 1})
    writer.write("task-1", {"id": 2})

    writer.stop(timeout_sec=2.0)

    assert client.closed is True
    assert len(client.batches) >= 1
    all_ids = [event["id"] for batch in client.batches for event in batch]
    assert 1 in all_ids
    assert 2 in all_ids


def test_async_writer_stop_is_idempotent():
    client = DummyClient()
    writer = AsyncWriter(client=client, flush_interval_sec=60.0)

    writer.write("task-1", {"id": 1})
    writer.stop(timeout_sec=2.0)
    writer.stop(timeout_sec=2.0)

    assert client.closed is True
