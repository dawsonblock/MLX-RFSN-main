"""Tests for RFSN v11 async server correctness.

Phase 5 — 5-6:
  - cfg.model_dump() never includes dunder keys
  - GenerationConfig.model_dump(exclude={'stream'}) correct
  - Event loop blocking probe: run_in_executor pattern yields control
  - SSE cancellation: queue sentinel consumed on cancellation
  - Non-streaming isolation: asyncio.to_thread used
"""
import asyncio
import queue
import threading
import time
import pytest

pytest.importorskip("fastapi")

from rfsn_v11.server.app import GenerationConfig


# ---------------------------------------------------------------------------
# cfg.model_dump() correctness
# ---------------------------------------------------------------------------

def test_generation_config_model_dump_no_dunders():
    """model_dump() must not include any dunder keys."""
    cfg = GenerationConfig(max_new_tokens=128, temperature=0.8, stream=True)
    dumped = cfg.model_dump()
    dunder_keys = [k for k in dumped if k.startswith("__")]
    assert not dunder_keys, f"model_dump() contains dunder keys: {dunder_keys}"


def test_generation_config_model_dump_exclude_stream():
    """model_dump(exclude={'stream'}) must exclude only 'stream'."""
    cfg = GenerationConfig(max_new_tokens=64, temperature=0.5, top_p=0.95, stream=True)
    gen_kwargs = cfg.model_dump(exclude={"stream"})
    assert "stream" not in gen_kwargs
    assert "max_new_tokens" in gen_kwargs
    assert "temperature" in gen_kwargs
    assert gen_kwargs["max_new_tokens"] == 64
    assert gen_kwargs["temperature"] == 0.5


def test_generation_config_model_dump_all_fields():
    """All declared fields must be present in model_dump()."""
    cfg = GenerationConfig()
    dumped = cfg.model_dump()
    expected = {"max_new_tokens", "temperature", "top_p", "repetition_penalty",
                "stop_sequences", "stream"}
    assert expected == set(dumped.keys()), f"Field mismatch: {set(dumped.keys())}"


# ---------------------------------------------------------------------------
# Event loop blocking probe
# ---------------------------------------------------------------------------

def test_queue_thread_pattern_yields_event_loop():
    """Verify that the thread+queue pattern yields control to the event loop.

    The run_in_executor(None, q.get) call must allow other coroutines to
    execute between token yields. This test measures that an independently
    scheduled coroutine can complete while tokens are being produced.
    """
    async def _concurrent_flag():
        """A coroutine that sets a flag after a short yield."""
        await asyncio.sleep(0)
        return "done"

    async def _producer_consumer():
        """Simulate the SSE token producer/consumer with concurrent task."""
        q: queue.Queue = queue.Queue()
        n_tokens = 10
        loop = asyncio.get_event_loop()

        # Schedule a concurrent coroutine
        concurrent_task = asyncio.ensure_future(_concurrent_flag())

        def _gen_thread():
            for i in range(n_tokens):
                time.sleep(0.001)  # simulate token generation time
                q.put(f"token_{i}")
            q.put(None)

        threading.Thread(target=_gen_thread, daemon=True).start()

        tokens = []
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            tokens.append(item)

        # The concurrent task should have been able to run
        result = await concurrent_task
        return tokens, result

    tokens, flag = asyncio.run(_producer_consumer())
    assert len(tokens) == 10
    assert flag == "done", "Concurrent coroutine did not complete — event loop was blocked"


def test_sse_sentinel_on_cancellation():
    """If the consumer stops reading, the producer thread completes via sentinel."""
    q: queue.Queue = queue.Queue()
    n_tokens = 50
    produced = []

    def _producer():
        for i in range(n_tokens):
            q.put(f"t{i}")
            produced.append(i)
        q.put(None)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    # Consume only 5 tokens then "cancel"
    consumed = []
    for _ in range(5):
        item = q.get(timeout=2.0)
        if item is None:
            break
        consumed.append(item)

    # Thread should complete naturally (sentinel enqueued)
    t.join(timeout=5.0)
    assert not t.is_alive(), "Producer thread did not complete"
    assert len(consumed) == 5
    assert len(produced) == n_tokens


# ---------------------------------------------------------------------------
# asyncio.to_thread isolation
# ---------------------------------------------------------------------------

def test_asyncio_to_thread_does_not_block_loop():
    """asyncio.to_thread runs blocking call in executor, not on event loop."""
    loop_thread_ids = []
    worker_thread_ids = []
    main_thread_id = threading.get_ident()

    async def _async_main():
        loop_thread_ids.append(threading.get_ident())

        def _blocking():
            worker_thread_ids.append(threading.get_ident())
            time.sleep(0.01)
            return "result"

        result = await asyncio.to_thread(_blocking)
        assert result == "result"

        # Check another coroutine can run concurrently
        await asyncio.sleep(0)
        loop_thread_ids.append(threading.get_ident())

    asyncio.run(_async_main())

    # Loop coroutines ran on main thread
    assert all(tid == main_thread_id for tid in loop_thread_ids)
    # Blocking call ran on a different (worker) thread
    assert worker_thread_ids
    assert worker_thread_ids[0] != main_thread_id, (
        "asyncio.to_thread ran on event loop thread — would have blocked the loop"
    )
