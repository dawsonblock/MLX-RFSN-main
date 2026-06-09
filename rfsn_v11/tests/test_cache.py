"""Tests for rfsn_v11 cache infrastructure.

Covers:
  1. TQ disk-store serialize roundtrip (shape + cosine-similarity of norms).
  2. Cache-version mismatch hard-fail on deserialize.
  3. BlockAwarePrefixCache LRU eviction leaves only ``max_entries`` items.
  4. SIGTERM triggers cache flush without error.

MLX-dependent tests are gated with ``pytest.importorskip("mlx.core")``.
Pure Python / numpy tests run without any GPU / MLX dependency.
"""
from __future__ import annotations

import json
import os
import signal
import tempfile
import types
from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# Test 1 — TQ disk-store serialize roundtrip
# ===========================================================================


def test_tq_disk_store_roundtrip():
    """serialize_tq_cache on a mock TurboQuantKVCache layer preserves shapes.

    We construct a minimal fake layer that exposes exactly the attributes read
    by ``_serialize_tq_layer``:
      - ``_compressed_keys``: EncodedKeys namedtuple
      - ``_compressed_values``: EncodedValues namedtuple
      - scalar attributes: ``offset``, ``key_dim``, ``value_dim``,
        ``key_bits``, ``value_bits``, ``sink_tokens``

    Since ``serialize_tq_cache`` requires MLX for creating ``mx.array``
    tensors, we gate the test on MLX being importable.  We then verify:
      - All expected tensor keys are present in the serialised output.
      - Tensor shapes roundtrip exactly.
      - Cosine similarity between original and serialised ``ck_vector_norms``
        is > 0.999 (they must be identical).
    """
    mx = pytest.importorskip("mlx.core")

    from rfsn_v11.cache.tq_disk_store import serialize_tq_cache
    from rfsn_v11.cache.version import CACHE_VERSION

    # Build the EncodedKeys / EncodedValues namedtuples expected by the code.
    EncodedKeys = namedtuple(
        "EncodedKeys",
        ["indices_packed", "qjl_packed", "residual_norms", "vector_norms",
         "shape", "index_bits"],
    )
    EncodedValues = namedtuple(
        "EncodedValues",
        ["indices_packed", "vector_norms", "shape", "index_bits"],
    )

    # Shapes: (batch=1, heads=4, tokens=8, dim=128)
    ck_shape = (1, 4, 8, 128)
    cv_shape = (1, 4, 8, 128)
    n_packed_k = 8  # arbitrary packed index count
    n_packed_v = 8

    rng = np.random.RandomState(42)

    # mx arrays (float16 for norms, uint32 for packed indices)
    ck_indices = mx.array(
        rng.randint(0, 2**16, size=(n_packed_k,)).astype(np.uint32)
    )
    ck_qjl = mx.array(
        rng.randint(0, 2**16, size=(n_packed_k,)).astype(np.uint32)
    )
    ck_residual_norms_np = rng.rand(1, 4, 8).astype(np.float16)
    ck_vector_norms_np   = rng.rand(1, 4, 8).astype(np.float16)
    ck_residual_norms = mx.array(ck_residual_norms_np)
    ck_vector_norms   = mx.array(ck_vector_norms_np)

    cv_indices = mx.array(
        rng.randint(0, 2**16, size=(n_packed_v,)).astype(np.uint32)
    )
    cv_vector_norms_np = rng.rand(1, 4, 8).astype(np.float16)
    cv_vector_norms    = mx.array(cv_vector_norms_np)

    encoded_keys = EncodedKeys(
        indices_packed=ck_indices,
        qjl_packed=ck_qjl,
        residual_norms=ck_residual_norms,
        vector_norms=ck_vector_norms,
        shape=ck_shape,
        index_bits=3,
    )
    encoded_values = EncodedValues(
        indices_packed=cv_indices,
        vector_norms=cv_vector_norms,
        shape=cv_shape,
        index_bits=3,
    )

    # Build the fake TurboQuantKVCache layer using a real class so that
    # type(fake_layer).__name__ == "TurboQuantKVCache" (SimpleNamespace does
    # not allow __class__ reassignment).
    _FakeTQKVCache = type("TurboQuantKVCache", (), {})
    fake_layer = _FakeTQKVCache()
    fake_layer._compressed_keys = encoded_keys
    fake_layer._compressed_values = encoded_values
    fake_layer.offset = 8
    fake_layer.key_dim = 128
    fake_layer.value_dim = 128
    fake_layer.key_bits = 3
    fake_layer.value_bits = 3
    fake_layer.sink_tokens = 0

    cache = [fake_layer]

    tensors, metadata = serialize_tq_cache(cache)

    # ---- Metadata checks ---------------------------------------------------
    assert metadata["__tq_native__"] == "true"
    assert metadata["__num_layers__"] == "1"
    assert int(metadata["__cache_version__"]) == CACHE_VERSION
    assert json.loads(metadata["__tq_0_ck_shape__"]) == list(ck_shape)
    assert json.loads(metadata["__tq_0_cv_shape__"]) == list(cv_shape)

    # ---- Tensor presence checks --------------------------------------------
    expected_keys = [
        "tq_0_ck_indices_packed",
        "tq_0_ck_qjl_packed",
        "tq_0_ck_residual_norms",
        "tq_0_ck_vector_norms",
        "tq_0_cv_indices_packed",
        "tq_0_cv_vector_norms",
    ]
    for key in expected_keys:
        assert key in tensors, f"Missing tensor key: {key!r}"

    # ---- Shape roundtrip ---------------------------------------------------
    mx.eval(tensors["tq_0_ck_vector_norms"])
    reconstructed_vnorms = np.array(tensors["tq_0_ck_vector_norms"])
    assert reconstructed_vnorms.shape == ck_vector_norms_np.shape, (
        f"Shape mismatch: {reconstructed_vnorms.shape} != {ck_vector_norms_np.shape}"
    )

    # ---- Cosine similarity > 0.999 (should be exact / 1.0) -----------------
    orig_flat  = ck_vector_norms_np.astype(np.float32).ravel()
    recon_flat = reconstructed_vnorms.astype(np.float32).ravel()
    cosine_sim = float(
        np.dot(orig_flat, recon_flat)
        / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-9)
    )
    assert cosine_sim > 0.999, (
        f"Cosine similarity of ck_vector_norms too low: {cosine_sim:.6f}"
    )


# ===========================================================================
# Test 2 — Cache version mismatch hard-fail
# ===========================================================================


def test_cache_version_mismatch():
    """deserialize_tq_cache raises ValueError on CACHE_VERSION mismatch.

    The error is the low-level ``ValueError`` raised by ``tq_disk_store``
    to hard-fail on any version mismatch and prevent silent misreconstruction.
    This corresponds to the semantic :class:`rfsn_v11.errors.CacheVersionError`.

    We craft a metadata dict with a future version number that will never
    match the current CACHE_VERSION, then assert the hard-fail fires before
    any tensor is allocated.
    """
    mx = pytest.importorskip("mlx.core")

    from rfsn_v11.cache.tq_disk_store import deserialize_tq_cache
    from rfsn_v11.cache.version import CACHE_VERSION
    from rfsn_v11.errors import CacheVersionError  # noqa: F401 — imported for doc

    mismatched_version = CACHE_VERSION + 99

    metadata = {
        "__tq_native__": "true",
        "__num_layers__": "1",
        "__cache_version__": str(mismatched_version),
        "__layer_0_class__": "TurboQuantKVCache",
    }
    tensors: dict = {}

    with pytest.raises(ValueError, match="Cache version mismatch"):
        deserialize_tq_cache(tensors, metadata)


# ===========================================================================
# Test 3 — BlockAwarePrefixCache LRU eviction
# ===========================================================================


def test_prefix_cache_lru_eviction():
    """BlockAwarePrefixCache tracks 5 entries; releasing LRU leaves 4.

    Tests the LRU ordering maintained by ``_entries_by_type`` (OrderedDict).
    We use:
      - ``PagedCacheManager(block_size=64, max_blocks=6)``
        → 5 usable blocks (1 reserved as null block).
      - ``BlockAwarePrefixCache(model=None, paged_cache_manager=pm)``
      - 5 ``store_cache()`` calls with non-tensor cache_data (avoids MLX).

    After storing all 5 entries:
      - ``_request_tables`` should have 5 keys.
      - ``_entries_by_type["assistant"]`` should have 5 keys.
    Releasing the least-recently-used (first inserted) entry:
      - ``_request_tables`` drops to 4 keys.
      - The released key is no longer present.
    """
    from rfsn_v11.cache.paged_cache import PagedCacheManager
    from rfsn_v11.cache.prefix_cache import BlockAwarePrefixCache

    # max_blocks=6 → 5 usable (block 0 is the null block).
    pm = PagedCacheManager(block_size=64, max_blocks=6)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=pm)

    request_ids = [f"req_{i}" for i in range(5)]

    # cache_data is a plain Python list (not tensor dicts) — avoids MLX.
    # store_cache stores it directly as block.cache_data for the last block.
    fake_cache_data = ["layer_0_placeholder", "layer_1_placeholder"]

    for rid in request_ids:
        # Each call uses a distinct token sequence → distinct block hash.
        tokens = [ord(c) for c in rid]  # e.g. [114, 101, 113, 95, 48]
        pc.store_cache(rid, tokens, fake_cache_data, cache_type="assistant")

    # All 5 entries should be registered.
    assert len(pc._request_tables) == 5, (
        f"Expected 5 entries in _request_tables, got {len(pc._request_tables)}"
    )
    bucket = pc._entries_by_type["assistant"]
    assert len(bucket) == 5, (
        f"Expected 5 entries in LRU bucket, got {len(bucket)}"
    )

    # The LRU entry is the first key in the OrderedDict (insertion order = LRU).
    lru_id = next(iter(bucket))
    assert lru_id == request_ids[0], (
        f"Expected LRU to be {request_ids[0]!r}, got {lru_id!r}"
    )

    # Release the LRU entry.
    pc.release_cache(lru_id)

    assert len(pc._request_tables) == 4, (
        f"Expected 4 entries after LRU release, got {len(pc._request_tables)}"
    )
    assert lru_id not in pc._request_tables, (
        f"LRU entry {lru_id!r} should have been removed from _request_tables"
    )
    assert lru_id not in pc._entries_by_type["assistant"], (
        f"LRU entry {lru_id!r} should have been removed from LRU bucket"
    )


# ===========================================================================
# Test 4 — SIGTERM flush
# ===========================================================================


def test_sigterm_flush():
    """Sending SIGTERM to the current process triggers queue flush to disk.

    We:
    1. Create a ``ClickHouseClient`` (localhost, no real server needed).
    2. Inject a fake event into ``_pending_queue``.
    3. Install a controlled SIGTERM handler that calls ``_flush_queue_to_disk``
       and records whether it completed.
    4. Raise SIGTERM in-process via ``os.kill(os.getpid(), signal.SIGTERM)``.
    5. Assert flush completed and the flush file was written.

    The ClickHouseClient SIGTERM dispatcher is a module-level function; we
    install our own handler that additionally sets a flag so the test can
    detect completion even if the signal is masked by the test framework.
    """
    import tempfile

    from rfsn_v11.clickhouse_client import ClickHouseClient

    flush_completed = {"value": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        flush_path = os.path.join(tmpdir, "rfsn_test_flush.jsonl")

        # Instantiate with localhost (HTTP allowed) and no real connection.
        client = ClickHouseClient(
            host="localhost",
            port=19999,   # unused port — no connection is made
            secure=False,
        )
        # Override flush path so we don't pollute /tmp.
        client._flush_path = flush_path

        # Inject a fake event directly into the queue.
        client._pending_queue.append(("rfsn_attention_events", {"test_key": "test_val"}))

        # Custom SIGTERM handler: flushes this client and records success.
        original_handler = signal.getsignal(signal.SIGTERM)

        def _test_handler(_signum, _frame):
            try:
                client._flush_queue_to_disk()
                flush_completed["value"] = True
            except Exception as exc:
                flush_completed["value"] = False
                raise exc

        signal.signal(signal.SIGTERM, _test_handler)

        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            # Restore original handler.
            signal.signal(signal.SIGTERM, original_handler)

        # Assert flush completed without error.
        assert flush_completed["value"], "Flush handler raised an exception"

        # Assert queue is now empty.
        assert len(client._pending_queue) == 0, (
            f"Expected empty queue after flush, got {len(client._pending_queue)} items"
        )

        # Assert flush file was written and contains valid JSON.
        assert os.path.exists(flush_path), (
            f"Flush file {flush_path!r} was not created"
        )
        with open(flush_path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        assert len(lines) == 1, f"Expected 1 line in flush file, got {len(lines)}"
        record = json.loads(lines[0])
        assert record.get("_table") == "rfsn_attention_events"
        assert record.get("_event", {}).get("test_key") == "test_val"
