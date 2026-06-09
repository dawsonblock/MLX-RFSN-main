#!/usr/bin/env python3
"""RFSN v10 — Disk persistence tests.

Covers WAL, cache persistence manager, recovery, eviction, and quota
enforcement without requiring MLX.
"""
from __future__ import annotations

from rfsn_v10.disk_persistence import (
    CacheMetadata,
    CachePersistenceManager,
    WriteAheadLog,
)

# ------------------------------------------------------------------
# CacheMetadata
# ------------------------------------------------------------------

class TestCacheMetadata:
    def test_creation(self):
        meta = CacheMetadata(
            model_id="test_model",
            layer_id="layer_0",
            seq_len=128,
            timestamp=1234567890.0,
            checksum="abc123",
            size_bytes=1024,
        )
        assert meta.model_id == "test_model"
        assert meta.layer_id == "layer_0"
        assert meta.seq_len == 128


# ------------------------------------------------------------------
# WriteAheadLog
# ------------------------------------------------------------------

class TestWriteAheadLog:
    def test_append_and_read(self, tmp_path):
        wal = WriteAheadLog(str(tmp_path / "wal.log"))
        wal.append({"action": "write", "path": "/a"})
        wal.append({"action": "write", "path": "/b"})
        entries = wal.read_entries()
        assert len(entries) == 2
        assert entries[0]["path"] == "/a"
        assert entries[1]["path"] == "/b"

    def test_read_empty_returns_empty(self, tmp_path):
        wal = WriteAheadLog(str(tmp_path / "wal.log"))
        assert wal.read_entries() == []

    def test_truncate(self, tmp_path):
        wal = WriteAheadLog(str(tmp_path / "wal.log"))
        wal.append({"data": "x"})
        wal.truncate()
        assert wal.read_entries() == []

    def test_ignores_malformed_lines(self, tmp_path):
        wal_path = tmp_path / "wal.log"
        wal = WriteAheadLog(str(wal_path))
        wal.append({"valid": True})
        # Manually append invalid JSON
        with open(wal_path, "a") as f:
            f.write("not json\n")
        entries = wal.read_entries()
        assert len(entries) == 1
        assert entries[0]["valid"] is True

    def test_rotation(self, tmp_path):
        wal = WriteAheadLog(str(tmp_path / "wal.log"), max_file_size_mb=0)
        wal.append({"data": "x" * 100})
        # Force rotation by appending more
        wal.append({"data": "y" * 100})
        # After rotation, old entries should be in backup
        entries = wal.read_entries()
        # Should only have entries since rotation
        assert all(e["data"].startswith("y") for e in entries)


# ------------------------------------------------------------------
# CachePersistenceManager
# ------------------------------------------------------------------

class TestCachePersistenceManager:
    def test_persist_and_load(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        data = b"hello cache"
        assert mgr.persist(data, meta) is True
        loaded = mgr.load(meta)
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        assert mgr.load(meta) is None

    def test_load_checksum_mismatch_returns_none(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck_a", size_bytes=128,
        )
        mgr.persist(b"data", meta)
        meta_bad = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck_b", size_bytes=128,
        )
        assert mgr.load(meta_bad) is None

    def test_evict_removes_files(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        mgr.persist(b"data", meta)
        assert mgr.load(meta) is not None
        assert mgr.evict(meta) is True
        assert mgr.load(meta) is None

    def test_recover_finds_existing_entries(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        mgr.persist(b"data", meta)
        # persist() truncates WAL on success; simulate a crash by re-adding a WAL entry
        mgr.wal.append({"action": "write", "path": str(tmp_path / "m1_l0_64.cache"), "timestamp": 1.0})
        recovered = mgr.recover()
        assert len(recovered) >= 1
        assert recovered[0]["action"] == "write"

    def test_get_cache_size(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        mgr.persist(b"12345678", meta)
        assert mgr.get_cache_size() == 8

    def test_list_cached_entries(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path))
        meta = CacheMetadata(
            model_id="m1", layer_id="l0", seq_len=64,
            timestamp=1.0, checksum="ck", size_bytes=128,
        )
        mgr.persist(b"data", meta)
        entries = mgr.list_cached_entries()
        assert len(entries) == 1
        assert entries[0].model_id == "m1"

    def test_enforce_quota_evicts_oldest(self, tmp_path):
        mgr = CachePersistenceManager(str(tmp_path), max_cache_size_gb=0.000001)
        for i in range(3):
            meta = CacheMetadata(
                model_id="m1", layer_id=f"l{i}", seq_len=64,
                timestamp=float(i), checksum=f"ck{i}", size_bytes=128,
            )
            mgr.persist(b"x" * 100, meta)
        mgr.enforce_quota()
        # Should have evicted oldest entries
        size = mgr.get_cache_size()
        assert size <= mgr.max_cache_size * 0.8
