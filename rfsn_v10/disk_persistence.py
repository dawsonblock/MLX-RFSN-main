#!/usr/bin/env python3
"""Disk persistence for KV cache.

Implements write-ahead logging and cache recovery for durability.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheMetadata:
    """Metadata for a persisted cache entry."""

    model_id: str
    layer_id: str
    seq_len: int
    timestamp: float
    checksum: str
    size_bytes: int


class WriteAheadLog:
    """Write-ahead log for durability."""

    def __init__(self, log_path: str, max_file_size_mb: int = 100):
        self.log_path = Path(log_path)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        """Append an entry to the WAL."""
        with self.lock:
            # Check if we need to rotate
            if self.log_path.exists() and self.log_path.stat().st_size >= self.max_file_size:
                self._rotate_log()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def _rotate_log(self) -> None:
        """Rotate the log file."""
        timestamp = int(time.time())
        backup_path = self.log_path.with_suffix(f".{timestamp}.bak")
        self.log_path.rename(backup_path)

    def read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from the WAL."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def truncate(self) -> None:
        """Truncate the WAL after successful checkpoint."""
        with self.lock:
            if self.log_path.exists():
                self.log_path.unlink()


class CachePersistenceManager:
    """Manages disk persistence for KV cache."""

    def __init__(self, cache_dir: str, max_cache_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024
        self.wal = WriteAheadLog(self.cache_dir / "wal.log")
        self.lock = threading.Lock()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, metadata: CacheMetadata) -> Path:
        """Get the cache file path for a given metadata."""
        filename = f"{metadata.model_id}_{metadata.layer_id}_{metadata.seq_len}.cache"
        return self.cache_dir / filename

    def _compute_checksum(self, data: bytes) -> str:
        """Compute a simple checksum for data."""
        return str(hash(data))

    def persist(
        self,
        data: bytes,
        metadata: CacheMetadata,
    ) -> bool:
        """Persist cache data to disk with WAL."""
        cache_path = self._get_cache_path(metadata)

        # Write to WAL first
        wal_entry = {
            "action": "write",
            "path": str(cache_path),
            "metadata": asdict(metadata),
            "timestamp": time.time(),
        }
        self.wal.append(wal_entry)

        try:
            # Write actual data
            with open(cache_path, "wb") as f:
                f.write(data)

            # Write metadata
            metadata_path = cache_path.with_suffix(".meta")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f)

            # Truncate WAL on success
            self.wal.truncate()
            return True
        except Exception:
            # WAL entry remains for recovery
            return False

    def load(self, metadata: CacheMetadata) -> bytes | None:
        """Load cache data from disk."""
        cache_path = self._get_cache_path(metadata)

        if not cache_path.exists():
            return None

        # Verify metadata
        metadata_path = cache_path.with_suffix(".meta")
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                stored_metadata = json.load(f)
                if stored_metadata.get("checksum") != metadata.checksum:
                    return None

        try:
            with open(cache_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def recover(self) -> list[dict[str, Any]]:
        """Recover from WAL after crash."""
        entries = self.wal.read_entries()
        recovered = []

        for entry in entries:
            if entry.get("action") == "write":
                path = Path(entry.get("path", ""))
                if path.exists():
                    recovered.append(entry)

        return recovered

    def evict(self, metadata: CacheMetadata) -> bool:
        """Evict a cache entry from disk."""
        cache_path = self._get_cache_path(metadata)
        metadata_path = cache_path.with_suffix(".meta")

        # Write to WAL
        wal_entry = {
            "action": "delete",
            "path": str(cache_path),
            "timestamp": time.time(),
        }
        self.wal.append(wal_entry)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            self.wal.truncate()
            return True
        except Exception:
            return False

    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for path in self.cache_dir.glob("*.cache"):
            total += path.stat().st_size
        return total

    def enforce_quota(self) -> None:
        """Enforce cache quota by evicting oldest entries."""
        current_size = self.get_cache_size()
        if current_size <= self.max_cache_size:
            return

        # Get all cache entries with metadata
        entries = []
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                    entries.append((metadata, meta_path))
            except Exception:
                continue

        # Sort by timestamp (oldest first)
        entries.sort(key=lambda x: x[0].get("timestamp", 0))

        # Evict oldest entries until under quota
        for metadata, meta_path in entries:
            if self.get_cache_size() <= self.max_cache_size * 0.8:  # Target 80% of max
                break

            cache_path = meta_path.with_suffix(".cache")
            try:
                if cache_path.exists():
                    cache_path.unlink()
                meta_path.unlink()
            except Exception:
                continue

    def list_cached_entries(self) -> list[CacheMetadata]:
        """List all cached entries."""
        entries = []
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                    entries.append(CacheMetadata(**metadata))
            except Exception:
                continue
        return entries
