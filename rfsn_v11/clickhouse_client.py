#!/usr/bin/env python3
"""
RFSN v10 - ClickHouse Client for Telemetry.

HTTPS-only HTTP-based client for writing telemetry batches to ClickHouse.
Prompt text is HMAC-SHA256 hashed before transmission; raw text never leaves
the process boundary.  A salted HMAC key is required via the
``RFSN_TELEMETRY_HMAC_KEY`` environment variable when telemetry is enabled.
Sanitization is recursive — nested dicts and lists of dicts are also cleaned.
"""
from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import os
import signal
import time
import warnings
import weakref
from typing import Any
from urllib import error, request

_active_clients: set[weakref.ref[ClickHouseClient]] = set()


def _sigterm_dispatcher(_signum, _frame) -> None:
    """Dispatch SIGTERM to all active ClickHouseClient instances."""
    for ref in list(_active_clients):
        client = ref()
        if client is not None:
            client._flush_queue_to_disk()


class ClickHouseClient:
    """HTTPS ClickHouse client with token auth,
    prompt hashing, and retry queue.

    Features:
    - HTTPS-only (raises if HTTP is used over non-localhost)
    - Token-based auth via ``RFSN-Auth`` header
    - SHA-256 prompt hashing before serialization
    - Exponential-backoff retry queue with SIGTERM flush to disk
    """

    ALLOWED_TABLES = {
        "rfsn_attention_events",
        "rfsn_kv_cache_events",
        "rfsn_audit_events",
        "rfsn_failures",
        "rfsn_sparsity_profiles",
    }

    _SENSITIVE_KEYS = {
        "prompt",
        "raw_prompt",
        "text",
        "input",
        "inputs",
        "input_text",
        "messages",
        "conversation",
        "completion",
        "output",
        "output_text",
        "response",
        "content",
        "user_message",
        "query",
    }
    _FLUSH_PATH = "/tmp/rfsn_telemetry_flush.jsonl"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        username: str = "default",
        password: str = "",
        database: str = "default",
        secure: bool = True,
        auth_token: str = "",
        create_tables: bool = False,
        sleep_fn=None,
    ):
        """
        Args:
            host: ClickHouse server hostname.
            port: ClickHouse HTTP port (usually 8443 for HTTPS).
            username: ClickHouse username.
            password: ClickHouse password.
            database: Target database.
            secure: Use HTTPS (default True).  HTTP is only allowed when
                host is ``localhost``.
            auth_token: Bearer token sent as ``RFSN-Auth`` header.
            create_tables: If True, attempt to create tables on init.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.secure = secure
        self.auth_token = auth_token
        self.create_tables = create_tables
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep

        if not secure and host not in ("localhost", "127.0.0.1"):
            raise ValueError(
                "HTTP is only allowed for localhost. "
                "Set secure=True for remote hosts."
            )

        self._base_url = f"{'https' if secure else 'http'}://{host}:{port}"
        # Queue stores (table_name, event_dict) tuples so routing metadata is
        # never mixed into the event payload.
        self._pending_queue: list[tuple[str, dict[str, Any]]] = []
        self._max_retries = 5
        self._flush_interval = 1.0
        self._flush_path = self._FLUSH_PATH

        # Register SIGTERM handler to flush queue to disk
        self._register_flush_handlers()
        # Replay any previously flushed events
        self._replay_flushed_events()

        if create_tables:
            self._create_tables_if_not_exist()

    def _sigterm_handler(self, _signum, _frame) -> None:
        self._flush_queue_to_disk()

    def _register_flush_handlers(self) -> None:
        """Register atexit and SIGTERM handlers for queue flush."""
        atexit.register(self._flush_queue_to_disk)
        _active_clients.add(weakref.ref(self))
        try:
            signal.signal(signal.SIGTERM, _sigterm_dispatcher)
        except (ValueError, OSError):
            pass  # May fail in threads or restricted environments

    def _flush_queue_to_disk(self) -> None:
        """Write pending queue to local JSONL file.

        Each line is a JSON object with a ``_table`` routing key and an
        ``_event`` payload key.  The routing key is stored outside the event
        payload so it never appears in ClickHouse rows.
        """
        if not self._pending_queue:
            return
        try:
            with open(self._flush_path, "a", encoding="utf-8") as f:
                for table, event in self._pending_queue:
                    record = {"_table": table, "_event": event}
                    f.write(json.dumps(record, default=str) + "\n")
            self._pending_queue.clear()
        except OSError:
            pass

    def _replay_flushed_events(self) -> None:
        """Read previously flushed events and add them to the queue."""
        if not os.path.exists(self._flush_path):
            return
        try:
            with open(self._flush_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        table = record.get("_table", "rfsn_attention_events")
                        event = record.get("_event", record)
                        # Strip legacy _table key if event was written by old code
                        event.pop("_table", None)
                        self._pending_queue.append((table, event))
            os.remove(self._flush_path)
        except (OSError, json.JSONDecodeError):
            pass

    @staticmethod
    def _hmac_hash_text(text: str, secret: str) -> str:
        """HMAC-SHA256 hash of *text* using *secret* as the key.

        Salted HMAC is used instead of bare SHA-256 so that common prompt
        prefixes cannot be pre-computed by an attacker with access to the
        telemetry database.
        """
        return hmac.new(
            secret.encode("utf-8"),
            text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _sanitize(
        obj: Any,
        *,
        secret: str,
    ) -> Any:
        """Recursively sanitize *obj*, replacing sensitive string values with
        HMAC-SHA256 hashes.

        Rules:
        - Strings under sensitive keys → HMAC hash (+ length sidecar).
        - Nested dicts → recursed.
        - Lists of dicts (e.g. chat messages) → each dict recursed.
        - Other values → passed through unchanged.
        """
        if isinstance(obj, dict):
            result: dict[str, Any] = {}
            for key, value in obj.items():
                if key in ClickHouseClient._SENSITIVE_KEYS:
                    if isinstance(value, str):
                        result[key] = ClickHouseClient._hmac_hash_text(value, secret)
                        result[f"{key}_length"] = len(value)
                    elif isinstance(value, list):
                        # List of message dicts (e.g. OpenAI chat format)
                        result[key] = ClickHouseClient._sanitize(value, secret=secret)
                    elif isinstance(value, dict):
                        result[key] = ClickHouseClient._sanitize(value, secret=secret)
                    else:
                        result[key] = value
                else:
                    result[key] = ClickHouseClient._sanitize(value, secret=secret)
            return result
        elif isinstance(obj, list):
            return [ClickHouseClient._sanitize(item, secret=secret) for item in obj]
        return obj

    @staticmethod
    def _has_sensitive_fields(obj: Any) -> bool:
        """Recursively check whether *obj* contains any sensitive string values."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ClickHouseClient._SENSITIVE_KEYS and isinstance(value, str):
                    return True
                if ClickHouseClient._has_sensitive_fields(value):
                    return True
        elif isinstance(obj, list):
            return any(ClickHouseClient._has_sensitive_fields(item) for item in obj)
        return False

    @staticmethod
    def _hash_sensitive_values(
        event: dict[str, Any],
        *,
        secret: str | None = None,
        _allow_missing_key: bool = False,
    ) -> dict[str, Any]:
        """Sanitize *event* using HMAC-SHA256 (recursive).

        Args:
            event:  The telemetry event dict to sanitize.
            secret: HMAC key.  When *None* the value of the
                    ``RFSN_TELEMETRY_HMAC_KEY`` environment variable is used.
            _allow_missing_key: If True, allow an empty key (for tests only).

        Raises:
            RuntimeError: If the HMAC key is missing and *_allow_missing_key*
                is False. Telemetry with sensitive fields and no HMAC key is a
                production security failure.
        """
        if secret is None:
            secret = os.environ.get("RFSN_TELEMETRY_HMAC_KEY", "")

        # Only enforce key presence when the event actually contains sensitive fields.
        # Events with no sensitive fields are safe to process with any key.
        has_sensitive = ClickHouseClient._has_sensitive_fields(event)
        if not secret and has_sensitive:
            if _allow_missing_key:
                warnings.warn(
                    "RFSN_TELEMETRY_HMAC_KEY is not set.  "
                    "Prompt hashing will use an empty HMAC key which provides "
                    "weaker protection.  Set this env var before enabling telemetry.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "RFSN_TELEMETRY_HMAC_KEY environment variable is not set.  "
                    "Telemetry with sensitive fields requires a salted HMAC key.  "
                    "Set RFSN_TELEMETRY_HMAC_KEY or disable telemetry."
                )
        return ClickHouseClient._sanitize(event, secret=secret)

    def _get_url(self, endpoint: str = "") -> str:
        """Construct full URL for ClickHouse HTTP interface."""
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> None:
        """Execute a ClickHouse query via HTTPS POST with token auth."""
        data = query.encode("utf-8")
        headers = {
            "Content-Type": "text/plain",
            "X-ClickHouse-User": self.username,
            "X-ClickHouse-Key": self.password,
        }
        if self.auth_token:
            headers["RFSN-Auth"] = self.auth_token

        req = request.Request(
            self._get_url(),
            data=data,
            headers=headers,
        )

        try:
            with request.urlopen(req, timeout=10) as response:
                response.read()  # consume body
                if response.status != 200:
                    raise RuntimeError(
                        f"ClickHouse error: {response.status} "
                        f"{response.reason}"
                    )
        except error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to ClickHouse: {e.reason}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"ClickHouse query failed: {e}"
            ) from e

    def _create_tables_if_not_exist(self) -> None:
        """Create telemetry tables if they don't exist."""
        tables = {
            "rfsn_attention_events": """
                CREATE TABLE IF NOT EXISTS rfsn_attention_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    seq_len UInt32,
                    head_count UInt32,
                    head_dim UInt32,
                    top_k_ratio Float32,
                    block_size UInt32,
                    num_active_blocks UInt32,
                    effective_sparsity Float32,
                    kv_cache_hit UInt8,
                    kv_cache_store_latency_ms Float64,
                    kv_cache_retrieve_latency_ms Float64,
                    attention_latency_ms Float64,
                    total_latency_ms Float64,
                    fallback_used UInt8,
                    sparse_success UInt8,
                    dense_success UInt8,
                    audit_enabled UInt8,
                    audit_cosine Nullable(Float64),
                    audit_rel_mae Nullable(Float64),
                    audit_max_abs_error Nullable(Float64),
                    quant_audit_cosine Nullable(Float64),
                    quant_audit_rel_mae Nullable(Float64),
                    quant_audit_max_abs_error Nullable(Float64),
                    sparse_audit_cosine Nullable(Float64),
                    sparse_audit_rel_mae Nullable(Float64),
                    sparse_audit_max_abs_error Nullable(Float64),
                    execution_mode String,
                    termination_reason String,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_kv_cache_events": """
                CREATE TABLE IF NOT EXISTS rfsn_kv_cache_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    operation String,  -- 'store' or 'retrieve'
                    key_size_bytes UInt64,
                    value_size_bytes UInt64,
                    latency_ms Float64,
                    success UInt8,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_audit_events": """
                CREATE TABLE IF NOT EXISTS rfsn_audit_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    sparse_output String,  -- JSON or hash for comparison
                    dense_output String,   -- JSON or hash for comparison
                    cosine_similarity Float64,
                    rel_mae Float64,
                    max_abs_error Float64,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_failures": """
                CREATE TABLE IF NOT EXISTS rfsn_failures (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    error_type String,
                    error_message String,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_sparsity_profiles": """
                CREATE TABLE IF NOT EXISTS rfsn_sparsity_profiles (
                    task_id String,
                    model_id String,
                    skill_pattern String,
                    avg_top_k_ratio Float64,
                    min_top_k_ratio Float64,
                    max_top_k_ratio Float64,
                    avg_effective_sparsity Float64,
                    fallback_rate Float64,
                    sample_count UInt32,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """
        }

        for table_name, ddl in tables.items():
            try:
                self._execute_query(ddl)
            except Exception as e:
                warnings.warn(
                    f"Could not create table {table_name}: {e}"
                )

    def write_telemetry_batch(self, events: list[dict[str, Any]]) -> None:
        """
        Write a batch of telemetry events to ClickHouse.

        Args:
            events: List of telemetry event dictionaries.
        """
        self.write_attention_events(events)

    def _write_events(self, table: str, events: list[dict[str, Any]]) -> None:
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Table not allowed: {table}")
        if not events:
            return

        # Hash sensitive fields before serialization
        hashed = [
            self._hash_sensitive_values(ev) for ev in events
        ]
        payload = "\n".join(
            json.dumps(ev, default=str, separators=(",", ":"))
            for ev in hashed
        )
        query = f"INSERT INTO {table} FORMAT JSONEachRow\n" + payload

        # Try with exponential backoff; on persistent failure, queue locally.
        # Routing metadata (table name) is stored separately from event payloads
        # so it never appears in ClickHouse rows.
        for attempt in range(self._max_retries):
            try:
                self._execute_query(query)
                # Also retry any previously queued events
                if self._pending_queue:
                    self._drain_pending_queue()
                return
            except (error.URLError, RuntimeError):
                backoff = min(2 ** attempt, 30)
                self._sleep(backoff)
        # Persist as (table, event) tuples — routing key is NOT in the payload
        self._pending_queue.extend((table, ev) for ev in hashed)

    def _drain_pending_queue(self) -> None:
        """Attempt to flush queued events, grouped by destination table."""
        if not self._pending_queue:
            return

        # Group events by table to issue one INSERT per table
        from collections import defaultdict
        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for tbl, ev in self._pending_queue:
            by_table[tbl].append(ev)

        all_flushed = True
        for tbl, evs in by_table.items():
            payload = "\n".join(
                json.dumps(ev, default=str, separators=(",", ":")) for ev in evs
            )
            query = f"INSERT INTO {tbl} FORMAT JSONEachRow\n" + payload
            try:
                self._execute_query(query)
            except (error.URLError, RuntimeError):
                all_flushed = False

        if all_flushed:
            self._pending_queue.clear()

    def write_attention_events(self, events: list[dict[str, Any]]) -> None:
        self._write_events("rfsn_attention_events", events)

    def write_kv_cache_events(self, events: list[dict[str, Any]]) -> None:
        self._write_events("rfsn_kv_cache_events", events)

    def write_audit_events(self, events: list[dict[str, Any]]) -> None:
        self._write_events("rfsn_audit_events", events)

    def write_failure_events(self, events: list[dict[str, Any]]) -> None:
        self._write_events("rfsn_failures", events)

    def write_sparsity_profile_events(
        self, events: list[dict[str, Any]]
    ) -> None:
        self._write_events("rfsn_sparsity_profiles", events)

    def close(self) -> None:
        """Close the client and cleanup resources."""
        atexit.unregister(self._flush_queue_to_disk)
        for ref in list(_active_clients):
            if ref() is self:
                _active_clients.discard(ref)
                break

    def ping(self) -> bool:
        """Check if ClickHouse is reachable."""
        try:
            self._execute_query("SELECT 1")
            return True
        except Exception:
            return False
