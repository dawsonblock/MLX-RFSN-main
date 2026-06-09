"""RFSN v10 — Experimental Quant Runtime.

A purely experimental integration layer that wraps
:class:`RFSNTurboQuantKVManager` with additional telemetry,
layer-wise quant policies, and adaptive / hybrid quant modes.

All logging is written to ``artifacts/runtime_logs/*.jsonl``.

Supported quant modes:

* ``stable_k8_v5_gs64``   — default; 8-bit keys, 5-bit values, group-size 64.
* ``adaptive``            — heuristically switch bits based on sequence length.
* ``experimental_hybrid`` — ``hybrid_polar_cartesian`` backend.
"""

from __future__ import annotations

import json
import sys
import time
import uuid

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None  # type: ignore[assignment]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    from rfsn_v10.compat import mx

from ..kv_manager import RFSNTurboQuantKVManager
from .adaptive_controller import AdaptiveQuantController
from .audit import AuditMetrics, audit_decode_step, check_drift
from .scoring_modes import (
    score_attention_fp16,
    score_attention_packed_block,
    score_attention_prepared,
    score_attention_quantized_metal,
    score_attention_reconstructed,
)

DEFAULT_TELEMETRY_DIR: Path = Path("artifacts/runtime_logs")
DEFAULT_QUANT_MODE: str = "stable_k8_v5_gs64"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LayerQuantPolicy:
    """Per-layer quant configuration."""

    k_bits: int = 8
    v_bits: int = 5
    group_size: int = 64
    use_wht: bool = True
    use_incoherent_signs: bool = True
    quant_mode: str = "cartesian"


@dataclass
class QuantTelemetryEvent:
    """Single decode-step telemetry record for the experimental
    quant runtime."""

    task_id: str
    model_id: str
    layer_id: str
    quant_mode: str
    layer_policy_applied: bool
    compressed_bytes: int
    dequant_time_ms: float
    tokens_per_sec: float
    fallback_event: str | None
    quality_audit: dict[str, Any] | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentalQuantState:
    """Mutable runtime state tracked across decode steps."""

    total_compressed_bytes: int = 0
    total_dequant_time_ms: float = 0.0
    total_tokens: int = 0
    fallback_count: int = 0
    audit_samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedCache:
    """Prepared cache mode storage.

    Caches scales, packed buffers, block offsets, and dequantized scratch
    blocks to avoid repeated decode overhead.
    """

    skill_pattern: str = ""
    k_packed: Any = None
    v_packed: Any = None
    k_scales: Any = None
    v_scales: Any = None
    block_offsets: list[int] = field(default_factory=list)
    dequantized_k: mx.array | None = None
    dequantized_v: mx.array | None = None
    hits: int = 0
    misses: int = 0

    def is_valid_for(self, skill_pattern: str) -> bool:
        return (
            self.skill_pattern == skill_pattern
            and self.dequantized_k is not None
            and self.dequantized_v is not None
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_layer_policy(
    path: Path | str | None,
) -> dict[str, LayerQuantPolicy]:
    """Load a JSON layer policy into a mapping of layer_id → policy."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return {}
    # Support both flat mapping and structured {"default": ..., "layers": ...}
    layers = raw.get("layers", raw) if "default" in raw else raw
    if not isinstance(layers, dict):
        return {}
    policy: dict[int, LayerQuantPolicy] = {}
    for layer_id, cfg in layers.items():
        if not isinstance(cfg, dict):
            continue
        try:
            layer_key = int(layer_id)
        except (ValueError, TypeError):
            continue
        policy[layer_key] = LayerQuantPolicy(
            k_bits=cfg.get("k_bits", 8),
            v_bits=cfg.get("v_bits", 5),
            group_size=cfg.get("group_size", 64),
            use_wht=cfg.get("use_wht", True),
            use_incoherent_signs=cfg.get("use_incoherent_signs", True),
            quant_mode=cfg.get("quant_mode", "cartesian"),
        )
    return policy


def _estimate_compressed_bytes(
    manager: RFSNTurboQuantKVManager,
    keys: mx.array,
    _values: mx.array,
) -> int:
    """Rough byte estimate for a stored KV pair."""
    _bsz, num_h, t_len, head_dim = keys.shape
    # Packed words + scales; this is a coarse over-estimate.
    k_cpw = 32 // manager.k_bits
    v_cpw = 32 // manager.v_bits
    k_words = (num_h * t_len * head_dim + k_cpw - 1) // k_cpw
    v_words = (num_h * t_len * head_dim + v_cpw - 1) // v_cpw
    groups_per_tensor = (
        num_h * t_len * head_dim + manager.group_size - 1
    ) // manager.group_size
    bytes_est = (k_words + v_words) * 4 + groups_per_tensor * 2 * 4
    return max(bytes_est, 1)


def _log_telemetry_event(
    event: QuantTelemetryEvent,
    log_dir: Path = DEFAULT_TELEMETRY_DIR,
) -> None:
    """Append a telemetry event to the JSONL stream."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "experimental_quant_telemetry.jsonl"
    record = {
        "task_id": event.task_id,
        "model_id": event.model_id,
        "layer_id": event.layer_id,
        "quant_mode": event.quant_mode,
        "layer_policy_applied": event.layer_policy_applied,
        "compressed_bytes": event.compressed_bytes,
        "dequant_time_ms": event.dequant_time_ms,
        "tokens_per_sec": event.tokens_per_sec,
        "fallback_event": event.fallback_event,
        "quality_audit": event.quality_audit,
        "timestamp": event.timestamp,
    }
    with path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(
                json.dumps(record, ensure_ascii=False, default=str)
                + "\n"
            )
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _log_fallback_event(
    reason: str,
    step_num: int,
    log_dir: Path = DEFAULT_TELEMETRY_DIR,
) -> None:
    """Record a fallback decision."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "fallback_events.jsonl"
    record = {
        "event_type": "fallback",
        "reason": reason,
        "step_num": step_num,
        "timestamp": time.time(),
    }
    with path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(
                json.dumps(record, ensure_ascii=False, default=str)
                + "\n"
            )
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Adaptive heuristic
# ---------------------------------------------------------------------------


def _adaptive_bits(seq_len: int) -> tuple[int, int, int]:
    """Return (k_bits, v_bits, group_size) based on
    sequence length heuristic."""
    if seq_len <= 0:
        raise ValueError(
            f"seq_len must be positive, got {seq_len}"
        )
    if seq_len <= 512:
        return 8, 5, 64
    elif seq_len <= 2048:
        return 6, 4, 64
    else:
        return 4, 3, 64


# ---------------------------------------------------------------------------
# Main runtime class
# ---------------------------------------------------------------------------


class ExperimentalQuantRuntime:
    """Experimental quant runtime that wraps :class:`RFSNTurboQuantKVManager`.

    Args:
        base_manager: Existing KV manager to wrap (mutations are avoided).
        quant_mode: One of ``stable_k8_v5_gs64``, ``adaptive``,
            ``experimental_hybrid``.
        layer_policy_path: Optional JSON path for per-layer quant configs.
        model_id: Identifier for telemetry.
        telemetry_dir: Directory for JSONL output.
    """

    def __init__(
        self,
        base_manager: RFSNTurboQuantKVManager | None = None,
        quant_mode: str = DEFAULT_QUANT_MODE,
        layer_policy_path: str | None = None,
        model_id: str = "experimental",
        telemetry_dir: str = str(DEFAULT_TELEMETRY_DIR),
        enable_live_adjust: bool = False,
    ):
        if quant_mode not in (
            "stable_k8_v5_gs64",
            "adaptive",
            "experimental_hybrid",
            "live_adaptive",
        ):
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")
        if enable_live_adjust and quant_mode not in (
            "adaptive",
            "live_adaptive",
        ):
            raise ValueError(
                "enable_live_adjust requires quant_mode 'adaptive' or "
                "'live_adaptive'"
            )

        self.quant_mode = quant_mode
        self.model_id = model_id
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Layer policy
        self.layer_policy = _load_layer_policy(layer_policy_path)
        self._layer_policy_path = layer_policy_path

        # Base manager (we may spawn child managers per-mode)
        if base_manager is None:
            base_manager = self._make_manager_for_mode(quant_mode)
        self.base_manager = base_manager

        # Live auto-adjust controller
        self._live_controller: AdaptiveQuantController | None = None
        if enable_live_adjust:
            self._live_controller = AdaptiveQuantController()

        # Runtime state
        self.state = ExperimentalQuantState()
        self._step_num = 0
        self._prepared_cache: dict[str, PreparedCache] = {}
        self._adaptive_managers: dict[
            tuple[int, int, int], RFSNTurboQuantKVManager
        ] = {}

        # Log startup
        self._log_startup()

    # ------------------------------------------------------------------
    # Manager factory
    # ------------------------------------------------------------------

    def _make_manager_for_mode(
        self, mode: str
    ) -> RFSNTurboQuantKVManager:
        """Construct a fresh manager tuned for the requested mode."""
        if mode == "stable_k8_v5_gs64":
            return RFSNTurboQuantKVManager(
                k_bits=8,
                v_bits=5,
                group_size=64,
                quant_mode="cartesian",
            )
        if mode == "adaptive":
            # Adaptive starts with default settings; bits are chosen per-step.
            return RFSNTurboQuantKVManager(
                k_bits=8,
                v_bits=5,
                group_size=64,
                quant_mode="cartesian",
            )
        if mode == "experimental_hybrid":
            return RFSNTurboQuantKVManager(
                k_bits=8,
                v_bits=5,
                group_size=64,
                quant_mode="hybrid_polar_cartesian",
            )
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Startup log
    # ------------------------------------------------------------------

    def _log_startup(self) -> None:
        path = self.telemetry_dir / "experimental_quant_telemetry.jsonl"
        record = {
            "event_type": "startup",
            "model_id": self.model_id,
            "quant_mode": self.quant_mode,
            "layer_policy_path": self._layer_policy_path,
            "layer_policy_layers": list(self.layer_policy.keys()),
            "timestamp": time.time(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(record, ensure_ascii=False, default=str)
                + "\n"
            )

    # ------------------------------------------------------------------
    # Per-layer manager resolution
    # ------------------------------------------------------------------

    def _manager_for_layer(self, layer_id: str) -> RFSNTurboQuantKVManager:
        """Return a manager instance respecting layer policy, if any."""
        try:
            layer_key = int(layer_id)
        except (ValueError, TypeError):
            layer_key = None
        if layer_key is not None and layer_key in self.layer_policy:
            policy = self.layer_policy[layer_key]
            base = self.base_manager
            return RFSNTurboQuantKVManager(
                k_bits=policy.k_bits,
                v_bits=policy.v_bits,
                group_size=policy.group_size,
                use_wht=policy.use_wht,
                use_incoherent_signs=policy.use_incoherent_signs,
                quant_mode=policy.quant_mode,
                max_memory_gb=base.max_memory_gb,
                max_pinned_memory_gb=base.max_pinned_memory_gb,
                cache_dir=str(base.cache_dir),
                prefer_metal_kernels=base.prefer_metal_kernels,
                strict_metal=base.strict_metal,
                validate_metal_codes=base.validate_metal_codes,
            )
        return self.base_manager

    # ------------------------------------------------------------------
    # Decode step
    # ------------------------------------------------------------------

    def decode_step(
        self,
        layer_id: str,
        skill_pattern: str,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        token_count: int,
        reference_logits: mx.array | None = None,
        block_indices: list[int] | None = None,
        scoring_mode: str = "reconstructed",
    ) -> tuple[mx.array, dict[str, Any]]:
        """Run one experimental decode step with quant KV and telemetry.

        Args:
            layer_id:        Identifier for the transformer layer.
            skill_pattern:   Cache key used by the KV manager.
            queries:         [B, H, T_q, D]
            keys:            [B, H, T_k, D]  (full-precision input)
            values:          [B, H, T_k, D]  (full-precision input)
            token_count:     Number of tokens represented by *keys* / *values*.
            reference_logits: Optional reference logits for audit
                comparison.
            block_indices:   If provided, use packed-block scoring
                (sparse blocks).
            scoring_mode:    One of ``fp16``, ``reconstructed``,
                ``prepared``, ``packed_block``.

        Returns:
            (attention_output, info_dict)
        """
        task_id = str(uuid.uuid4())
        t_start = time.monotonic()
        self._step_num += 1

        manager = self._manager_for_layer(layer_id)
        try:
            _layer_key = int(layer_id)
        except (ValueError, TypeError):
            _layer_key = None
        policy_applied = (
            _layer_key is not None and _layer_key in self.layer_policy
        )

        # ------------------------------------------------------------------
        # 1. Adaptive bit selection (if mode == adaptive)
        # ------------------------------------------------------------------
        effective_k_bits = manager.k_bits
        effective_v_bits = manager.v_bits
        effective_group_size = manager.group_size
        controller_adjusted = False
        if self.quant_mode == "adaptive":
            seq_len = keys.shape[2]
            effective_k_bits, effective_v_bits, effective_group_size = (
                _adaptive_bits(seq_len)
            )
            policy_applied = False
            manager = self._adaptive_managers.setdefault(
                (effective_k_bits, effective_v_bits, effective_group_size),
                RFSNTurboQuantKVManager(
                    k_bits=effective_k_bits,
                    v_bits=effective_v_bits,
                    group_size=effective_group_size,
                    quant_mode="cartesian",
                ),
            )
        elif self.quant_mode == "live_adaptive":
            if self._live_controller is not None:
                (
                    effective_k_bits,
                    effective_v_bits,
                    effective_group_size,
                ) = self._live_controller.get_effective_bits()
                controller_adjusted = True
            else:
                seq_len = keys.shape[2]
                (
                    effective_k_bits,
                    effective_v_bits,
                    effective_group_size,
                ) = _adaptive_bits(seq_len)
            policy_applied = False
            manager = self._adaptive_managers.setdefault(
                (effective_k_bits, effective_v_bits, effective_group_size),
                RFSNTurboQuantKVManager(
                    k_bits=effective_k_bits,
                    v_bits=effective_v_bits,
                    group_size=effective_group_size,
                    quant_mode="cartesian",
                ),
            )

        # ------------------------------------------------------------------
        # 2. Store KV
        # ------------------------------------------------------------------
        manager.store(
            skill_pattern,
            keys,
            values,
            token_count=token_count,
            k_bits=effective_k_bits,
            v_bits=effective_v_bits,
        )
        compressed_bytes = _estimate_compressed_bytes(manager, keys, values)

        # ------------------------------------------------------------------
        # 3. Retrieve / score according to scoring_mode
        # ------------------------------------------------------------------
        attn_output: mx.array
        dequant_time_ms = 0.0

        if scoring_mode == "fp16":
            # Baseline: bypass quant entirely
            attn_output = score_attention_fp16(queries, keys, values)

        elif scoring_mode == "reconstructed":
            t_deq = time.monotonic()
            retrieved = manager.retrieve(skill_pattern)
            dequant_time_ms = (time.monotonic() - t_deq) * 1000.0
            if retrieved is None:
                raise RuntimeError("retrieve() returned None after store()")
            k_rec, v_rec = retrieved
            attn_output = score_attention_reconstructed(
                queries,
                keys,  # packet placeholders
                values,
                dequant_fn=lambda _kp, _vp: (k_rec, v_rec),
            )

        elif scoring_mode == "prepared":
            # Prepared cache mode: reuse dequantized blocks across decode steps
            cache_key = f"{layer_id}:{skill_pattern}"
            prep = self._prepared_cache.get(cache_key)
            if prep is None or not prep.is_valid_for(skill_pattern):
                t_deq = time.monotonic()
                retrieved = manager.retrieve(skill_pattern)
                dequant_time_ms = (time.monotonic() - t_deq) * 1000.0
                if retrieved is None:
                    raise RuntimeError(
                        "retrieve() returned None after store()"
                    )
                k_prep, v_prep = retrieved
                prep = PreparedCache(
                    skill_pattern=skill_pattern,
                    dequantized_k=k_prep,
                    dequantized_v=v_prep,
                    misses=1,
                )
                self._prepared_cache[cache_key] = prep
            else:
                k_prep = prep.dequantized_k
                v_prep = prep.dequantized_v
                prep.hits += 1
                if k_prep is None or v_prep is None:
                    raise RuntimeError(
                        "PreparedCache has stale dequantized blocks"
                    )
            attn_output = score_attention_prepared(
                queries, k_prep, v_prep
            )

        elif scoring_mode == "packed_block":
            if block_indices is None:
                raise ValueError(
                    "packed_block scoring requires block_indices"
                )
            t_deq = time.monotonic()
            retrieved = manager.retrieve_blocks(
                skill_pattern,
                block_indices=block_indices,
                block_size=manager.block_size,
            )
            dequant_time_ms = (time.monotonic() - t_deq) * 1000.0
            if retrieved is None:
                raise RuntimeError(
                    "retrieve_blocks() returned None after store()"
                )
            k_blk, v_blk = retrieved
            attn_output = score_attention_packed_block(
                queries,
                keys,
                values,
                block_indices=block_indices,
                block_dequant_fn=lambda _kp, _vp, _bi: (k_blk, v_blk),
            )

        elif scoring_mode == "quantized_metal":
            # Fused Metal kernel: never materialises full K/V tensors
            cache = manager.active_caches.get(skill_pattern)
            if cache is None:
                raise RuntimeError(
                    "quantized_metal requires active cache for skill_pattern"
                )
            t_deq = time.monotonic()
            attn_output = score_attention_quantized_metal(
                queries,
                packed_k=cache.k_packed,
                packed_v=cache.v_packed,
                scales_k=cache.k_scales,
                scales_v=cache.v_scales,
                n_keys=cache.token_count,
                bits=cache.k_bits,
                group_size=manager.group_size,
            )
            dequant_time_ms = (time.monotonic() - t_deq) * 1000.0

        else:
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

        mx.eval(attn_output)
        total_time = time.monotonic() - t_start
        tokens_per_sec = token_count / max(total_time, 1e-6)

        # ------------------------------------------------------------------
        # 4. Audit (if reference logits provided)
        # ------------------------------------------------------------------
        fallback_event: str | None = None
        quality_audit: dict[str, Any] | None = None
        audit_metrics: AuditMetrics | None = None

        if reference_logits is not None:
            # We treat attn_output as a proxy for "compressed logits" in this
            # experimental path.  For a real model you would compare the
            # full-model logits instead.
            audit_metrics = audit_decode_step(
                compressed_logits=attn_output,
                reference_logits=reference_logits,
                step_num=self._step_num,
                audit_interval=10,
            )
            if audit_metrics is not None:
                fallback_event = check_drift(audit_metrics)
                quality_audit = {
                    "logit_cosine": audit_metrics.logit_cosine,
                    "top5_overlap": audit_metrics.top5_overlap,
                    "kl_divergence": audit_metrics.kl_divergence,
                    "nll_delta": audit_metrics.nll_delta,
                    "has_nan_inf": audit_metrics.has_nan_inf,
                }
                if fallback_event:
                    _log_fallback_event(
                        fallback_event, self._step_num, self.telemetry_dir
                    )
                    self.state.fallback_count += 1

                # Feed metrics to the live controller if enabled.
                if (
                    self._live_controller is not None
                    and audit_metrics is not None
                ):
                    self._live_controller.update(
                        audit_metrics, self._step_num
                    )

        # ------------------------------------------------------------------
        # 5. Telemetry
        # ------------------------------------------------------------------
        self.state.total_compressed_bytes += compressed_bytes
        self.state.total_dequant_time_ms += dequant_time_ms
        self.state.total_tokens += token_count

        event = QuantTelemetryEvent(
            task_id=task_id,
            model_id=self.model_id,
            layer_id=layer_id,
            quant_mode=self.quant_mode,
            layer_policy_applied=policy_applied,
            compressed_bytes=compressed_bytes,
            dequant_time_ms=dequant_time_ms,
            tokens_per_sec=tokens_per_sec,
            fallback_event=fallback_event,
            quality_audit=quality_audit,
        )
        _log_telemetry_event(event, self.telemetry_dir)

        info: dict[str, Any] = {
            "task_id": task_id,
            "quant_mode": self.quant_mode,
            "layer_policy_applied": policy_applied,
            "effective_k_bits": effective_k_bits,
            "effective_v_bits": effective_v_bits,
            "effective_group_size": effective_group_size,
            "compressed_bytes": compressed_bytes,
            "dequant_time_ms": dequant_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "fallback_event": fallback_event,
            "audit_metrics": quality_audit,
            "controller_adjusted": controller_adjusted,
        }
        if self._live_controller is not None:
            info["controller_state"] = self._live_controller.state_dict()
        return attn_output, info

    # ------------------------------------------------------------------
    # Quality audit sample export
    # ------------------------------------------------------------------

    def export_quality_audit_samples(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return the most recent *limit* quality audit samples."""
        # Samples are stored in the JSONL files; this helper returns an
        # in-memory snapshot gathered from telemetry parsing for convenience.
        samples: list[dict[str, Any]] = []
        path = self.telemetry_dir / "experimental_quant_telemetry.jsonl"
        if not path.exists():
            return samples
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("quality_audit") is not None:
                    samples.append(record)
        return samples[-limit:]

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_state_summary(self) -> dict[str, Any]:
        """Return a snapshot of the runtime state."""
        elapsed_sec = max(self.state.total_dequant_time_ms / 1000.0, 1e-3)
        return {
            "quant_mode": self.quant_mode,
            "layer_policy_layers": list(self.layer_policy.keys()),
            "total_compressed_bytes": self.state.total_compressed_bytes,
            "total_dequant_time_ms": self.state.total_dequant_time_ms,
            "total_tokens": self.state.total_tokens,
            "fallback_count": self.state.fallback_count,
            "avg_tokens_per_sec": self.state.total_tokens / elapsed_sec,
        }
