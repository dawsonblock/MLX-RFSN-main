"""RFSN v10 Runtime Orchestrator.

Integration layer between the KV cache manager, sparse attention engine,
memory guard, and telemetry subsystem. Coordinates a single decode step by:

1. Validating input tensors
2. Generating a composite cache key for KV retrieval/storage
3. Attempting sparse block-selective attention
4. Falling back to dense attention when sparse is unsafe
5. Optionally auditing sparse vs dense output quality
6. Recording a structured TelemetryEvent for observability
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

from ..adaptive_sparsity import AdaptiveSparsityController
from ..attention import AdaptiveBlockSparseAttention
from ..attention_reference import causal_attention_dense
from ..compat import mx
from ..kv_manager import RFSNTurboQuantKVManager
from ..memory_guard import MemoryGuard


@dataclass
class TelemetryEvent:
    """Single decode-step telemetry record."""

    task_id: str
    model_id: str
    layer_id: str
    batch_id: str
    skill_pattern: str
    seq_len: int
    head_count: int
    head_dim: int
    top_k_ratio: float
    block_size: int
    num_active_blocks: int
    effective_sparsity: float
    kv_cache_hit: bool
    kv_cache_store_latency_ms: float
    kv_cache_retrieve_latency_ms: float
    attention_latency_ms: float
    total_latency_ms: float
    fallback_used: bool
    sparse_success: bool
    dense_success: bool
    audit_enabled: bool
    audit_cosine: float | None
    audit_rel_mae: float | None
    audit_max_abs_error: float | None
    quant_audit_cosine: float | None
    quant_audit_rel_mae: float | None
    quant_audit_max_abs_error: float | None
    sparse_audit_cosine: float | None
    sparse_audit_rel_mae: float | None
    sparse_audit_max_abs_error: float | None
    execution_mode: str
    termination_reason: str


class RFSNRuntime:
    """Production-grade decode-step runtime orchestrator for RFSN v10."""

    def __init__(
        self,
        kv_manager: RFSNTurboQuantKVManager,
        model_id: str = "default",
        block_size: int = 64,
        audit_mode: bool = False,
        top_k_ratio: float = 1.0,
        enable_sparse_decode: bool = False,
        reserved_sink_blocks: int = 1,
        reserved_recent_blocks: int = 2,
        use_compressed_on_miss: bool = False,
        use_custom_kernel: bool | None = None,
        adaptive_sparsity_controller: AdaptiveSparsityController | None = None,
        memory_guard: MemoryGuard | None = None,
    ):
        self.kv_manager = kv_manager
        self.model_id = model_id
        self.block_size = block_size
        self.audit_mode = audit_mode
        self.top_k_ratio = top_k_ratio
        self.enable_sparse_decode = bool(enable_sparse_decode)
        self.reserved_sink_blocks = int(reserved_sink_blocks)
        self.reserved_recent_blocks = int(reserved_recent_blocks)
        self.use_compressed_on_miss = use_compressed_on_miss
        if use_custom_kernel is not None:
            self.kv_manager.use_custom_kernel = bool(use_custom_kernel)
        self.adaptive_sparsity_controller = adaptive_sparsity_controller
        self.memory_guard = memory_guard
        self._telemetry_log: list[TelemetryEvent] = []

    @staticmethod
    def _make_cache_key(
        model_id: str,
        layer_id: str,
        batch_id: str,
        skill_pattern: str,
        shape: tuple,
        dtype: str,
        k_bits: int,
        v_bits: int,
        group_size: int,
        format_version: str,
        use_wht: bool | None = None,
        use_incoherent_signs: bool | None = None,
        use_incoherent: bool | None = None,
    ) -> str:
        if use_wht is None and use_incoherent_signs is None:
            legacy = True if use_incoherent is None else bool(use_incoherent)
            use_wht = legacy
            use_incoherent_signs = legacy
        else:
            use_wht = True if use_wht is None else bool(use_wht)
            use_incoherent_signs = (
                True if use_incoherent_signs is None else bool(use_incoherent_signs)
            )

        return "|".join([
            model_id, layer_id, batch_id, skill_pattern,
            "x".join(str(d) for d in shape),
            str(dtype), str(k_bits), str(v_bits),
            str(group_size), str(use_wht), str(use_incoherent_signs),
            format_version,
        ])

    @staticmethod
    def _cosine_similarity(a: mx.array, b: mx.array) -> float:
        a_f = a.flatten().astype(mx.float32)
        b_f = b.flatten().astype(mx.float32)
        dot = mx.sum(a_f * b_f)
        norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
        return (dot / mx.maximum(norm, mx.array(1e-8))).item()

    @staticmethod
    def _rel_mae(a: mx.array, b: mx.array) -> float:
        a_f = a.flatten().astype(mx.float32)
        b_f = b.flatten().astype(mx.float32)
        denom = mx.maximum(mx.mean(mx.abs(a_f)), mx.array(1e-8))
        return (mx.mean(mx.abs(a_f - b_f)) / denom).item()

    @staticmethod
    def _max_abs_error(a: mx.array, b: mx.array) -> float:
        return mx.max(mx.abs(a.flatten().astype(mx.float32) - b.flatten().astype(mx.float32))).item()

    def execute_decode_step(
        self,
        skill_pattern: str,
        layer_id: str,
        batch_id: str,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float | None = None,
        reserved_sink_blocks: int | None = None,
        reserved_recent_blocks: int | None = None,
    ) -> tuple[mx.array, dict]:
        task_id = str(uuid.uuid4())
        t_start = time.monotonic()

        # 1. Validate tensors
        if len(queries.shape) != 4:
            raise ValueError(f"queries must be 4D, got {queries.shape}")
        if len(keys.shape) != 4:
            raise ValueError(f"keys must be 4D, got {keys.shape}")
        if len(values.shape) != 4:
            raise ValueError(f"values must be 4D, got {values.shape}")
        if keys.shape != values.shape:
            raise ValueError(f"keys/values shape mismatch: {keys.shape} vs {values.shape}")
        if keys.shape[-1] != queries.shape[-1]:
            raise ValueError(f"head_dim mismatch: keys={keys.shape[-1]}, queries={queries.shape[-1]}")

        B, H, T_q, D = queries.shape
        T_k = keys.shape[2]  # keys is [B, H, T_k, D]
        sink_blocks = self.reserved_sink_blocks if reserved_sink_blocks is None else int(reserved_sink_blocks)
        recent_blocks = self.reserved_recent_blocks if reserved_recent_blocks is None else int(reserved_recent_blocks)

        adaptive_decision = None
        if self.adaptive_sparsity_controller is not None:
            adaptive_decision = self.adaptive_sparsity_controller.get_decision(
                model_id=self.model_id,
                layer_id=layer_id,
                skill_pattern=skill_pattern,
                seq_len=int(T_k),
            )

        # Per-call override has highest priority, then adaptive controller, then default.
        if top_k_ratio is not None:
            effective_top_k = top_k_ratio
        elif adaptive_decision is not None:
            effective_top_k = adaptive_decision.top_k_ratio
        else:
            effective_top_k = self.top_k_ratio

        quantized_enabled = True
        sparse_enabled = True
        sparse_gate_reason = None
        if not self.enable_sparse_decode:
            sparse_enabled = False
            sparse_gate_reason = "disabled_by_default"

        if adaptive_decision is not None:
            quantized_enabled = not adaptive_decision.disable_quantized
            if self.enable_sparse_decode:
                sparse_enabled = not adaptive_decision.disable_sparse
                if adaptive_decision.disable_sparse:
                    sparse_gate_reason = "adaptive_controller_disabled"

        if self.memory_guard is not None:
            quantized_enabled = not self.memory_guard.should_disable_quantized()
            if self.memory_guard.should_disable_sparse():
                sparse_enabled = False
                sparse_gate_reason = "memory_guard_disabled"

        effective_attention_top_k = effective_top_k if sparse_enabled else 1.0

        # 2. Generate composite cache key
        cache_key = self._make_cache_key(
            model_id=self.model_id,
            layer_id=layer_id,
            batch_id=batch_id,
            skill_pattern=skill_pattern,
            shape=tuple(keys.shape),
            dtype=str(keys.dtype),
            k_bits=self.kv_manager.k_bits,
            v_bits=self.kv_manager.v_bits,
            group_size=self.kv_manager.group_size,
            use_wht=self.kv_manager.use_wht,
            use_incoherent_signs=self.kv_manager.use_incoherent_signs,
            format_version="rfsn_v10",
        )

        # Keep audit tensors local to avoid cross-request mutation races.
        original_keys = keys if self.audit_mode else None
        original_values = values if self.audit_mode else None

        # 3-4. Try retrieve or store
        retrieve_latency_ms = 0.0
        store_latency_ms = 0.0
        kv_cache_hit = False
        if quantized_enabled:
            t_retrieve_start = time.monotonic()
            kv_result = self.kv_manager.retrieve(cache_key, out_dtype=keys.dtype)
            retrieve_latency_ms = (time.monotonic() - t_retrieve_start) * 1000.0

            kv_cache_hit = kv_result is not None
            if not kv_cache_hit:
                t_store_start = time.monotonic()

                allow_store = True

                if self.memory_guard is not None:
                    estimated_cache_bytes = self.kv_manager.estimate_compressed_bytes_for_shape(
                        shape=tuple(keys.shape),
                        k_bits=self.kv_manager.k_bits,
                        v_bits=self.kv_manager.v_bits,
                        group_size=self.kv_manager.group_size,
                    )
                    self.memory_guard.enforce_safety(estimated_cache_bytes)
                    if self.memory_guard.should_disable_quantized():
                        allow_store = False
                        quantized_enabled = False
                        self.kv_manager.last_reconstruction_kernel = "quantized_disabled"
                    if self.memory_guard.should_disable_sparse():
                        sparse_enabled = False
                        effective_attention_top_k = 1.0

                if allow_store:
                    self.kv_manager.store(cache_key, keys, values, T_k)
                    store_latency_ms = (time.monotonic() - t_store_start) * 1000.0

                if allow_store and self.use_compressed_on_miss:
                    t_retrieve_check_start = time.monotonic()
                    kv_result = self.kv_manager.retrieve(cache_key, out_dtype=keys.dtype)
                    retrieve_latency_ms += (time.monotonic() - t_retrieve_check_start) * 1000.0
                    if kv_result is not None:
                        keys, values = kv_result
            else:
                if kv_result is None:
                    raise RuntimeError("KV cache reported hit but retrieve returned None")
                keys, values = kv_result
        else:
            self.kv_manager.last_reconstruction_kernel = "quantized_disabled"

        # 5. Try sparse attention
        t_attn_start = time.monotonic()
        sparse_success = False
        dense_success = False
        fallback_used = False
        attn_output = None
        num_active_blocks = 0
        termination_reason = "unknown"

        sparse_output = None
        execution_mode = "unknown"
        try:
            sparse_output, num_active_blocks, execution_mode = AdaptiveBlockSparseAttention.execute(
                queries, keys, values,
                top_k_ratio=effective_attention_top_k,
                block_size=self.block_size,
                kv_is_strictly_past=True,
                reserved_sink_blocks=sink_blocks,
                reserved_recent_blocks=recent_blocks,
                memory_guard=self.memory_guard,
            )
            mx.eval(sparse_output)
            sparse_success = (execution_mode == "sparse_compacted")
            termination_reason = execution_mode

            # If attention already returned dense output, use it directly to avoid recomputation
            if execution_mode.startswith("dense_"):
                attn_output = sparse_output
                dense_success = True
                fallback_used = execution_mode not in {"dense_requested", "dense_ragged_batch"}
        except Exception as e:
            termination_reason = f"sparse_failed: {e}"

        already_dense_output = execution_mode.startswith("dense_")
        if ((not sparse_success and not already_dense_output) or self.audit_mode):
            # 6. Dense fallback / audit
            try:
                dense_output = causal_attention_dense(
                    queries, keys, values,
                    scale=1.0 / math.sqrt(D),
                    backend="mlx",
                )
                mx.eval(dense_output)
                dense_success = True

                if not sparse_success:
                    attn_output = dense_output
                    fallback_used = True
                    termination_reason = "dense_fallback"
                    num_active_blocks = (keys.shape[2] + self.block_size - 1) // self.block_size
            except Exception as e:
                termination_reason = f"catastrophic_failure: {e}"
                raise RuntimeError(f"All attention paths failed: {e}")

        # Use sparse output if it succeeded and dense didn't override
        if sparse_success and attn_output is None:
            attn_output = sparse_output

        if attn_output is None:
            raise RuntimeError("No valid attention output produced")

        attention_latency_ms = (time.monotonic() - t_attn_start) * 1000.0

        # 8. Audit comparison - split into quantization and sparsity errors
        audit_cosine = None
        audit_rel_mae = None
        audit_max_abs_error = None
        quant_audit_cosine = None
        quant_audit_rel_mae = None
        quant_audit_max_abs_error = None
        sparse_audit_cosine = None
        sparse_audit_rel_mae = None
        sparse_audit_max_abs_error = None
        if self.audit_mode:
            # Compute dense attention from original keys/values for quantization error
            original_dense_output = causal_attention_dense(
                queries, original_keys, original_values,
                scale=1.0 / math.sqrt(D),
                backend="mlx",
            )
            mx.eval(original_dense_output)

            # Compute dense attention from working keys/values (after storage/retrieval)
            working_dense_output = causal_attention_dense(
                queries, keys, values,
                scale=1.0 / math.sqrt(D),
                backend="mlx",
            )
            mx.eval(working_dense_output)

            # Quantization error: original dense vs working dense
            quant_audit_cosine = self._cosine_similarity(original_dense_output, working_dense_output)
            quant_audit_rel_mae = self._rel_mae(original_dense_output, working_dense_output)
            quant_audit_max_abs_error = self._max_abs_error(original_dense_output, working_dense_output)

            # Sparsity error: working dense vs working sparse (if sparse succeeded)
            if sparse_success and sparse_output is not None:
                sparse_audit_cosine = self._cosine_similarity(working_dense_output, sparse_output)
                sparse_audit_rel_mae = self._rel_mae(working_dense_output, sparse_output)
                sparse_audit_max_abs_error = self._max_abs_error(working_dense_output, sparse_output)

                # For telemetry, we'll report the sparsity error as the main audit metrics
                # (this maintains backward compatibility with existing telemetry expectations)
                audit_cosine = sparse_audit_cosine
                audit_rel_mae = sparse_audit_rel_mae
                audit_max_abs_error = sparse_audit_max_abs_error

                # We could also store quantization error in separate telemetry fields if needed
                # For now, we're focusing on getting the basic split working
            else:
                # If sparse didn't succeed, fall back to comparing sparse vs dense outputs
                # (this maintains existing behavior for backward compatibility)
                if sparse_output is not None and dense_output is not None:
                    audit_cosine = self._cosine_similarity(sparse_output, dense_output)
                    audit_rel_mae = self._rel_mae(sparse_output, dense_output)
                    audit_max_abs_error = self._max_abs_error(sparse_output, dense_output)

        # Update adaptive sparsity controller if available and in audit mode
        if self.adaptive_sparsity_controller is not None and self.audit_mode:
            adaptive_decision = self.adaptive_sparsity_controller.update(
                sparse_success=sparse_success,
                fallback_used=fallback_used,
                sparse_audit_cosine=sparse_audit_cosine,
                sparse_audit_rel_mae=sparse_audit_rel_mae,
                quant_audit_cosine=quant_audit_cosine,
                quant_audit_rel_mae=quant_audit_rel_mae,
                model_id=self.model_id,
                layer_id=layer_id,
                skill_pattern=skill_pattern,
                seq_len=int(T_k),
            )

        total_latency_ms = (time.monotonic() - t_start) * 1000.0

        # Compute effective sparsity
        total_blocks = max(1, (keys.shape[2] + self.block_size - 1) // self.block_size)
        effective_sparsity = 1.0 - (num_active_blocks / total_blocks)

        # 9. Record telemetry
        event = TelemetryEvent(
            task_id=task_id,
            model_id=self.model_id,
            layer_id=layer_id,
            batch_id=batch_id,
            skill_pattern=skill_pattern,
            seq_len=int(T_k),
            head_count=int(H),
            head_dim=int(D),
            top_k_ratio=effective_top_k,
            block_size=self.block_size,
            num_active_blocks=int(num_active_blocks),
            effective_sparsity=effective_sparsity,
            kv_cache_hit=kv_cache_hit,
            kv_cache_store_latency_ms=store_latency_ms,
            kv_cache_retrieve_latency_ms=retrieve_latency_ms,
            attention_latency_ms=attention_latency_ms,
            total_latency_ms=total_latency_ms,
            fallback_used=fallback_used,
            sparse_success=sparse_success,
            dense_success=dense_success,
            audit_enabled=self.audit_mode,
            audit_cosine=audit_cosine,
            audit_rel_mae=audit_rel_mae,
            audit_max_abs_error=audit_max_abs_error,
            quant_audit_cosine=quant_audit_cosine,
            quant_audit_rel_mae=quant_audit_rel_mae,
            quant_audit_max_abs_error=quant_audit_max_abs_error,
            sparse_audit_cosine=sparse_audit_cosine,
            sparse_audit_rel_mae=sparse_audit_rel_mae,
            sparse_audit_max_abs_error=sparse_audit_max_abs_error,
            execution_mode=execution_mode,
            termination_reason=termination_reason,
        )
        self._telemetry_log.append(event)

        return attn_output, {
            "task_id": task_id,
            "kv_cache_hit": kv_cache_hit,
            "kv_reconstruction_kernel": self.kv_manager.last_reconstruction_kernel,
            "adaptive_reason": adaptive_decision.reason if adaptive_decision is not None else None,
            "quantized_enabled": quantized_enabled,
            "sparse_enabled": sparse_enabled,
            "sparse_allowed_by_gate": sparse_enabled,
            "sparse_gate_reason": sparse_gate_reason,
            "sparse_success": sparse_success,
            "dense_success": dense_success,
            "fallback_used": fallback_used,
            "reserved_sink_blocks": sink_blocks,
            "reserved_recent_blocks": recent_blocks,
            "num_active_blocks": int(num_active_blocks),
            "effective_sparsity": effective_sparsity,
            "total_latency_ms": total_latency_ms,
            "execution_mode": execution_mode,
        }

    def get_telemetry(self) -> list[TelemetryEvent]:
        """Return all recorded telemetry events."""
        return list(self._telemetry_log)

    def clear_telemetry(self) -> None:
        """Clear the telemetry log."""
        self._telemetry_log.clear()
