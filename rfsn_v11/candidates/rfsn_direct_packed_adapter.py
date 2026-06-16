"""Candidate: RFSN direct-packed attention (configurable bit-width).

This is a direct-packed attention candidate that uses:
- Configurable K/V bit-widths (K16/V16, K8/V16, K16/V8, K8/V8, K8/V6, K8/V5)
- Direct packed attention without full dense reconstruction
- Strict no-fallback execution mode

Bit-width isolation ladder for correctness validation:
- K16/V16: Near-lossless reference (baseline for quality comparison)
- K8/V16: Conservative value quantization
- K16/V8: Conservative key quantization
- K8/V8: Balanced quantization (primary validation target)
- K8/V6: Aggressive value quantization
- K8/V5: Maximum compression (original RFSN target)
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .memory_metrics import estimate_kv_memory_mb
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE


@dataclass
class CandidateExecutionError(Exception):
    """Structured error for candidate execution failures."""
    candidate: str
    stage: str
    exception_type: str
    exception_message: str
    traceback_path: str

    def __str__(self) -> str:
        return (
            f"CandidateExecutionError({self.candidate}, {self.stage}, "
            f"{self.exception_type}: {self.exception_message})"
        )


class RFSNDirectPackedCandidate(KVCompressionCandidate):
    """RFSN direct-packed attention with K8/V8 quantization.

    This candidate uses packed_reference=True to enable direct packed
    attention without full dense reconstruction. It runs in strict mode
    where any fallback to dense attention immediately fails.
    """

    candidate_status = CandidateStatus.EXPERIMENTAL

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 8,
        group_size: int = 64,
        staging_capacity: int = 64,  # Canonical block size
        dense_residual_window: int = 0,
    ) -> None:
        # P0-1: Direct packed Metal currently supports only K8/V8 GS64.
        if key_bits != 8 or value_bits != 8 or group_size != 64:
            raise ValueError(
                "RFSNDirectPackedCandidate currently requires K8/V8 GS64; "
                f"got K{key_bits}/V{value_bits} GS{group_size}"
            )
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window
        # Only append _bs suffix for non-canonical block sizes
        # (smoke/test variants). Canonical BS64 uses the plain name:
        # rfsn_direct_packed_k{k}v{v}_gs{gs}
        _bs_suffix = (
            f"_bs{staging_capacity}" if staging_capacity != 64 else ""
        )
        self.name = (
            f"rfsn_direct_packed_k{key_bits}v{value_bits}"
            f"_gs{group_size}{_bs_suffix}"
        )

    @property
    def supports_teacher_forced_capture(self) -> bool:
        return True

    def is_available(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
            import rfsn_v10  # noqa: F401
            return True
        except ImportError:
            return False

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_text: str,
        max_tokens: int = 200,
        temp: float = 0.0,
        max_context_tokens: int = 16384,
    ) -> Any:
        """Capture teacher-forced log-probs with direct packed attention.

        Uses packed_reference=True and strict mode to ensure no fallback
        to dense reconstruction occurs.
        """
        try:
            import numpy as np
            import mlx.core as mx
            from rfsn_v10.cache.cartesian_codec import CartesianCodec
            from rfsn_v10.cache.session import GenerationCacheSession
            from rfsn_v10.integrations.mlx_lm_model_support import (
                RfsnDirectPackedKVCache,
                packed_attention_context,
            )
            # Configure K8/V8 quantization
            key_codec = CartesianCodec(
                bits=self.key_bits, group_size=self.group_size
            )
            value_codec = CartesianCodec(
                bits=self.value_bits, group_size=self.group_size
            )

            # P0-6: Direct packed candidate must request persistent paged storage.
            import math

            max_pages = math.ceil(max_context_tokens / self.staging_capacity)
            session = GenerationCacheSession(
                model_id="direct_packed_teacher_forced",
                num_layers=len(model.layers),
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=self.staging_capacity,
                dense_residual_window=self.dense_residual_window,
                use_paged_arena=True,
                max_pages=max_pages,
            )

            # P0 Fix: Use try/finally to ensure session destruction
            try:
                # Create direct-packed caches
                cache_list = [
                    RfsnDirectPackedKVCache(
                        layer_id=i,
                        key_codec=key_codec,
                        value_codec=value_codec,
                        staging_capacity=self.staging_capacity,
                        dense_residual_window=self.dense_residual_window,
                        strict=True,  # Strict mode for validation
                        session=session,
                    )
                    for i in range(len(model.layers))
                ]

                # Use context manager to ensure wrapper cleanup
                with packed_attention_context(model, cache_list, strict=True):
                    prompt_ids = tokenizer.encode(prompt)
                    target_ids = tokenizer.encode(target_text)

                    # Same logic as capture_teacher_forced_logprobs
                    if (
                        len(target_ids) >= len(prompt_ids)
                        and target_ids[: len(prompt_ids)] == prompt_ids
                    ):
                        gen_ids = target_ids[len(prompt_ids):]
                    else:
                        gen_ids = target_ids

                    if not gen_ids:
                        return None

                    # Prefill
                    y = mx.array(prompt_ids)
                    while y.size > 512:
                        model(y[:512][None], cache=cache_list)
                        y = y[512:]
                    prefill_logits = model(y[None], cache=cache_list)
                    prefill_logits = prefill_logits[:, -1, :]
                    prefill_logprobs = prefill_logits - mx.logsumexp(
                        prefill_logits, keepdims=True
                    )
                    first_lp = np.array(
                        prefill_logprobs.astype(mx.float32).squeeze(0)
                    )

                    # Teacher-forced decode
                    logprob_list: list[np.ndarray] = [first_lp]
                    for forced_token_id in gen_ids[:-1]:
                        logits = model(
                            mx.array([forced_token_id])[None], cache=cache_list
                        )
                        logits = logits[:, -1, :]
                        logprobs = logits - mx.logsumexp(
                            logits, keepdims=True
                        )
                        lp_np = np.array(
                            logprobs.astype(mx.float32).squeeze(0)
                        )
                        logprob_list.append(lp_np)

                    assert len(logprob_list) == len(gen_ids), (
                        f"Teacher-forced length mismatch: "
                        f"{len(logprob_list)} log-probs for "
                        f"{len(gen_ids)} tokens"
                    )

                    # Collect proof counters from the session
                    try:
                        n_layers = len(model.layers)
                    except Exception:
                        n_layers = 0
                    self._last_runtime_counters = (
                        session.runtime_counters.to_dict()
                    )
                    self._last_runtime_counters["layers_active"] = n_layers
                    self._last_runtime_counters["requested_strict_mode"] = True
                    self._last_runtime_counters["effective_strict_mode"] = True

                    # P0 Fix: Capture memory report before session destruction
                    try:
                        memory_report = session.memory_report()
                        self._last_runtime_counters["memory_report"] = (
                            memory_report.to_dict()
                        )
                    except Exception:
                        pass  # Memory report is best-effort

                    # Verify no dense fallback occurred using unified counters
                    df_calls = session.runtime_counters.dense_fallback_calls
                    if df_calls > 0:
                        raise RuntimeError(
                            f"Strict mode violation: {df_calls} "
                            "dense fallback calls detected"
                        )

                    # Store detailed runtime counters for instrumentation
                    self._runtime_counters = session.runtime_counters.to_dict()

                    return np.stack(logprob_list, axis=0)
            finally:
                # P0 Fix: Always destroy session to free memory
                session.destroy()
        except Exception as exc:
            # Raise structured error instead of swallowing
            raise CandidateExecutionError(
                candidate=self.name,
                stage="capture_logprobs",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback_path="".join(traceback.format_tb(exc.__traceback__)),
            ) from exc

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error="rfsn_v10 or mlx_lm not importable",
            )
        try:
            import contextlib
            import io

            from rfsn_v10.runtime.generation import RFSNGenerator
            from rfsn_v10.config import (
                RFSNConfig,
                RuntimeConfig,
            )

            # Pass explicit strict configuration into normal generation
            # Construct explicit runtime config with strict_packed_mode=True
            # Do not derive benchmark semantics from env vars or defaults
            runtime_config = RuntimeConfig(
                strict_packed_mode=True,
            )
            explicit_config = RFSNConfig(
                runtime=runtime_config,
            )

            # Pass exact key and value bits to generator
            # Candidate bits must be used, not generator defaults (8, 5)
            generator = RFSNGenerator(
                model,
                tokenizer,
                config=explicit_config,
                enable_quantized_kv=True,
                packed_reference=True,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                group_size=self.group_size,
                staging_capacity=self.staging_capacity,
                dense_residual_window=self.dense_residual_window,
            )

            # Force strict mode on the adapter directly (bypass config loading)
            if generator._adapter:
                generator._adapter.strict = True

            # Suppress mlx-lm deprecated-arg print()s from internals
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = list(generator.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                ))
            total_ms = (time.perf_counter() - t0) * 1000
            result_text = "".join(tokens)

            gen_tokens = max(len(tokens), 1)
            tps = gen_tokens / (total_ms / 1000)

            # P0 #8: Use actual measured memory from generator when available
            actual_kv_memory_mb = None
            estimated_kv_memory_mb = estimate_kv_memory_mb(
                model,
                tokenizer,
                prompt,
                gen_tokens,
                bits=self.key_bits,
            )
            size_ratio = self.key_bits / 16.0
            compression_factor = 16.0 / self.key_bits
            measurement_kind = "ESTIMATED"

            # Check runtime counters for fallback and collect instrumentation
            # Generator now outputs flattened counters, not nested
            packed_attention_calls = 0
            dense_fallback_calls = 0
            packed_bytes_read = 0
            packed_bytes_written = 0
            decoded_block_bytes = 0
            scratch_bytes_peak = 0
            execution_backend = "unknown"
            packed_blocks_created = 0
            packed_blocks_read = 0
            full_history_materialization_calls = 0

            if hasattr(generator, "_last_counters"):
                counters = generator._last_counters
                packed_attention_calls = counters.get(
                    "packed_attention_calls", packed_attention_calls
                )
                dense_fallback_calls = counters.get(
                    "dense_fallback_calls", dense_fallback_calls
                )
                packed_bytes_read = counters.get(
                    "packed_bytes_read", packed_bytes_read
                )
                packed_bytes_written = counters.get(
                    "packed_bytes_written", packed_bytes_written
                )
                decoded_block_bytes = counters.get(
                    "decoded_block_bytes", decoded_block_bytes
                )
                scratch_bytes_peak = counters.get(
                    "scratch_bytes_peak", scratch_bytes_peak
                )
                execution_backend = counters.get(
                    "execution_backend", "unknown"
                )
                packed_blocks_created = counters.get(
                    "packed_blocks_created", packed_blocks_created
                )
                packed_blocks_read = counters.get(
                    "packed_blocks_read", packed_blocks_read
                )
                full_history_materialization_calls = counters.get(
                    "full_history_materialization_calls",
                    full_history_materialization_calls
                )

                # P0 #8: Try to get actual measured memory from generator
                mem_report = counters.get("_last_memory_report", {})
                if mem_report:
                    payload_bytes = mem_report.get("payload_bytes", 0)
                    if payload_bytes > 0:
                        actual_kv_memory_mb = payload_bytes / (1024 * 1024)
                        measurement_kind = "MEASURED"

                if dense_fallback_calls > 0:
                    return CandidateResult(
                        name=self.name,
                        model_id=getattr(model, "name_or_path", "unknown"),
                        prompt=prompt,
                        gate_status="ERROR",
                        error=(
                            f"Strict mode violation: "
                            f"{dense_fallback_calls} dense fallback calls"
                        ),
                        promotion_eligible=False,
                        packed_attention_calls=packed_attention_calls,
                        dense_fallback_calls=dense_fallback_calls,
                        packed_bytes_read=packed_bytes_read,
                        packed_bytes_written=packed_bytes_written,
                        decoded_block_bytes=decoded_block_bytes,
                        scratch_bytes_peak=scratch_bytes_peak,
                        execution_backend=execution_backend,
                        packed_blocks_created=packed_blocks_created,
                        packed_blocks_read=packed_blocks_read,
                        full_history_materialization_calls=(
                            full_history_materialization_calls
                        ),
                    )

            # Use actual measured memory when available, otherwise estimate
            kv_memory_mb = (
                actual_kv_memory_mb
                if actual_kv_memory_mb is not None
                else estimated_kv_memory_mb
            )
            memory_note = (
                "[memory measured from cache tensors]"
                if measurement_kind == "MEASURED"
                else "[memory is estimated, not runtime-measured]"
            )

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=result_text,
                actual_kv_memory_mb=kv_memory_mb,
                estimated_kv_memory_mb=estimated_kv_memory_mb,
                measurement_kind=measurement_kind,
                size_ratio=size_ratio,
                compression_factor=compression_factor,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                promotion_eligible=False,
                cache_backend_used="rfsn_v10_direct_packed",
                notes=(
                    "Direct packed attention with K8/V8 quantization"
                    f" (strict mode) {memory_note}"
                ),
                packed_attention_calls=packed_attention_calls,
                dense_fallback_calls=dense_fallback_calls,
                packed_bytes_read=packed_bytes_read,
                packed_bytes_written=packed_bytes_written,
                decoded_block_bytes=decoded_block_bytes,
                scratch_bytes_peak=scratch_bytes_peak,
                execution_backend=execution_backend,
                packed_blocks_created=packed_blocks_created,
                packed_blocks_read=packed_blocks_read,
                full_history_materialization_calls=(
                    full_history_materialization_calls
                ),
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                gate_status="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                promotion_eligible=False,
            )
