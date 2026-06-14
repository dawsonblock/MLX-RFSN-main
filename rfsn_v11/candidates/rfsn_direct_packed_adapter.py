"""Candidate: RFSN direct-packed K8/V8 (conservative quantization).

This is a direct-packed attention candidate that uses:
- K8/V8 quantization (conservative, higher quality than K8/V5)
- Direct packed attention without full dense reconstruction
- Strict no-fallback execution mode

This is the primary correctness validation candidate before pursuing
lower bit-widths or fused Metal implementations.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .memory_metrics import estimate_kv_memory_mb
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE


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
        staging_capacity: int = 64,
        dense_residual_window: int = 128,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window
        self.name = (
            f"rfsn_direct_packed_k{key_bits}v{value_bits}_gs{group_size}"
        )

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
            from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache
            from rfsn_v10.runtime.generation import RFSNGenerator
            from rfsn_v10.config import QuantizationConfig, RFSNConfig

            # Configure K8/V8 quantization
            key_codec = CartesianCodec(bits=self.key_bits, group_size=self.group_size)
            value_codec = CartesianCodec(bits=self.value_bits, group_size=self.group_size)
            session = GenerationCacheSession(
                name="direct_packed_teacher_forced",
                num_layers=len(model.layers),
                key_codec=key_codec,
                value_codec=value_codec,
            )
            cache_list = [
                RfsnQuantizedKVCache(
                    layer_cache=session.get_layer_cache(i),
                    session=session,
                )
                for i in range(len(model.layers))
            ]

            # Configure generator with packed_reference=True
            cfg = RFSNConfig(
                quantization=QuantizationConfig(
                    default_bits=self.key_bits,
                    group_size=self.group_size,
                ),
                runtime=RFSNConfig.RuntimeConfig(
                    strict_packed_mode=True,  # Strict no-fallback
                ),
            )
            generator = RFSNGenerator(
                model,
                tokenizer,
                cfg,
                enable_quantized_kv=True,
                packed_reference=True,  # Enable direct packed attention
                staging_capacity=self.staging_capacity,
                dense_residual_window=self.dense_residual_window,
            )

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
                f"{len(logprob_list)} log-probs for {len(gen_ids)} tokens"
            )

            # Collect proof counters from the session
            counters = session.counters()
            try:
                n_layers = len(model.layers)
            except Exception:
                n_layers = 0
            counters["layers_active"] = n_layers
            self._last_runtime_counters = counters

            # Verify no dense fallback occurred
            if counters.get("dense_fallback_calls", 0) > 0:
                raise RuntimeError(
                    f"Strict mode violation: {counters['dense_fallback_calls']} "
                    "dense fallback calls detected"
                )

            # Store detailed runtime counters for instrumentation
            self._runtime_counters = {
                "packed_attention_calls": counters.get("packed_attention_calls", 0),
                "dense_fallback_calls": counters.get("dense_fallback_calls", 0),
                "packed_bytes_read": counters.get("packed_bytes_read", 0),
                "packed_bytes_written": counters.get("packed_bytes_written", 0),
                "decoded_block_bytes": counters.get("decoded_block_bytes", 0),
                "scratch_bytes_peak": counters.get("scratch_bytes_peak", 0),
                "block_seal_events": counters.get("block_seal_events", 0),
                "execution_backend": counters.get("execution_backend", "unknown"),
            }

            return np.stack(logprob_list, axis=0)
        except Exception as exc:
            print(f"ERROR in capture_logprobs for {self.name}: {exc}")
            return None

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

            from rfsn_v10.config import QuantizationConfig, RFSNConfig
            from rfsn_v10.runtime.generation import RFSNGenerator

            cfg = RFSNConfig(
                quantization=QuantizationConfig(
                    default_bits=self.key_bits,
                    group_size=self.group_size,
                ),
                runtime=RFSNConfig.RuntimeConfig(
                    strict_packed_mode=True,  # Strict no-fallback
                ),
            )
            generator = RFSNGenerator(
                model,
                tokenizer,
                cfg,
                enable_quantized_kv=True,
                packed_reference=True,  # Enable direct packed attention
                staging_capacity=self.staging_capacity,
                dense_residual_window=self.dense_residual_window,
            )

            # Suppress mlx-lm deprecated-arg print()s from internals
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = list(generator.generate(
                    prompt, max_new_tokens=max_tokens, temperature=temp,
                ))
            total_ms = (time.perf_counter() - t0) * 1000
            result_text = "".join(tokens)

            gen_tokens = max(len(tokens), 1)
            tps = gen_tokens / (total_ms / 1000)

            actual_kv_memory_mb = estimate_kv_memory_mb(
                model, tokenizer, prompt, gen_tokens,
                bits=self.key_bits,  # Use key_bits for estimate
            )
            size_ratio = self.key_bits / 16.0
            compression_factor = 16.0 / self.key_bits

            # Check runtime counters for fallback and collect instrumentation
            packed_attention_calls = 0
            dense_fallback_calls = 0
            packed_bytes_read = 0
            packed_bytes_written = 0
            decoded_block_bytes = 0
            scratch_bytes_peak = 0
            block_seal_events = 0
            execution_backend = "unknown"

            if hasattr(generator, "_last_counters"):
                counters = generator._last_counters
                packed_attention_calls = counters.get("packed_attention_calls", 0)
                dense_fallback_calls = counters.get("dense_fallback_calls", 0)
                packed_bytes_read = counters.get("packed_bytes_read", 0)
                packed_bytes_written = counters.get("packed_bytes_written", 0)
                decoded_block_bytes = counters.get("decoded_block_bytes", 0)
                scratch_bytes_peak = counters.get("scratch_bytes_peak", 0)
                block_seal_events = counters.get("block_seal_events", 0)
                execution_backend = counters.get("execution_backend", "unknown")

                if counters.get("dense_fallback_calls", 0) > 0:
                    return CandidateResult(
                        name=self.name,
                        model_id=getattr(model, "name_or_path", "unknown"),
                        prompt=prompt,
                        gate_status="ERROR",
                        error=f"Strict mode violation: {counters['dense_fallback_calls']} dense fallback calls",
                        promotion_eligible=False,
                        packed_attention_calls=packed_attention_calls,
                        dense_fallback_calls=dense_fallback_calls,
                        packed_bytes_read=packed_bytes_read,
                        packed_bytes_written=packed_bytes_written,
                        decoded_block_bytes=decoded_block_bytes,
                        scratch_bytes_peak=scratch_bytes_peak,
                        block_seal_events=block_seal_events,
                        execution_backend=execution_backend,
                    )

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=result_text,
                actual_kv_memory_mb=actual_kv_memory_mb,
                size_ratio=size_ratio,
                compression_factor=compression_factor,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                promotion_eligible=False,
                cache_backend_used="rfsn_v10_direct_packed",
                notes="Direct packed attention with K8/V8 quantization (strict mode)",
                packed_attention_calls=packed_attention_calls,
                dense_fallback_calls=dense_fallback_calls,
                packed_bytes_read=packed_bytes_read,
                packed_bytes_written=packed_bytes_written,
                decoded_block_bytes=decoded_block_bytes,
                scratch_bytes_peak=scratch_bytes_peak,
                block_seal_events=block_seal_events,
                execution_backend=execution_backend,
            )
        except Exception as exc:
            import traceback
            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                gate_status="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                promotion_eligible=False,
            )
