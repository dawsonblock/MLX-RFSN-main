"""Candidate: TurboPolar — PolarQuant + optional QJL + optional fused Metal.

Build order enforced by the candidate itself:
  1. Offline PolarQuant encoder/decoder must pass attention-score gate.
  2. Offline QJL must improve score error (otherwise disabled).
  3. Teacher-forced logit comparison against baseline must pass.
  4. Runtime cache with real counters must be proven.
  5. Fused Metal kernel must match Python reference.
  6. Online softmax with dense V must match dense attention output.
  7. Only then is promotion even considered.

Default baseline remains rfsn_v10_k8_v5_gs64.
TurboPolar is EXPERIMENTAL and never the default.
"""
from __future__ import annotations

import time
from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus, get_status_for_name
from .quality_gates import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_PENDING_LOGIT_GATE,
    evaluate_quality_gate,
    logit_quality_metrics,
)
from .turbo_polar_config import TurboPolarConfig
from .turbo_polar_trace import TurboPolarTrace


class _TurboPolarMLXCache:
    """MLX-LM compatible KV cache backed by TurboPolar compression.

    Keys are compressed via PolarQuant on every ``update_and_fetch``.
    Values are kept dense (fp16).  Decompressed keys are concatenated
    incrementally so attention receives the full sequence of reconstructed
    keys.  This is O(N) per step because only the newest block is
    decompressed.
    """

    def __init__(self, config: TurboPolarConfig) -> None:
        from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCache
        self._tp_cache = TurboPolarKVCache(config)
        self.offset = 0
        self._cached_keys: mx.array | None = None
        self._cached_values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Append *keys*/*values*, return full cached sequence."""
        self._tp_cache.update(keys, values)
        block_idx = len(self._tp_cache.key_blocks) - 1
        new_k = self._tp_cache.fetch_keys_for_block(block_idx)
        new_v = self._tp_cache.fetch_values_for_block(block_idx)
        if self._cached_keys is None:
            self._cached_keys = new_k
            self._cached_values = new_v
        else:
            self._cached_keys = mx.concatenate([self._cached_keys, new_k], axis=2)
            self._cached_values = mx.concatenate([self._cached_values, new_v], axis=2)
        self.offset = int(self._cached_keys.shape[2])
        return self._cached_keys, self._cached_values

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        return self._cached_keys, self._cached_values


class TurboPolarAdapter(KVCompressionCandidate):
    """TurboPolar candidate adapter.

    Parameters
    ----------
    config
        TurboPolarConfig instance. Defaults to the first experimental preset.
    """

    candidate_status = CandidateStatus.EXPERIMENTAL

    def __init__(self, config: TurboPolarConfig | None = None) -> None:
        self.cfg = config or TurboPolarConfig()
        self.name = self.cfg.candidate_name
        self._trace = TurboPolarTrace()

    def is_available(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_trace(self) -> TurboPolarTrace:
        """Return the current trace, resetting internal state."""
        return self._trace

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        """Run generation with TurboPolar-compressed KV cache."""
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                gate_status="ERROR",
                error="mlx or mlx_lm not importable",
            )

        t0 = time.perf_counter()

        try:
            n_layers = len(model.layers)
            cache_list = [_TurboPolarMLXCache(self.cfg) for _ in range(n_layers)]

            prompt_ids = tokenizer.encode(prompt)
            y = mx.array(prompt_ids)
            while y.size > 512:
                model(y[:512][None], cache=cache_list)
                y = y[512:]

            logits = model(y[None], cache=cache_list)
            logits = logits[:, -1, :]

            generated_tokens = 0
            tokens: list[int] = []
            for _ in range(max_tokens):
                if temp == 0.0:
                    next_token = int(mx.argmax(logits, axis=-1).item())
                else:
                    probs = mx.softmax(logits / temp)
                    next_token = int(mx.random.categorical(mx.log(probs)).item())
                tokens.append(next_token)
                generated_tokens += 1
                if next_token == tokenizer.eos_token_id:
                    break
                logits = model(mx.array([next_token])[None], cache=cache_list)
                logits = logits[:, -1, :]

            result_text = tokenizer.decode(tokens)
            total_ms = (time.perf_counter() - t0) * 1000
            tps = generated_tokens / (total_ms / 1000) if total_ms > 0 else 0.0

            # Compute actual KV bytes from the cache
            total_bytes = 0
            for cache in cache_list:
                total_bytes += cache._tp_cache.bytes_written_actual
            actual_kv_mb = total_bytes / (1024 * 1024)

            trace = self._build_trace()
            trace.real_cache_used = True
            trace.cache_backend_used = "turbo_polar_k_only"
            trace.cache_bytes_written_actual = total_bytes
            trace.methodology_status = "RUNTIME_GENERATION_COMPLETE"
            trace.mark_event("run_called")
            trace.mark_event("runtime_generation_complete")

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=generated_tokens,
                generated_text=result_text,
                actual_kv_memory_mb=actual_kv_mb,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                promotion_eligible=False,
                candidate_status=CandidateStatus.EXPERIMENTAL,
                cache_backend_used=trace.cache_backend_used,
                cache_events=trace.events,
                cache_bytes_written=trace.cache_bytes_written_actual,
                cache_bytes_read=trace.cache_bytes_read_actual,
                notes=f"TurboPolar runtime generation. config={self.cfg}",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                gate_status="ERROR",
                error=str(exc),
            )

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_text: str | None = None,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> Any:
        """Teacher-forced logit capture using TurboPolar-compressed KV cache.

        Creates a custom MLX-LM cache for every layer that compresses keys
        with PolarQuant and stores values dense.  The model is then run
        teacher-forced over the exact *target_text* token sequence and
        per-step log-probability vectors are returned.
        """
        if not self.is_available() or target_text is None:
            return None
        try:
            import numpy as np

            prompt_ids = tokenizer.encode(prompt)
            target_ids = tokenizer.encode(target_text)
            if (
                len(target_ids) >= len(prompt_ids)
                and target_ids[: len(prompt_ids)] == prompt_ids
            ):
                gen_ids = target_ids[len(prompt_ids):]
            else:
                gen_ids = target_ids
            if not gen_ids:
                return None

            # Build a TurboPolar cache for every layer
            n_layers = len(getattr(model, "layers", []))
            cache_list = [
                _TurboPolarMLXCache(self.cfg)
                for _ in range(n_layers)
            ]

            # Prefill
            y = mx.array(prompt_ids)
            while y.size > 512:
                model(y[:512][None], cache=cache_list)
                y = y[512:]

            logprob_list: list[np.ndarray] = []
            # Prefill final chunk + first decode prediction
            logits = model(y[None], cache=cache_list)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            lp_np = np.array(logprobs.astype(mx.float32).squeeze(0))
            logprob_list.append(lp_np)

            # Teacher-forced decode over remaining generated tokens
            for forced_token_id in gen_ids[:-1]:
                logits = model(
                    mx.array([forced_token_id])[None], cache=cache_list
                )
                logits = logits[:, -1, :]
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                lp_np = np.array(
                    logprobs.astype(mx.float32).squeeze(0)
                )
                logprob_list.append(lp_np)

            assert len(logprob_list) == len(gen_ids)
            return np.stack(logprob_list, axis=0)
        except Exception:
            return None
