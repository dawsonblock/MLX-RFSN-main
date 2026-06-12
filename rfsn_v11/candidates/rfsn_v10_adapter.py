"""Candidate: RFSN v10 stable baseline (k8_v5_gs32 and k8_v5_gs64).

This wraps the validated rfsn_v10 quantization path so the shootout can
compare it against newer candidates on equal footing.

Config name mapping
-------------------
k8_v5_gs32  →  default_bits=8, group_size=32   (recommended)
k8_v5_gs64  →  default_bits=8, group_size=64   (also validated)
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .memory_metrics import estimate_kv_memory_mb
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE

# Map the human-readable preset names to actual QuantizationConfig kwargs.
# rfsn_v10.config.RFSNConfig has no from_preset() — we build it directly.
_PRESET_MAP: dict[str, dict[str, Any]] = {
    "k8_v5_gs32": {"default_bits": 8, "group_size": 32},
    "k8_v5_gs64": {"default_bits": 8, "group_size": 64},
}


class RFSNV10Candidate(KVCompressionCandidate):
    """RFSN v10 with a given quantization config."""

    candidate_status = CandidateStatus.BASELINE

    def __init__(self, config_name: str = "k8_v5_gs32") -> None:
        if config_name not in _PRESET_MAP:
            raise ValueError(
                f"Unknown rfsn_v10 preset {config_name!r}. "
                f"Valid: {list(_PRESET_MAP)}"
            )
        self.config_name = config_name
        self.name = f"rfsn_v10_{config_name}"

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
        """Capture teacher-forced log-probs with RFSN v10 real cache active.

        Creates an RFSNGenerator with ``enable_sparse_decode=True`` so the
        SDPA patch routes through RFSNRuntime and the quantized KV cache.
        The teacher-forced loop then feeds the exact baseline token sequence
        through the model and captures per-step log-probability vectors.
        """
        try:
            import numpy as np
            import mlx.core as mx
            from mlx_lm.models import cache as mlx_cache
            from rfsn_v10.config import QuantizationConfig, RFSNConfig
            from rfsn_v10.runtime.generation import (
                RFSNGenerator,
                _RFSNSDPAPatcher,
                _unwrap_layers_for_rfsn,
                _wrap_layers_for_rfsn,
            )

            quant_kwargs = _PRESET_MAP[self.config_name]
            cfg = RFSNConfig(
                quantization=QuantizationConfig(**quant_kwargs),
            )
            generator = RFSNGenerator(
                model,
                tokenizer,
                cfg,
                enable_quantized_kv=True,
                enable_sparse_decode=True,
                use_compressed_on_miss=True,
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

            _wrap_layers_for_rfsn(model)
            try:
                if hasattr(model, "make_cache"):
                    cache_list = model.make_cache()
                else:
                    cache_list = [
                        mlx_cache.KVCache()
                        for _ in range(len(model.layers))
                    ]

                # Prefill (outside patcher — prefill does not use SDPA decode path).
                # We intentionally do NOT call maybe_quantize_kv_cache here.
                # RFSN v10's own RFSNTurboQuantKVManager (inside RFSNRuntime)
                # handles quantization during decode-step interception.
                # Calling mlx_lm's maybe_quantize_kv_cache would replace the
                # caches with QuantizedKVCache, whose update_and_fetch returns
                # tuples that the RFSNRuntime SDPA wrapper cannot handle,
                # causing the wrapper to fall through and producing zero counters.
                y = mx.array(prompt_ids)
                while y.size > 512:
                    model(y[:512][None], cache=cache_list)
                    y = y[512:]
                # Final prefill chunk (also the only chunk for short prompts).
                # Capture logits for the first generated token here.
                prefill_logits = model(y[None], cache=cache_list)
                prefill_logits = prefill_logits[:, -1, :]
                prefill_logprobs = prefill_logits - mx.logsumexp(
                    prefill_logits, keepdims=True
                )
                first_lp = np.array(
                    prefill_logprobs.astype(mx.float32).squeeze(0)
                )

                patcher = _RFSNSDPAPatcher(generator._runtime)
                patcher.__enter__()
                logprob_list: list[np.ndarray] = [first_lp]
                try:
                    # Teacher-forced decode.
                    # After prefill we already have the log-prob for predicting
                    # the FIRST generated token (g1).  To get the log-prob for
                    # predicting g2 we feed g1, for g3 we feed g2, etc.
                    # Loop over gen_ids[:-1] — every token except the last.
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
                finally:
                    patcher.__exit__(None, None, None)

                assert len(logprob_list) == len(gen_ids), (
                    f"Teacher-forced length mismatch: "
                    f"{len(logprob_list)} log-probs for {len(gen_ids)} tokens"
                )

                # Collect runtime counters from the RFSNRuntime
                if generator._runtime is not None:
                    counters = generator._runtime.get_counters()
                    try:
                        n_layers = len(model.layers)
                    except Exception:
                        n_layers = 0
                    # Count prefill events: each prefill chunk processes all layers.
                    # The RFSNRuntime is only active during decode, so prefill
                    # events are reported via record_prefill_quantize.
                    prefill_chunks = max(1, (len(prompt_ids) + 511) // 512)
                    generator._runtime.record_prefill_quantize(
                        n_layers * prefill_chunks
                    )
                    counters.layers_wrapped_actual = n_layers
                    self._last_runtime_counters = counters.as_dict()
                else:
                    self._last_runtime_counters = None

                return np.stack(logprob_list, axis=0)
            finally:
                _unwrap_layers_for_rfsn(model)
        except Exception:
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

            quant_kwargs = _PRESET_MAP[self.config_name]
            cfg = RFSNConfig(
                quantization=QuantizationConfig(**quant_kwargs),
            )
            generator = RFSNGenerator(
                model,
                tokenizer,
                cfg,
                enable_quantized_kv=True,
                enable_sparse_decode=True,
                use_compressed_on_miss=True,
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
                bits=quant_kwargs["default_bits"],
            )
            size_ratio = quant_kwargs["default_bits"] / 16.0
            compression_factor = 16.0 / quant_kwargs["default_bits"]

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
                candidate_status=self.candidate_status,
                cache_backend_used="rfsn_v10_quantized_kv",
                cache_events=["prefill_quantize", "decode_quantized_fetch"],
                notes=(
                    f"RFSN v10 stable baseline — config={self.config_name} "
                    f"bits={quant_kwargs['default_bits']} "
                    f"gs={quant_kwargs['group_size']}  "
                    "Real RFSN v10 quantized KV cache active via SDPA patch."
                ),
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error=str(exc),
            )
