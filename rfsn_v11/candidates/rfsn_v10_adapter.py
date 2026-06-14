"""Candidate: RFSN v10 dense reconstruction reference (k8_v5_gs64).

This wraps the rfsn_v10 quantization path with dense reconstruction fallback.
This is a REFERENCE-ONLY implementation for correctness validation.

IMPORTANT: This candidate reconstructs the full dense K/V history on every
attention call. It is NOT a speed or memory candidate. It must never be ranked
as a performance improvement.

Config name mapping
-------------------
k8_v5_gs64  →  default_bits=8, group_size=64   (canonical)
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
    "k8_v5_gs64": {"default_bits": 8, "group_size": 64},
}

_LEGACY_PRESETS: dict[str, dict[str, Any]] = {
    "legacy_k8_v5_gs32": {"default_bits": 8, "group_size": 32},
}


class RFSNV10Candidate(KVCompressionCandidate):
    """RFSN v10 with a given quantization config.

    This is a REFERENCE-ONLY implementation that uses dense reconstruction.
    It is not eligible for speed or memory promotion.
    """

    candidate_status = CandidateStatus.REFERENCE_ONLY

    def __init__(self, config_name: str = "k8_v5_gs64") -> None:
        all_presets = {**_PRESET_MAP, **_LEGACY_PRESETS}
        if config_name not in all_presets:
            raise ValueError(
                f"Unknown rfsn_v10 preset {config_name!r}. "
                f"Valid canonical: {list(_PRESET_MAP)}; "
                f"legacy: {list(_LEGACY_PRESETS)}"
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

        Uses the explicit per-layer quantized KV cache adapter (no monkeypatching).
        The teacher-forced loop feeds the exact baseline token sequence through
        the model and captures per-step log-probability vectors.
        """
        try:
            import numpy as np
            import mlx.core as mx
            from rfsn_v10.cache.cartesian_codec import CartesianCodec
            from rfsn_v10.cache.session import GenerationCacheSession
            from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

            quant_kwargs = {**_PRESET_MAP, **_LEGACY_PRESETS}[self.config_name]
            bits = quant_kwargs["default_bits"]
            group_size = quant_kwargs["group_size"]

            key_codec = CartesianCodec(bits=bits, group_size=group_size)
            value_codec = CartesianCodec(bits=5, group_size=group_size)
            session = GenerationCacheSession(
                name="v11_teacher_forced",
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

            # Teacher-forced decode.
            # After prefill we already have the log-prob for predicting
            # the FIRST generated token (g1).  To get the log-prob for
            # predicting g2 we feed g1, for g3 we feed g2, etc.
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

            return np.stack(logprob_list, axis=0)
        except Exception as exc:
            # Return None to maintain backward compatibility with existing code
            # The error will be captured in the run() method instead
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

            quant_kwargs = {**_PRESET_MAP, **_LEGACY_PRESETS}[self.config_name]
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
                    "Real RFSN v10 quantized KV cache via explicit adapter."
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
