"""Candidate: MLX-LM built-in quantized KV cache.

If MLX-LM already exposes a maintained quantized KV cache that passes
quality gates, custom compression may not be necessary. This adapter
measures it fairly so the decision is data-driven.

NOTE: The ``kv_bits`` parameter availability depends on your installed mlx-lm
version.  If this candidate returns ``is_available() == False``, your installed
version does not expose quantized KV via ``generate()``.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .memory_metrics import estimate_kv_memory_mb
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE


class MLXLMQuantizedKV(KVCompressionCandidate):
    """MLX-LM generation with its built-in quantized KV cache flag."""

    name = "mlx_lm_quantized_kv"
    candidate_status = CandidateStatus.CONTROL

    def __init__(self, kv_bits: int = 8) -> None:
        self.kv_bits = kv_bits
        self.name = f"mlx_lm_quantized_kv_b{kv_bits}"

    def is_available(self) -> bool:
        # kv_bits flows through generate() **kwargs → stream_generate →
        # generate_step. Check generate_step directly since generate() uses
        # **kwargs.
        try:
            import inspect

            from mlx_lm.utils import generate_step
            sig = inspect.signature(generate_step)
            return "kv_bits" in sig.parameters
        except Exception:
            return False

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
                notes=(
                    "mlx_lm.utils.generate_step does not expose kv_bits. "
                    "Upgrade mlx-lm or skip this candidate."
                ),
                error="kv_bits parameter not available in generate_step",
            )
        try:
            import mlx_lm
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp)
            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                kv_bits=self.kv_bits,
                kv_group_size=64,
                sampler=sampler,
                verbose=False,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            input_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(output)
            gen_tokens = max(len(output_ids) - len(input_ids), 1)
            tps = gen_tokens / (total_ms / 1000)

            actual_kv_memory_mb = estimate_kv_memory_mb(
                model, tokenizer, prompt, gen_tokens, bits=self.kv_bits,
            )
            size_ratio = self.kv_bits / 16.0
            compression_factor = 16.0 / self.kv_bits

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=output,
                actual_kv_memory_mb=actual_kv_memory_mb,
                size_ratio=size_ratio,
                compression_factor=compression_factor,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                candidate_status=self.candidate_status,
                cache_backend_used="mlx_lm_quantized_kv",
                cache_events=["prefill_quantize", "decode_quantized_fetch"],
                notes=f"MLX-LM built-in {self.kv_bits}-bit KV quantization",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error=str(exc),
            )
