"""Candidate: MLX-LM built-in quantized KV cache.

If MLX-LM already exposes a maintained quantized KV cache that passes quality
gates, custom compression may not be necessary.  This adapter measures it fairly
so the decision is data-driven.

NOTE: The ``kv_bits`` parameter availability depends on your installed mlx-lm
version.  If this candidate returns ``is_available() == False``, your installed
version does not expose quantized KV via ``generate()``.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate


class MLXLMQuantizedKV(KVCompressionCandidate):
    """MLX-LM generation with its built-in quantized KV cache flag."""

    name = "mlx_lm_quantized_kv"

    def __init__(self, kv_bits: int = 8) -> None:
        self.kv_bits = kv_bits
        self.name = f"mlx_lm_quantized_kv_b{kv_bits}"

    def is_available(self) -> bool:
        try:
            import inspect
            import mlx_lm
            sig = inspect.signature(mlx_lm.generate)
            return "kv_bits" in sig.parameters
        except Exception:
            return False

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                notes=(
                    "mlx_lm.generate() does not expose kv_bits in this install. "
                    "Upgrade mlx-lm or skip this candidate."
                ),
                error="kv_bits parameter not available",
            )
        try:
            import mlx_lm

            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                kv_bits=self.kv_bits,
                verbose=False,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            input_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(output)
            gen_tokens = max(len(output_ids) - len(input_ids), 1)
            tps = gen_tokens / (total_ms / 1000)

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=output,
                passed_quality_gate=False,  # filled by shootout quality eval
                notes=f"MLX-LM built-in {self.kv_bits}-bit KV quantization",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error=str(exc),
            )
