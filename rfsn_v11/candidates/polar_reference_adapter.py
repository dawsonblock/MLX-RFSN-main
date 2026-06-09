"""Candidate: PolarQuant reference adapter.

Status: experimental / expected slower / reference only.
Uses ideas from external/mlx-turboquant (polar_quant.py + turbo_quant.py).

This adapter runs plain MLX-LM generation but records timings and notes
for the PolarQuant conceptual path. A full KV-integrated PolarQuant
implementation is tracked in rfsn_v11/quant/value_quant.py.

The purpose here is to benchmark the reference implementation's standalone
quality, not to provide a production KV cache.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate


class PolarReferenceAdapter(KVCompressionCandidate):
    """PolarQuant reference — experimental, expected slower, reference only.

    Ideas from:
      external/mlx-turboquant/mlx_turboquant/polar_quant.py
      external/mlx-turboquant/mlx_turboquant/turbo_quant.py

    Key insight: random orthogonal rotation → Beta-distributed coordinates →
    data-oblivious Lloyd-Max quantization with near-optimal distortion.
    """

    name = "polar_reference"

    def __init__(self, bits: int = 4, dim: int = 128, seed: int = 42) -> None:
        self.bits = bits
        self.dim = dim
        self.seed = seed
        self.name = f"polar_reference_b{bits}_d{dim}"

    def is_available(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
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
                error="mlx or mlx_lm not available",
            )
        try:
            import mlx_lm

            # Reference: run without compression to measure baseline quality.
            # Full PolarQuant KV integration is in rfsn_v11/quant/value_quant.py.
            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
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
                notes=(
                    f"PolarQuant reference: b{self.bits} d{self.dim}  "
                    "EXPERIMENTAL — expected slower — reference only.  "
                    "Ideas: external/mlx-turboquant/mlx_turboquant/polar_quant.py"
                ),
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error=str(exc),
            )
