"""Candidate: MLX-LM baseline (no KV compression).

This is the control. Every compression candidate must beat or match it in the
right dimension (memory ↓, speed ↑) without unacceptable quality loss.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CandidateResult, KVCompressionCandidate
from .quality_gates import evaluate_quality_gate, logit_quality_metrics


class MLXLMBaseline(KVCompressionCandidate):
    """Plain MLX-LM generation with no KV compression."""

    name = "mlx_lm_baseline"

    def is_available(self) -> bool:
        try:
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
        try:
            import mlx.core as mx
            import mlx_lm

            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            # Tokenize to count generated tokens
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
                # Baseline has perfect quality by definition
                logit_cosine=1.0,
                kl_divergence=0.0,
                top1_match=1.0,
                top5_overlap=1.0,
                top10_overlap=1.0,
                max_logit_delta=0.0,
                first_divergent_token=None,
                passed_quality_gate=True,
                notes="FP16 baseline — no compression applied",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error=str(exc),
            )
