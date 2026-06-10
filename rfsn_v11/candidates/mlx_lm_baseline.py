"""Candidate: MLX-LM baseline (no KV compression).

This is the control. Every compression candidate must beat or match it in the
right dimension (memory ↓, speed ↑) without unacceptable quality loss.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .quality_gates import (
    GATE_STATUS_PASS,
    compute_promotion_eligibility,
)


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
        temp: float = 0.0,
    ) -> CandidateResult:
        try:
            import mlx.core as mx
            import mlx_lm

            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp)
            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            # Tokenize to count generated tokens
            input_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(output)
            gen_tokens = max(len(output_ids) - len(input_ids), 1)
            tps = gen_tokens / (total_ms / 1000)

            # Baseline has perfect quality by definition
            promotion_eligible, gate_status = compute_promotion_eligibility(
                logit_gate_passed=True,
                memory_gate_passed=True,
                actual_kv_memory_mb=None,  # baseline does not compress
                working_set_memory_mb=None,
                size_ratio=1.0,
                compression_factor=1.0,
            )

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=output,
                logit_cosine=1.0,
                kl_divergence=0.0,
                top1_match=1.0,
                top5_overlap=1.0,
                top10_overlap=1.0,
                max_logit_delta=0.0,
                first_divergent_token=None,
                logit_gate_passed=True,
                memory_gate_passed=True,
                promotion_eligible=promotion_eligible,
                gate_status=gate_status,
                notes="FP16 baseline — no compression applied",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error=str(exc),
            )
