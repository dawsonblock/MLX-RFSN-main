"""Candidate: RFSN v10 stable baseline (k8_v5_gs32 and k8_v5_gs64).

This wraps the validated rfsn_v10 quantization path so the shootout can
compare it against newer candidates on equal footing.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate


class RFSNV10Candidate(KVCompressionCandidate):
    """RFSN v10 with a given quantization config."""

    def __init__(self, config_name: str = "k8_v5_gs32") -> None:
        self.config_name = config_name
        self.name = f"rfsn_v10_{config_name}"

    def is_available(self) -> bool:
        try:
            import rfsn_v10  # noqa: F401
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
                error="rfsn_v10 package not importable",
            )
        try:
            import mlx.core as mx
            from rfsn_v10.config import RFSNConfig
            from rfsn_v10.runtime.generation import GenerationConfig, RFSNGenerator

            cfg = RFSNConfig.from_preset(self.config_name)
            gen_cfg = GenerationConfig(max_new_tokens=max_tokens)
            generator = RFSNGenerator(model, tokenizer, cfg)

            t0 = time.perf_counter()
            result_text = generator.generate(prompt, gen_cfg)
            total_ms = (time.perf_counter() - t0) * 1000

            input_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(result_text)
            gen_tokens = max(len(output_ids) - len(input_ids), 1)
            tps = gen_tokens / (total_ms / 1000)

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=result_text,
                passed_quality_gate=False,  # filled by shootout quality eval
                notes=f"RFSN v10 stable baseline — config={self.config_name}",
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error=str(exc),
            )
