"""Candidate: RFSN v11 fusion prototype.

Tests the rfsn_v11 asymmetric K/V compression path:
  - WHT key quantization
  - PolarQuant value quantization
  - Optional paged / prefix cache

Status: experimental — must beat rfsn_v10 in shootout before promotion.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate


class RFSNV11Candidate(KVCompressionCandidate):
    """RFSN v11 fusion compressor."""

    name = "rfsn_v11_fusion"

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 8,
        use_wht: bool = True,
        use_polar: bool = True,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.use_wht = use_wht
        self.use_polar = use_polar
        self.name = (
            f"rfsn_v11_k{key_bits}v{value_bits}"
            f"{'_wht' if use_wht else ''}"
            f"{'_polar' if use_polar else ''}"
        )

    def is_available(self) -> bool:
        try:
            from rfsn_v11.quant.kv_compressor import KVCompressor  # noqa: F401
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
                error="rfsn_v11 quant module not importable",
            )
        try:
            import mlx.core as mx
            import mlx_lm
            from rfsn_v11.quant.kv_compressor import KVCompressor
            from rfsn_v11.quant.key_quant import KeyQuant
            from rfsn_v11.quant.value_quant import make_value_quantizer

            key_quant = KeyQuant(
                bits=self.key_bits,
                use_wht=self.use_wht,
            )
            value_quant = make_value_quantizer(
                bits=self.value_bits,
                use_polar=self.use_polar,
            )
            compressor = KVCompressor(
                key_quantizer=key_quant,
                value_quantizer=value_quant,
            )

            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                kv_cache=compressor,
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
                    f"RFSN v11 fusion: k{self.key_bits}v{self.value_bits} "
                    f"wht={self.use_wht} polar={self.use_polar}"
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
