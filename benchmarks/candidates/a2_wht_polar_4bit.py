"""Candidate A2: PolarQuant 4-bit.

Wraps external/mlx-turboquant's PolarQuant-based TurboQuantKVCache.
PolarQuant reduces/eliminates per-block normalization metadata by transforming
vectors into polar form after random rotation.

This is treated as an experiment against A1 and A3, not the default.
Only promotes if metadata memory falls AND quality remains high AND
decode overhead is acceptable.

Candidate name: A2_wht_polar_4bit
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

_EXT_POLAR = str(Path(__file__).parent.parent.parent / "external" / "mlx-turboquant")


def _ensure_ext_on_path() -> None:
    if _EXT_POLAR not in sys.path:
        sys.path.insert(0, _EXT_POLAR)


class A2_WHT_Polar(BenchmarkCandidate):
    """A2: PolarQuant reference — rotation + Lloyd-Max codebook."""

    candidate_name = "A2_wht_polar_4bit"

    def __init__(self, bits: int = 4, seed: int = 42) -> None:
        self.bits = bits
        self.seed = seed

    def is_available(self) -> bool:
        try:
            _ensure_ext_on_path()
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            from mlx_turboquant.cache import TurboQuantKVCache  # noqa: F401
            return True
        except ImportError:
            return False

    def run_on_model(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prompt_id: str,
        prompt: str,
        output_tokens: int = 100,
        seed: int = 42,
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                error="mlx-turboquant not importable",
            )
        try:
            return self._run(model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed)
        except Exception as exc:
            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                error=str(exc),
            )

    def _run(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prompt_id: str,
        prompt: str,
        output_tokens: int,
        seed: int,
    ) -> CandidateResult:
        # Delegate to the existing PolarReferenceAdapter and translate metrics
        from rfsn_v11.candidates.polar_reference_adapter import PolarReferenceAdapter
        adapter = PolarReferenceAdapter(bits=self.bits, seed=self.seed)
        old_result = adapter.run(model, tokenizer, prompt, max_tokens=output_tokens, temp=0.0)

        # Translate old CandidateResult to new schema
        return CandidateResult(
            candidate_name=self.candidate_name,
            model_id=model_id,
            prompt_id=prompt_id,
            preconditioner="polar_rotation",
            quantizer="lloyd_max_codebook",
            value_bits=float(self.bits),
            key_bits=float(self.bits),
            group_size=64,
            peak_memory_mb=None,
            kv_cache_memory_mb=None,
            compressed_kv_memory_mb=None,
            compression_factor=None,
            effective_bits_per_kv_element=float(self.bits),
            total_latency_ms=old_result.total_ms,
            tokens_per_sec=old_result.tokens_per_sec,
            generated_text=old_result.generated_text,
            generated_tokens=old_result.generated_tokens,
            logit_cosine=old_result.logit_cosine,
            top5_overlap=old_result.top5_overlap,
            top10_overlap=old_result.top10_overlap,
            perplexity_delta=old_result.kl_divergence,
            notes=f"PolarQuant {self.bits}-bit via mlx-turboquant",
        )
