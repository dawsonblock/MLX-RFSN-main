"""Dense MLX baseline candidate for the benchmark harness.

Wraps benchmarks.baseline_mlx.run_single() in the BenchmarkCandidate interface.
This is the control — every compression candidate must compare against it.
"""
from __future__ import annotations

from typing import Any

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult


class DenseMlxBaseline(BenchmarkCandidate):
    """Plain FP16 MLX-LM generation.  Perfect quality by definition."""

    candidate_name = "dense_mlx_baseline"

    def is_available(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
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
        from benchmarks.baseline_mlx import run_single
        return run_single(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            prompt_id=prompt_id,
            prompt=prompt,
            output_tokens=output_tokens,
            seed=seed,
        )
