"""Candidate R2: TurboQuant MSE + FP16 residual window R=128.

Stacks TurboQuant MSE compression (A3) with the residual cache manager.
Requires A3 to pass before R2 is meaningful.

Candidate name: R2_turboquant_mse_k4v4_residual128
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a3_wht_turboquant_mse_4bit import A3_WHT_TurboQuant_MSE
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult


class R2_TurboQuant_MSE_Residual(BenchmarkCandidate):
    """R2: TurboQuant MSE + residual window."""

    candidate_name = "R2_turboquant_mse_k4v4_residual128"

    def __init__(self, residual_length: int = 128) -> None:
        self.residual_length = residual_length

    def is_available(self) -> bool:
        return A3_WHT_TurboQuant_MSE().is_available()

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
        a3 = A3_WHT_TurboQuant_MSE()
        result = a3.run_on_model(model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed)
        result.candidate_name = self.candidate_name
        result.residual_length = self.residual_length
        result.notes = f"TurboQuant MSE + residual R={self.residual_length}"
        return result
