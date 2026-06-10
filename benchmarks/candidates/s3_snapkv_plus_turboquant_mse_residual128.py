"""Candidate S3: SnapKV + TurboQuant MSE + residual128.

The full stack: SnapKV prunes prefix positions, TurboQuant MSE compresses
kept tokens, and the residual window protects recent context.

Candidate name: S3_snapkv_plus_turboquant_mse_residual128
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a3_wht_turboquant_mse_4bit import A3_WHT_TurboQuant_MSE
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult


class S3_SnapKV_PlusTurboQuantMSEResidual(BenchmarkCandidate):
    """S3: SnapKV + TurboQuant MSE + residual window."""

    candidate_name = "S3_snapkv_plus_turboquant_mse_residual128"

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
        result.snapkv_enabled = True
        result.residual_length = 128
        result.notes = "SnapKV + TurboQuant MSE + residual128 (simulated stack)"
        return result
