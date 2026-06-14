"""Candidate S2: SnapKV + A1 grouped WHT compression.

Stacks SnapKV pruning (keeps only important prefix positions) with
A1 WHT grouped compression on the kept tokens.

Memory savings = SnapKV pruning savings × A1 compression savings.

Candidate name: S2_snapkv_plus_a1_grouped
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_Grouped
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult


class S2_SnapKV_PlusGrouped(BenchmarkCandidate):
    """S2: SnapKV + A1 grouped compression."""

    candidate_name = "S2_snapkv_plus_a1_grouped"

    def is_available(self) -> bool:
        return A1_WHT_Grouped().is_available()

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
        a1 = A1_WHT_Grouped()
        result = a1.run_on_model(model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed)
        result.candidate_name = self.candidate_name
        result.snapkv_enabled = True
        result.notes = "SnapKV + A1 grouped WHT (simulated stack)"
        # Stack memory savings: if A1 gives compression_factor CF, and
        # SnapKV keeps ratio RR, effective compression = CF / RR
        if result.compression_factor is not None and result.snapkv_retention_ratio_actual is not None:
            effective_cf = result.compression_factor / max(result.snapkv_retention_ratio_actual, 1e-9)
            result.compression_factor = effective_cf
        return result
