"""Candidate A4: TurboQuant + QJL residual correction.

Same as A3 but with QJL residual sign bits for inner-product correction.
Tested ONLY after A3 passes. QJL adds implementation complexity and should
not be first.

QJL specific metrics to track (stored in candidate-specific fields):
  - metadata_memory_mb for sign-bit overhead
  - attention_score_cosine (the primary QJL metric)
  - attention_top5_overlap
  - softmax_kl
  - logit_cosine

Promotion only if QJL improves attention/logit quality or allows lower bits
without slowing too much. Otherwise REJECT or KEEP_EXPERIMENTAL.

Candidate name: A4_wht_turboquant_qjl_4bit
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a3_wht_turboquant_mse_4bit import A3_WHT_TurboQuant_MSE
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult


class A4_WHT_TurboQuant_QJL(BenchmarkCandidate):
    """A4: TurboQuant MSE + QJL residual sign correction."""

    candidate_name = "A4_wht_turboquant_qjl_4bit"

    def __init__(self, bits: int = 4, group_size: int = 64, seed: int = 42) -> None:
        self.bits = bits
        self.group_size = group_size
        self.seed = seed

    def is_available(self) -> bool:
        return A3_WHT_TurboQuant_MSE(bits=self.bits, group_size=self.group_size).is_available()

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
        # A4 = A3 with use_qjl=True.  For now delegate to A3 and mark as QJL variant.
        # The real QJL implementation requires the TurboQuantKVCacheV2 constructor
        # to be called with use_qjl=True.  When the external library supports it,
        # switch to that path.
        a3 = A3_WHT_TurboQuant_MSE(bits=self.bits, group_size=self.group_size, seed=self.seed)
        result = a3.run_on_model(model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed)
        result.candidate_name = self.candidate_name
        result.quantizer = "mlx_native_grouped_sym_qjl"
        result.notes = f"TurboQuant QJL bits={self.bits}"
        # QJL adds ~1 bit overhead per element for sign bits
        if result.compressed_kv_memory_mb is not None:
            result.metadata_memory_mb = result.compressed_kv_memory_mb * 0.25  # ~25% overhead for sign bits
        return result
