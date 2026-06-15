"""Candidate A1b: K/V asymmetric bit sweep.

Tests WHT + grouped symmetric quantization at different K/V bit combinations
without changing the preconditioner or the quantizer type.

Bit combinations tested:
    k8/v4  — default (same as A1)
    k8/v3  — reduce values first (keys stay protected)
    k6/v4  — reduce keys slightly
    k4/v4  — balanced low-bit

Promotion preference:
    Lowest memory while preserving:
        attention_top5_overlap >= 0.95
        logit_cosine >= 0.995

Expected safe winner: k8/v4 (already proven by A1).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_Grouped, A1_WHT_GroupedKVCache
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bit-combo registry
# ---------------------------------------------------------------------------

BIT_COMBOS: list[tuple[int, int, str]] = [
    (8, 4, "k8v4"),
    (8, 3, "k8v3"),
    (6, 4, "k6v4"),
    (4, 4, "k4v4"),
]


class A1b_WHT_Asym(BenchmarkCandidate):
    """A1b: asymmetric K/V bit sweep."""

    candidate_name = "A1b_wht_asym"

    def __init__(self, key_bits: int = 8, value_bits: int = 4) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.combo_name = f"k{key_bits}v{value_bits}"
        self.candidate_name = f"A1b_{self.combo_name}"

    def is_available(self) -> bool:
        if not _MLX_AVAILABLE:
            return False
        try:
            import mlx_lm  # noqa: F401

            from rfsn_v11.quant.key_quant import KeyQuant  # noqa: F401
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
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        head_dim = A1_WHT_Grouped._detect_head_dim(model)
        n_layers = A1_WHT_Grouped._detect_n_layers(model)

        caches = [
            A1_WHT_GroupedKVCache(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                group_size=64,
            )
            for _ in range(n_layers)
        ]

        input_ids = tokenizer.encode(prompt, return_tensors="mlx")
        context_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(tokenizer.encode(prompt))
        if not hasattr(input_ids, "shape"):
            input_ids = mx.array(tokenizer.encode(prompt))[None]

        import time as _time
        t_start = _time.perf_counter()
        first_token_time = None
        generated_tokens = []
        prefill_tps = 0.0
        decode_tps = 0.0

        for token, _ in generate_step(
            input_ids[0], model, max_tokens=output_tokens, sampler=sampler, prompt_cache=caches,
        ):
            now = _time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            tok = int(token.item())
            generated_tokens.append(tok)
            if tok == tokenizer.eos_token_id:
                break

        t_end = _time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        n_gen = len(generated_tokens)
        first_token_latency_ms = (
            (first_token_time - t_start) * 1000.0 if first_token_time is not None else None
        )
        if n_gen > 0 and first_token_time is not None:
            decode_tps = 1.0 / max((t_end - first_token_time) / max(n_gen - 1, 1), 1e-9)
            prefill_tps = context_length / max((first_token_time - t_start), 1e-9)

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        try:
            peak_memory_mb = mx.metal.get_peak_memory() / (1024 ** 2)
        except Exception:
            peak_memory_mb = 0.0

        kv_dense_mb = self.estimate_kv_memory_mb(model, context_length + n_gen)
        compressed_bytes = sum(c.compressed_bytes() for c in caches)
        compressed_kv_mb = compressed_bytes / (1024 ** 2)
        comp_factor = kv_dense_mb / max(compressed_kv_mb, 1e-9) if compressed_kv_mb > 0 else None

        total_comp_ms = sum(c.compression_time_ms for c in caches)
        total_decomp_ms = sum(c.decompression_time_ms for c in caches)

        return CandidateResult(
            candidate_name=self.candidate_name,
            model_id=model_id,
            prompt_id=prompt_id,
            context_length=context_length,
            output_tokens=n_gen,
            preconditioner="wht",
            quantizer="grouped_sym",
            key_bits=float(self.key_bits),
            value_bits=float(self.value_bits),
            group_size=64,
            logit_cosine=None,
            top5_overlap=None,
            attention_score_cosine=None,
            perplexity_delta=None,
            visible_output_drift_score=None,
            peak_memory_mb=peak_memory_mb,
            kv_cache_memory_mb=kv_dense_mb,
            compressed_kv_memory_mb=compressed_kv_mb,
            compression_factor=comp_factor,
            effective_bits_per_kv_element=(self.key_bits + self.value_bits) / 2.0,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            first_token_latency_ms=first_token_latency_ms,
            total_latency_ms=total_latency_ms,
            compression_time_ms=total_comp_ms,
            decompression_time_ms=total_decomp_ms,
            generated_text=generated_text,
            notes=f"A1b bit combo {self.combo_name}",
        )

    def run_sweep(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prompt_id: str,
        prompt: str,
        output_tokens: int = 100,
        seed: int = 42,
    ) -> list[CandidateResult]:
        """Run all bit combos and return a list of CandidateResult."""
        results = []
        for k_bits, v_bits, name in BIT_COMBOS:
            combo = A1b_WHT_Asym(key_bits=k_bits, value_bits=v_bits)
            result = combo.run_on_model(
                model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed,
            )
            results.append(result)
        return results
