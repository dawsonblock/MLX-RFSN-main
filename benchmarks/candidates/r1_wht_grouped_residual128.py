"""Candidate R1: WHT grouped compression + FP16 residual window R=128.

Uses rfsn_v11.cache.residual_cache.ResidualKVCache to keep recent 128 tokens
in FP16 while compressing older history with WHT + grouped symmetric.

Also supports sweep: R=32, 64, 128, 256.
Expected safe default: R=128.

Candidate name: R1_wht_grouped_k8v4_residual128
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_Grouped
from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    pass


class R1_WHT_Grouped_Residual(BenchmarkCandidate):
    """R1: WHT grouped + FP16 residual window."""

    candidate_name = "R1_wht_grouped_k8v4_residual128"

    def __init__(
        self,
        residual_length: int = 128,
        key_bits: int = 8,
        value_bits: int = 4,
        group_size: int = 64,
    ) -> None:
        self.residual_length = residual_length
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size

    def is_available(self) -> bool:
        return _MLX_AVAILABLE

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
        if not _MLX_AVAILABLE:
            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                error="mlx not available",
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
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step

        from rfsn_v11.cache.residual_cache import ResidualKVCache

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        head_dim = A1_WHT_Grouped._detect_head_dim(model)
        n_layers = len(model.layers) if hasattr(model, "layers") else 24

        caches = [
            ResidualKVCache(
                head_dim=head_dim,
                residual_length=self.residual_length,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                group_size=self.group_size,
            )
            for _ in range(n_layers)
        ]

        input_ids = tokenizer.encode(prompt, return_tensors="mlx")
        if not hasattr(input_ids, "shape"):
            input_ids = mx.array(tokenizer.encode(prompt))[None]
        context_length = input_ids.shape[-1]

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
            prefill_tps = context_length / max(first_token_time - t_start, 1e-9)

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        try:
            peak_memory_mb = mx.metal.get_peak_memory() / (1024 ** 2)
        except Exception:
            peak_memory_mb = 0.0

        kv_dense_mb = self.estimate_kv_memory_mb(model, context_length + n_gen)
        compressed_mb = sum(c.compressed_bytes() for c in caches) / (1024 ** 2)
        residual_mb = sum(c.residual_memory_mb for c in caches)
        comp_factor = kv_dense_mb / max(compressed_mb + residual_mb, 1e-9)

        total_comp_ms = sum(c.compressed_cache.compression_time_ms for c in caches)
        total_decomp_ms = sum(c.compressed_cache.decompression_time_ms for c in caches)

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
            group_size=self.group_size,
            residual_length=self.residual_length,
            peak_memory_mb=peak_memory_mb,
            kv_cache_memory_mb=kv_dense_mb,
            compressed_kv_memory_mb=compressed_mb,
            residual_memory_mb=residual_mb,
            compression_factor=comp_factor,
            effective_bits_per_kv_element=(self.key_bits + self.value_bits) / 2.0,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            first_token_latency_ms=first_token_latency_ms,
            total_latency_ms=total_latency_ms,
            compression_time_ms=total_comp_ms,
            decompression_time_ms=total_decomp_ms,
            generated_text=generated_text,
            notes=f"Residual R={self.residual_length}",
        )
