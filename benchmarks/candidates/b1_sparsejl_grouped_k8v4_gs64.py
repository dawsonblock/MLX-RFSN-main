"""Candidate B1: Sparse JL preconditioner ablation.

Tests whether sparse JL preconditioning (random ±1 projection with sparsity)
outperforms WHT preconditioning for KV quantization.

Sparse JL construction (deterministic given seed):
  S is a D×D matrix where each column has exactly sqrt(D) non-zero entries.
  Each non-zero entry is ±1/sqrt(k) where k = sqrt(D).
  This satisfies the JL property: preserves pairwise distances with
  distortion ~1 + O(1/k).

For efficiency, we apply this as a permutation + sign flip rather than
explicit sparse matrix multiplication: generate a random permutation of
indices and a random ±1 sign vector, both deterministic from seed.

This candidate exists for evidence only. Expected outcome:
  KEEP_EXPERIMENTAL or REJECT (WHT is the safer practical default).

Candidate name: B1_sparsejl_grouped_k8v4_gs64
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    pass


class B1_SparseJL_Grouped(BenchmarkCandidate):
    """B1: Sparse JL preconditioning + grouped symmetric quantization."""

    candidate_name = "B1_sparsejl_grouped_k8v4_gs64"

    def __init__(self, key_bits: int = 8, value_bits: int = 4, group_size: int = 64, seed: int = 42) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.seed = seed

    def is_available(self) -> bool:
        return _MLX_AVAILABLE

    def _make_transform(self, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic sparse JL transform: (permutation, signs)."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(dim)
        signs = np.where(rng.random(dim) < 0.5, 1.0, -1.0).astype(np.float32)
        return perm, signs

    def _apply_sparse_jl(self, x: mx.array, seed: int) -> mx.array:
        """Apply deterministic sparse JL to last dimension."""
        dim = x.shape[-1]
        perm, signs = self._make_transform(dim, seed)
        perm_mx = mx.array(perm)
        signs_mx = mx.array(signs)
        return x[..., perm_mx] * signs_mx

    def _inverse_sparse_jl(self, x: mx.array, seed: int) -> mx.array:
        """Inverse: permute back and apply same signs (signs are self-inverse)."""
        dim = x.shape[-1]
        perm, signs = self._make_transform(dim, seed)
        # Inverse permutation
        inv_perm = np.argsort(perm)
        inv_perm_mx = mx.array(inv_perm)
        signs_mx = mx.array(signs)
        return (x * signs_mx)[..., inv_perm_mx]

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

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        # Detect architecture
        args = getattr(model, "args", None)
        head_dim = getattr(args, "head_dim", None) if args else None
        if head_dim is None:
            hidden = getattr(args, "hidden_size", 0) if args else 0
            n_heads = getattr(args, "num_attention_heads", 1) if args else 1
            head_dim = hidden // n_heads if hidden and n_heads else 128
        n_layers = len(model.layers) if hasattr(model, "layers") else 24

        # Build sparse-JL caches — a wrapper around mx.quantize that applies
        # the transform before quantizing and inverse after dequantizing
        class _SparseJLCache:
            step = 256
            def __init__(slf, dim, kb, vb, gs, sd):
                slf.head_dim = dim; slf.key_bits = kb; slf.value_bits = vb; slf.group_size = gs
                slf.seed = sd; slf.offset = 0
                slf.keys = slf.values = None
                slf.comp_ms = slf.decomp_ms = 0.0

            def update_and_fetch(slf, keys, values):
                import time as _time
                B, H, T, D = keys.shape
                # Sparse JL transform
                t0 = _time.perf_counter()
                k_jl = self._apply_sparse_jl(keys, slf.seed)
                v_jl = self._apply_sparse_jl(values, slf.seed + 1)
                # Quantize
                kq, ks, kb = mx.quantize(k_jl.reshape(-1, D), group_size=slf.group_size, bits=slf.key_bits)
                vq, vs, vb = mx.quantize(v_jl.reshape(-1, D), group_size=slf.group_size, bits=slf.value_bits)
                # Store in pre-allocated buffer
                prev = slf.offset
                if slf.keys is None or (prev + T) > slf.keys[0].shape[0]:
                    cap = ((slf.step + T - 1) // slf.step) * slf.step
                    new_k = (mx.zeros((cap, D * slf.key_bits // 32), mx.uint32),
                             mx.zeros((cap, D // slf.group_size), mx.float32),
                             mx.zeros((cap, D // slf.group_size), mx.float32))
                    new_v = (mx.zeros((cap, D * slf.value_bits // 32), mx.uint32),
                             mx.zeros((cap, D // slf.group_size), mx.float32),
                             mx.zeros((cap, D // slf.group_size), mx.float32))
                    if slf.keys is not None:
                        new_k = tuple(mx.concatenate([slf.keys[i][:prev], new_k[i]], axis=0) for i in range(3))
                        new_v = tuple(mx.concatenate([slf.values[i][:prev], new_v[i]], axis=0) for i in range(3))
                    slf.keys, slf.values = new_k, new_v
                for i in range(3):
                    slf.keys[i][prev:prev+T] = (kq, ks, kb)[i]
                    slf.values[i][prev:prev+T] = (vq, vs, vb)[i]
                slf.offset += T
                mx.eval(slf.keys, slf.values)
                slf.comp_ms += (_time.perf_counter() - t0) * 1000.0

                # Decompress full history
                t0 = _time.perf_counter()
                total = slf.offset
                k_dec_jl = mx.dequantize(slf.keys[0][:total], slf.keys[1][:total], slf.keys[2][:total],
                                         group_size=slf.group_size, bits=slf.key_bits).reshape(B, H, total, D)
                v_dec_jl = mx.dequantize(slf.values[0][:total], slf.values[1][:total], slf.values[2][:total],
                                         group_size=slf.group_size, bits=slf.value_bits).reshape(B, H, total, D)
                k_dec = self._inverse_sparse_jl(k_dec_jl, slf.seed)
                v_dec = self._inverse_sparse_jl(v_dec_jl, slf.seed + 1)
                mx.eval(k_dec, v_dec)
                slf.decomp_ms += (_time.perf_counter() - t0) * 1000.0
                return k_dec, v_dec

            def compressed_bytes(slf):
                if slf.keys is None: return 0
                T = slf.offset
                D = slf.head_dim
                k_bytes = T * (D * slf.key_bits // 8 + D // slf.group_size * 4 * 2)
                v_bytes = T * (D * slf.value_bits // 8 + D // slf.group_size * 4 * 2)
                return k_bytes + v_bytes

        caches = [_SparseJLCache(head_dim, self.key_bits, self.value_bits, self.group_size, self.seed + i)
                  for i in range(n_layers)]

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
        compressed_bytes = sum(c.compressed_bytes() for c in caches)
        compressed_kv_mb = compressed_bytes / (1024 ** 2)
        comp_factor = kv_dense_mb / max(compressed_kv_mb, 1e-9) if compressed_kv_mb > 0 else None
        total_comp_ms = sum(c.comp_ms for c in caches)
        total_decomp_ms = sum(c.decomp_ms for c in caches)

        return CandidateResult(
            candidate_name=self.candidate_name,
            model_id=model_id,
            prompt_id=prompt_id,
            context_length=context_length,
            output_tokens=n_gen,
            preconditioner="sparse_jl",
            quantizer="grouped_sym",
            key_bits=float(self.key_bits),
            value_bits=float(self.value_bits),
            group_size=self.group_size,
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
            notes="Sparse JL ablation — expected to underperform WHT",
        )
