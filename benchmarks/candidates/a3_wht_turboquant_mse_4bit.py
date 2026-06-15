"""Candidate A3: TurboQuant MSE-only.

Uses external/turboquant-mlx's TurboQuantKVCacheV2 with random QR rotation
followed by MLX-native affine quantization (mx.quantize / mx.dequantize).

TurboQuant's research direction: rotation preconditioning makes vectors
easier to quantize; the key implementation lesson is to test the MSE-only
path before adding QJL residual correction.

Candidate name: A3_wht_turboquant_mse_4bit
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

_EXT_TQ = str(Path(__file__).parent.parent.parent / "external" / "turboquant-mlx")


def _ensure_ext_on_path() -> None:
    if _EXT_TQ not in sys.path:
        sys.path.insert(0, _EXT_TQ)


class A3_WHT_TurboQuant_MSE(BenchmarkCandidate):
    """A3: TurboQuant MSE-only — random QR rotation + MLX native quantize."""

    candidate_name = "A3_wht_turboquant_mse_4bit"

    def __init__(self, bits: int = 4, group_size: int = 64, seed: int = 42) -> None:
        self.bits = bits
        self.group_size = group_size
        self.seed = seed

    def is_available(self) -> bool:
        try:
            _ensure_ext_on_path()
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            from turboquant.cache_v2 import TurboQuantKVCacheV2  # noqa: F401
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
                error="turboquant-mlx not importable",
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
        import turboquant.patch
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step
        from turboquant.cache_v2 import TurboQuantKVCacheV2

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        head_dim = self._detect_head_dim(model)
        use_rotation = head_dim >= 128
        n_layers = len(model.layers) if hasattr(model, "layers") else 24

        caches = [
            TurboQuantKVCacheV2(
                head_dim=head_dim,
                bits=self.bits,
                group_size=self.group_size,
                use_rotation=use_rotation,
                use_normalization=False,
                seed=self.seed + i,
            )
            for i in range(n_layers)
        ]

        # Apply turboquant SDPA patch
        turboquant.patch.apply()

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="mlx")
            if not hasattr(input_ids, "shape"):
                input_ids = mx.array(tokenizer.encode(prompt))[None]
            context_length = input_ids.shape[-1]

            t_start = time.perf_counter()
            first_token_time = None
            generated_tokens = []
            prefill_tps = 0.0
            decode_tps = 0.0

            for token, _ in generate_step(
                input_ids[0], model, max_tokens=output_tokens, sampler=sampler, prompt_cache=caches,
            ):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                tok = int(token.item())
                generated_tokens.append(tok)
                if tok == tokenizer.eos_token_id:
                    break

            t_end = time.perf_counter()
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
            compressed_bytes = sum(c.nbytes() for c in caches)
            compressed_kv_mb = compressed_bytes / (1024 ** 2)
            comp_factor = kv_dense_mb / max(compressed_kv_mb, 1e-9) if compressed_kv_mb > 0 else None

            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                context_length=context_length,
                output_tokens=n_gen,
                preconditioner="turboquant_rot" if use_rotation else "none",
                quantizer="mlx_native_grouped_sym",
                key_bits=float(self.bits),
                value_bits=float(self.bits),
                group_size=self.group_size,
                peak_memory_mb=peak_memory_mb,
                kv_cache_memory_mb=kv_dense_mb,
                compressed_kv_memory_mb=compressed_kv_mb,
                compression_factor=comp_factor,
                effective_bits_per_kv_element=float(self.bits),
                prefill_tps=prefill_tps,
                decode_tps=decode_tps,
                first_token_latency_ms=first_token_latency_ms,
                total_latency_ms=total_latency_ms,
                generated_text=generated_text,
                notes=f"TurboQuant MSE bits={self.bits} rot={use_rotation}",
            )
        finally:
            # Revert turboquant SDPA patch
            try:
                turboquant.patch.revert()
            except Exception:
                pass

    @staticmethod
    def _detect_head_dim(model: Any) -> int:
        args = getattr(model, "args", None)
        if args:
            hd = getattr(args, "head_dim", None)
            if hd:
                return int(hd)
            hidden = getattr(args, "hidden_size", None) or getattr(args, "dim", 0)
            n_heads = getattr(args, "num_attention_heads", None) or getattr(args, "n_heads", 1)
            if hidden and n_heads:
                return hidden // n_heads
        for attr in ("self_attn", "attention", "attn"):
            layer = model.layers[0] if model.layers else None
            if layer and hasattr(layer, attr):
                a = getattr(layer, attr)
                if hasattr(a, "head_dim"):
                    return int(a.head_dim)
        raise ValueError("Cannot auto-detect head_dim")
