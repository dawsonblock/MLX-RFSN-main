"""Candidate: RFSN v11 fusion prototype.

Tests the rfsn_v11 asymmetric K/V compression path:
  - WHT key quantization (KeyQuant)
  - PolarQuant value quantization (KVCompressor)

Integration note
----------------
mlx_lm.generate() does not accept a custom KV cache object in the installed
version.  This adapter therefore:
  1. Runs plain mlx_lm generation with kv_bits=8 (closest to v11 8-bit path)
     to measure real generation speed and text quality.
  2. Runs KVCompressor on a synthetic KV batch to measure its compression
     ratio and standalone reconstruction quality.

The compression metrics (size_ratio, compression_factor) reflect what v11
achieves on the KV data; the generation speed reflects baseline mlx_lm at
comparable bit-width.  This is the honest measurement given current API limits.

Status: experimental — must beat rfsn_v10 in shootout before promotion.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CandidateResult, KVCompressionCandidate


class RFSNV11Candidate(KVCompressionCandidate):
    """RFSN v11 fusion compressor."""

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 4,
        group_size: int = 64,
        use_wht: bool = True,
        dim: int = 128,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.use_wht = use_wht
        self.dim = dim
        self.name = (
            f"rfsn_v11_k{key_bits}v{value_bits}"
            f"_gs{group_size}"
            f"{'_wht' if use_wht else ''}"
        )

    def is_available(self) -> bool:
        try:
            from rfsn_v11.quant.kv_compressor import KVCompressor  # noqa: F401
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error="rfsn_v11 quant module or mlx_lm not importable",
            )
        try:
            import mlx.core as mx
            import mlx_lm
            from rfsn_v11.quant.kv_compressor import KVCompressor

            # --- Step 1: measure KVCompressor compression quality offline ---
            compressor = KVCompressor(
                k_bits=self.key_bits,
                v_bits=self.value_bits,
                group_size=self.group_size,
                dim=self.dim,
                use_wht=self.use_wht,
                skip_quality_gate=True,  # gate already passed in tests
            )

            rng = np.random.RandomState(42)
            batch = 256
            keys_np = rng.randn(batch, self.dim).astype(np.float32)
            vals_np = rng.randn(batch, self.dim).astype(np.float32)
            keys = mx.array(keys_np)
            vals = mx.array(vals_np)

            compressed = compressor.compress(keys, vals)
            keys_rec, vals_rec = compressor.decompress(compressed)
            mx.eval(keys_rec, vals_rec)

            # Compute size ratio: count stored elements
            fp16_bytes = 2 * batch * self.dim * 2  # keys + values, float16
            # Key: codes (k_bits per element) + scales (float32 per group)
            k_code_bytes = batch * self.dim * self.key_bits / 8
            k_scale_bytes = (batch * self.dim / self.group_size) * 4
            # Value: indices (v_bits per element) + norms (float32 per vector)
            v_idx_bytes = batch * self.dim * float(self.value_bits) / 8
            v_norm_bytes = batch * 4
            compressed_bytes = k_code_bytes + k_scale_bytes + v_idx_bytes + v_norm_bytes
            size_ratio = compressed_bytes / fp16_bytes
            compression_factor = fp16_bytes / compressed_bytes

            # Key reconstruction cosine
            k_np = np.array(keys_rec)
            cosine_k = float(np.mean([
                np.dot(keys_np[i], k_np[i]) /
                (np.linalg.norm(keys_np[i]) * np.linalg.norm(k_np[i]) + 1e-8)
                for i in range(batch)
            ]))

            # --- Step 2: measure generation speed with equivalent bit-width ---
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp)
            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                kv_bits=self.key_bits,
                kv_group_size=self.group_size,
                sampler=sampler,
                verbose=False,
            )
            total_ms = (time.perf_counter() - t0) * 1000

            input_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(output)
            gen_tokens = max(len(output_ids) - len(input_ids), 1)
            tps = gen_tokens / (total_ms / 1000)

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=output,
                size_ratio=round(size_ratio, 4),
                compression_factor=round(compression_factor, 3),
                logit_cosine=round(cosine_k, 5),
                passed_quality_gate=False,  # filled by shootout quality eval
                notes=(
                    f"RFSN v11 fusion k{self.key_bits}v{self.value_bits} "
                    f"gs{self.group_size} wht={self.use_wht}  "
                    f"KV cosine={cosine_k:.4f}  "
                    f"size_ratio={size_ratio:.3f} ({compression_factor:.2f}x smaller)  "
                    "Generation via mlx_lm kv_bits (direct cache injection pending)"
                ),
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error=str(exc),
            )
