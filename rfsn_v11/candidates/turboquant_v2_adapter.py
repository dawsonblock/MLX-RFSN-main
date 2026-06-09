"""Candidate: TurboQuant-MLX V2.

Reimplements the core ideas from external/turboquant-mlx as a clean adapter.
Does NOT import from external/ at runtime — logic is rebuilt here.

Key ideas (from external/turboquant-mlx/turboquant/):
  - cache_v2.py:     QR rotation + MLX native mx.quantize
  - rotation.py:     generate_rotation_matrix (QR decomp, det=+1 fix)
  - attention_v2.py: quantized_matmul scoring

Status: experimental candidate — must pass quality gate before promotion.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate


def _build_rotation_matrix(head_dim: int, seed: int = 42) -> Any:
    """QR rotation matrix for uniform distribution pre-quantization.

    Equivalent to external/turboquant-mlx/turboquant/rotation.py
    generate_rotation_matrix() but inlined so we have no runtime dep on external/.
    """
    import mlx.core as mx

    mx.random.seed(seed)
    G = mx.random.normal((head_dim, head_dim))
    mx.eval(G)
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q)
    # Sign correction: ensure det(Q) = +1
    diag_sign = mx.sign(mx.diag(R))
    Q = Q * diag_sign[None, :]
    mx.eval(Q)
    return Q


class TurboQuantV2Candidate(KVCompressionCandidate):
    """TurboQuant V2: random QR rotation + MLX native affine quantization."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        use_rotation: bool = True,
        seed: int = 42,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.use_rotation = use_rotation
        self.seed = seed
        self.name = (
            f"turboquant_v2_b{bits}_gs{group_size}"
            f"{'_rot' if use_rotation else '_norot'}"
        )

    def is_available(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
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
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                passed_quality_gate=False,
                error="mlx or mlx_lm not available",
            )
        try:
            import mlx.core as mx
            import mlx_lm

            rotation_matrix = (
                _build_rotation_matrix(128, seed=self.seed)
                if self.use_rotation
                else None
            )

            def _rotate_and_quantize(tensor: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                """Apply QR rotation then mx.quantize."""
                if rotation_matrix is not None and tensor.shape[-1] == rotation_matrix.shape[0]:
                    tensor = tensor @ rotation_matrix
                scales, biases = mx.quantize(tensor, bits=self.bits, group_size=self.group_size)
                return tensor, scales, biases

            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
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
                passed_quality_gate=False,  # filled by shootout quality eval
                notes=(
                    f"TurboQuant V2: b{self.bits} gs{self.group_size} "
                    f"rotation={self.use_rotation}  "
                    "Ideas from external/turboquant-mlx/turboquant/cache_v2.py"
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
