"""Candidate: RFSN v10 stable baseline (k8_v5_gs32 and k8_v5_gs64).

This wraps the validated rfsn_v10 quantization path so the shootout can
compare it against newer candidates on equal footing.

Config name mapping
-------------------
k8_v5_gs32  →  default_bits=8, group_size=32   (recommended)
k8_v5_gs64  →  default_bits=8, group_size=64   (also validated)
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .logit_capture import capture_teacher_forced_logprobs
from .memory_metrics import estimate_kv_memory_mb
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE

# Map the human-readable preset names to actual QuantizationConfig kwargs.
# rfsn_v10.config.RFSNConfig has no from_preset() — we build it directly.
_PRESET_MAP: dict[str, dict[str, Any]] = {
    "k8_v5_gs32": {"default_bits": 8, "group_size": 32},
    "k8_v5_gs64": {"default_bits": 8, "group_size": 64},
}


class RFSNV10Candidate(KVCompressionCandidate):
    """RFSN v10 with a given quantization config."""

    candidate_status = CandidateStatus.BASELINE

    def __init__(self, config_name: str = "k8_v5_gs32") -> None:
        if config_name not in _PRESET_MAP:
            raise ValueError(
                f"Unknown rfsn_v10 preset {config_name!r}. "
                f"Valid: {list(_PRESET_MAP)}"
            )
        self.config_name = config_name
        self.name = f"rfsn_v10_{config_name}"

    def is_available(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
            import rfsn_v10  # noqa: F401
            return True
        except ImportError:
            return False

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_text: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> Any:
        """Capture teacher-forced log-probs via standard MLX-LM path.

        RFSN v10's current generation path (``enable_sparse_decode=False``)
        delegates to ``mlx_lm.stream_generate`` without SDPA patching, so the
        teacher-forced capture can use the standard MLX-LM forward pass.
        """
        return capture_teacher_forced_logprobs(
            model, tokenizer, prompt, target_text,
        )

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
                gate_status="ERROR",
                error="rfsn_v10 or mlx_lm not importable",
            )
        try:
            import contextlib
            import io

            from rfsn_v10.config import QuantizationConfig, RFSNConfig
            from rfsn_v10.runtime.generation import RFSNGenerator

            quant_kwargs = _PRESET_MAP[self.config_name]
            cfg = RFSNConfig(
                quantization=QuantizationConfig(**quant_kwargs),
            )
            generator = RFSNGenerator(
                model,
                tokenizer,
                cfg,
                enable_quantized_kv=True,
                enable_sparse_decode=False,
            )

            # Suppress mlx-lm deprecated-arg print()s from internals
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = list(generator.generate(
                    prompt, max_new_tokens=max_tokens, temperature=temp,
                ))
            total_ms = (time.perf_counter() - t0) * 1000
            result_text = "".join(tokens)

            gen_tokens = max(len(tokens), 1)
            tps = gen_tokens / (total_ms / 1000)

            actual_kv_memory_mb = estimate_kv_memory_mb(
                model, tokenizer, prompt, gen_tokens,
                bits=quant_kwargs["default_bits"],
            )
            size_ratio = quant_kwargs["default_bits"] / 16.0
            compression_factor = 16.0 / quant_kwargs["default_bits"]

            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=result_text,
                actual_kv_memory_mb=actual_kv_memory_mb,
                size_ratio=size_ratio,
                compression_factor=compression_factor,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                candidate_status=self.candidate_status,
                cache_backend_used="rfsn_v10_quantized_kv",
                cache_events=["prefill_quantize", "decode_quantized_fetch"],
                notes=(
                    f"RFSN v10 stable baseline — config={self.config_name} "
                    f"bits={quant_kwargs['default_bits']} "
                    f"gs={quant_kwargs['group_size']}  "
                    "Generation path currently delegates to mlx_lm.stream_generate; "
                    "real RFSN v10 cache injection is pending."
                ),
            )
        except Exception as exc:
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error=str(exc),
            )
