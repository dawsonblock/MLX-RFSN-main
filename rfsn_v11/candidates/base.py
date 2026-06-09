"""Shared candidate interface for the KV-compression shootout.

Every compression method must implement KVCompressionCandidate and return a
CandidateResult so results are comparable across methods.

Metric definitions
------------------
size_ratio
    compressed_size / baseline_size   (lower is better)
compression_factor
    baseline_size / compressed_size   (higher is better)

Example: size_ratio=0.265 means "compressed size is 26.5% of FP16"
         compression_factor=3.77 means "3.77× smaller than FP16"

Do NOT report these as "0.265× compression" — that is misleading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateResult:
    # Identity
    name: str
    model_id: str
    prompt: str

    # Memory
    actual_kv_memory_mb: float | None = None
    working_set_memory_mb: float | None = None

    # Compression
    size_ratio: float | None = None          # compressed / baseline (lower is better)
    compression_factor: float | None = None  # baseline / compressed (higher is better)

    # Timing (milliseconds)
    prefill_ms: float | None = None
    decode_ms: float | None = None
    total_ms: float | None = None

    # Throughput
    tokens_per_sec: float | None = None

    # Quality vs. FP16 baseline
    logit_cosine: float | None = None
    kl_divergence: float | None = None
    top1_match: float | None = None
    top5_overlap: float | None = None
    top10_overlap: float | None = None
    max_logit_delta: float | None = None
    first_divergent_token: int | None = None

    # Gate outcome
    passed_quality_gate: bool = False

    # Free-form notes
    notes: str = ""
    error: str = ""

    # Raw generated text for drift inspection
    generated_text: str = ""
    generated_tokens: int = 0

    def compression_summary(self) -> str:
        """Human-readable compression description."""
        if self.size_ratio is None or self.compression_factor is None:
            return "compression: unknown"
        return (
            f"Compressed size: {self.size_ratio * 100:.1f}% of FP16  "
            f"({self.compression_factor:.2f}x smaller)"
        )

    def quality_summary(self) -> str:
        parts = []
        if self.logit_cosine is not None:
            parts.append(f"cosine={self.logit_cosine:.5f}")
        if self.kl_divergence is not None:
            parts.append(f"KL={self.kl_divergence:.2e}")
        if self.top5_overlap is not None:
            parts.append(f"top5={self.top5_overlap:.3f}")
        gate = "PASS" if self.passed_quality_gate else "FAIL"
        return f"[{gate}] " + "  ".join(parts) if parts else f"[{gate}]"


class KVCompressionCandidate:
    """Base class for all KV-compression candidates.

    Subclasses must set ``name`` and implement ``run()``.
    """

    name: str = "unnamed"

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run()"
        )

    def is_available(self) -> bool:
        """Return False if the candidate cannot run in the current environment."""
        return True
