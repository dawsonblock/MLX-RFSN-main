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
         compression_factor=3.77 means "3.77x smaller than FP16"

Do NOT report these as "0.265x compression" — that is misleading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidate_status import CandidateStatus


@dataclass
class CandidateResult:
    # Identity
    name: str
    model_id: str
    prompt_id: str = ""
    prompt: str = ""

    # Memory
    actual_kv_memory_mb: float | None = None
    working_set_memory_mb: float | None = None
    measurement_kind: str = "ESTIMATED"  # P0 #8: Distinguish ESTIMATED vs RUNTIME_COUNTERS

    # Compression
    size_ratio: float | None = None  # compressed / baseline
    compression_factor: float | None = None  # baseline / compressed

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

    # Gate outcomes
    text_heuristic_passed: bool | None = None
    logit_gate_passed: bool | None = None
    memory_gate_passed: bool | None = None
    promotion_eligible: bool = False
    gate_status: str = "PENDING_LOGIT_GATE"
    failed_gate_reasons: list[str] = field(default_factory=list)

    # Candidate lifecycle status (CONTROL, BASELINE, EXPERIMENTAL, etc.)
    candidate_status: CandidateStatus = field(
        default=CandidateStatus.EXPERIMENTAL
    )

    # Real cache proof (required for promotion eligibility)
    cache_backend_used: str = ""          # e.g. "turboquant_v2", "mlx_lm_fp16"
    cache_events: list[str] = field(default_factory=list)
    cache_bytes_written: int | None = None
    cache_bytes_read: int | None = None

    # Detailed runtime counters (for instrumentation and validation)
    packed_blocks_created: int = 0
    packed_blocks_read: int = 0
    packed_attention_calls: int = 0
    dense_fallback_calls: int = 0
    full_history_materialization_calls: int = 0
    packed_bytes_read: int = 0
    packed_bytes_written: int = 0
    decoded_block_bytes: int = 0
    scratch_bytes_peak: int = 0
    # Fix P1: removed block_seal_events; use packed_blocks_created instead
    execution_backend: str = ""  # e.g. "metal", "cpu", "reference"

    # Patch safety proof (for candidates that patch SDPA)
    patch_scope: str | None = None        # e.g. "controlled_context", "global"
    global_patch_restored: bool | None = None

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
        gate = self.gate_status
        return f"[{gate}] " + "  ".join(parts) if parts else f"[{gate}]"


class KVCompressionCandidate:
    """Base class for all KV-compression candidates.

    Subclasses must set ``name`` and implement ``run()``.
    """

    name: str = "unnamed"
    candidate_status: CandidateStatus = CandidateStatus.EXPERIMENTAL

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
        """Return False if the candidate cannot run in the
        current environment."""
        return True
