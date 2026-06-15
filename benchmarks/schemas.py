"""Canonical CandidateResult schema for the RFSN benchmark harness.

This is the single source of truth for every metric produced by every
compression candidate.  All candidates, the judge, and the report generator
import from here.

Field conventions
-----------------
All fields are Optional[float] unless they are identity strings.
A field that is None means "not measured" — not zero, not unknown.
The judge treats any required field that is None as "missing" and
will block promotion.

Metric categories
-----------------
identity     : who/what ran
quality      : logit-level fidelity vs dense baseline
attention    : attention-score-level fidelity
memory       : bytes consumed at various granularities
runtime      : latency and throughput
compression  : compression metadata (bits, ratios)
candidate    : method-specific optional metrics (residual, snapkv, …)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CandidateResult:
    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    name: str = ""  # Alias for candidate_name for compatibility
    candidate_name: str = ""
    model_id: str = ""
    prompt_id: str = ""
    context_length: int = 0
    output_tokens: int = 0
    preconditioner: str = ""       # e.g. "wht", "sparse_jl", "none"
    quantizer: str = ""            # e.g. "grouped_sym", "polar", "turboquant_mse"
    key_bits: float | None = None
    value_bits: float | None = None
    group_size: int | None = None
    residual_length: int | None = None
    snapkv_enabled: bool = False
    paged_cache_enabled: bool = False
    is_benchmark_only: bool = False  # True for benchmark-only candidates (e.g., A1 with MLX limitations)

    def __post_init__(self) -> None:
        """Sync name and candidate_name for backward compatibility."""
        if not self.name and self.candidate_name:
            self.name = self.candidate_name
        elif not self.candidate_name and self.name:
            self.candidate_name = self.name

    # ------------------------------------------------------------------
    # Quality metrics (vs dense baseline logits)
    # ------------------------------------------------------------------
    logit_cosine: float | None = None
    kl_divergence: float | None = None
    top1_match: float | None = None
    top1_match_rate: float | None = None
    top5_overlap: float | None = None
    top10_overlap: float | None = None
    perplexity_delta: float | None = None          # candidate_ppl - baseline_ppl
    visible_output_drift_score: float | None = None  # 0=identical, 1=completely different
    max_logit_delta: float | None = None
    first_divergent_token: int | None = None
    text_heuristic_passed: bool | None = None
    logit_gate_passed: bool | None = None
    memory_gate_passed: bool | None = None

    # ------------------------------------------------------------------
    # Attention metrics
    # ------------------------------------------------------------------
    attention_score_cosine: float | None = None
    attention_score_mae: float | None = None
    attention_top5_overlap: float | None = None
    softmax_kl: float | None = None

    # ------------------------------------------------------------------
    # Memory metrics (all in MB)
    # ------------------------------------------------------------------
    peak_memory_mb: float | None = None            # peak device memory during generation
    kv_cache_memory_mb: float | None = None        # dense FP16 KV size for this run
    compressed_kv_memory_mb: float | None = None   # compressed representation size
    metadata_memory_mb: float | None = None        # codebook indices, norms, scales, etc.
    effective_bits_per_kv_element: float | None = None
    compression_factor: float | None = None        # kv_cache_memory_mb / compressed_kv_memory_mb

    # ------------------------------------------------------------------
    # Runtime metrics
    # ------------------------------------------------------------------
    prefill_tps: float | None = None               # tokens/sec during prefill
    decode_tps: float | None = None                # tokens/sec during decode
    tokens_per_sec: float | None = None            # alias for decode_tps
    first_token_latency_ms: float | None = None
    total_latency_ms: float | None = None
    total_ms: float | None = None                  # alias for total_latency_ms
    compression_time_ms: float | None = None       # time to compress KV vectors
    decompression_time_ms: float | None = None     # time to decompress for attention
    attention_time_ms: float | None = None         # time for attention computation

    # ------------------------------------------------------------------
    # Candidate-specific optional metrics
    # ------------------------------------------------------------------
    # Residual cache
    residual_memory_mb: float | None = None
    compressed_history_memory_mb: float | None = None
    streaming_logit_cosine: float | None = None
    multi_turn_drift_score: float | None = None

    # SnapKV
    snapkv_vote_time_ms: float | None = None
    snapkv_retention_ratio_actual: float | None = None
    snapkv_selected_tokens: int | None = None
    snapkv_hit_rate: float | None = None           # fraction of selected positions that matched dense attention top-k
    snapkv_memory_saved_mb: float | None = None

    # Prefix cache
    prefix_cache_hit_rate: float | None = None
    prefix_cache_blocks_reused: int | None = None
    prefix_cache_blocks_evicted: int | None = None
    prefix_cache_memory_saved_mb: float | None = None
    prefix_cache_allocator_overhead_ms: float | None = None

    # Sparse JL specific
    sparse_selection_overhead_ms: float | None = None

    # PolarQuant specific
    angle_codebook_kl: float | None = None
    angle_quantization_p95: float | None = None
    radius_relative_error_p95: float | None = None

    # Reconstruction (set by test_a1_reconstruction)
    k_reconstruction_cosine: float | None = None
    v_reconstruction_cosine: float | None = None
    k_mse: float | None = None
    v_mse: float | None = None
    k_snr_db: float | None = None
    v_snr_db: float | None = None

    # ------------------------------------------------------------------
    # Output text (for drift inspection)
    # ------------------------------------------------------------------
    generated_text: str = ""
    baseline_text: str = ""

    # ------------------------------------------------------------------
    # Provenance (Phase 1 governance — prevents synthetic/fallback promotion)
    # ------------------------------------------------------------------
    run_type: str = "unknown"                       # "synthetic" | "real_model" | "smoke"
    source_type: str = "unknown"                    # "checkout" | "installed_wheel"
    requested_backend: str = "unknown"            # e.g. "metal", "reference"
    executed_backend: str = "unknown"               # e.g. "metal", "reference", "fallback"
    metal_executed: bool = False
    fallback_used: bool = False
    promotion_eligible: bool = True                 # False for smoke/benchmark-only data
    candidate_status: str = "UNKNOWN"              # "ACTIVE" | "REFERENCE_ONLY" | "CONTROL" | "DEPRECATED"
    gate_status: str = "UNKNOWN"                   # "PASS" | "FAIL" | "ERROR" | "PENDING_*"
    commit_hash: str = ""
    corpus_hash: str = ""
    token_sequence_hash: str = ""
    mlx_version: str = ""
    device: str = ""
    measured_memory: bool = False                   # True if memory was actually measured
    estimated_memory: bool = False                  # True if memory was estimated (not measured)

    # ------------------------------------------------------------------
    # Proof counters (strict mode validation)
    # ------------------------------------------------------------------
    proof_counters: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Runtime counters (Fix #4, #5, #6: actual block creation, packed calls, bytes)
    # ------------------------------------------------------------------
    packed_blocks_created: int = 0
    packed_blocks_read: int = 0
    packed_attention_calls: int = 0
    dense_fallback_calls: int = 0
    full_history_materialization_calls: int = 0
    packed_bytes_written: int = 0
    packed_bytes_read: int = 0
    actual_kv_memory_mb: float | None = None
    working_set_memory_mb: float | None = None
    scratch_memory_mb: float | None = None
    measurement_kind: str = "UNKNOWN"  # "ACTUAL" | "ESTIMATED" | "UNKNOWN"

    # ------------------------------------------------------------------
    # Errors / notes
    # ------------------------------------------------------------------
    error: str = ""
    notes: str = ""

    # ------------------------------------------------------------------
    # Raw logits (not serialised to JSON by default — large)
    # ------------------------------------------------------------------
    _logits: Any = field(default=None, repr=False, compare=False)
    _baseline_logits: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self, include_logits: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_logits:
            d.pop("_logits", None)
            d.pop("_baseline_logits", None)
        return d

    def to_json(self, indent: int = 2, include_logits: bool = False) -> str:
        return json.dumps(self.to_dict(include_logits=include_logits), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateResult:
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Quick summaries
    # ------------------------------------------------------------------

    def compression_summary(self) -> str:
        if self.compressed_kv_memory_mb is None or self.kv_cache_memory_mb is None:
            return "compression: unknown"
        ratio = self.compressed_kv_memory_mb / max(self.kv_cache_memory_mb, 1e-9)
        factor = self.compression_factor or (1.0 / max(ratio, 1e-9))
        return (
            f"Compressed size: {ratio * 100:.1f}% of FP16  "
            f"({factor:.2f}x smaller)"
        )

    def quality_summary(self) -> str:
        parts = []
        if self.logit_cosine is not None:
            parts.append(f"logit_cos={self.logit_cosine:.5f}")
        if self.top5_overlap is not None:
            parts.append(f"top5={self.top5_overlap:.3f}")
        if self.attention_score_cosine is not None:
            parts.append(f"attn_cos={self.attention_score_cosine:.5f}")
        if self.perplexity_delta is not None:
            parts.append(f"ppl_delta={self.perplexity_delta:+.4f}")
        return "  ".join(parts) if parts else "(no quality metrics)"
