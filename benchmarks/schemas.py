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
from typing import Any, Optional


@dataclass
class CandidateResult:
    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    candidate_name: str = ""
    model_id: str = ""
    prompt_id: str = ""
    context_length: int = 0
    output_tokens: int = 0
    preconditioner: str = ""       # e.g. "wht", "sparse_jl", "none"
    quantizer: str = ""            # e.g. "grouped_sym", "polar", "turboquant_mse"
    key_bits: Optional[float] = None
    value_bits: Optional[float] = None
    group_size: Optional[int] = None
    residual_length: Optional[int] = None
    snapkv_enabled: bool = False
    paged_cache_enabled: bool = False

    # ------------------------------------------------------------------
    # Quality metrics (vs dense baseline logits)
    # ------------------------------------------------------------------
    logit_cosine: Optional[float] = None
    top1_match_rate: Optional[float] = None
    top5_overlap: Optional[float] = None
    top10_overlap: Optional[float] = None
    perplexity_delta: Optional[float] = None          # candidate_ppl - baseline_ppl
    visible_output_drift_score: Optional[float] = None  # 0=identical, 1=completely different

    # ------------------------------------------------------------------
    # Attention metrics
    # ------------------------------------------------------------------
    attention_score_cosine: Optional[float] = None
    attention_score_mae: Optional[float] = None
    attention_top5_overlap: Optional[float] = None
    softmax_kl: Optional[float] = None

    # ------------------------------------------------------------------
    # Memory metrics (all in MB)
    # ------------------------------------------------------------------
    peak_memory_mb: Optional[float] = None            # peak device memory during generation
    kv_cache_memory_mb: Optional[float] = None        # dense FP16 KV size for this run
    compressed_kv_memory_mb: Optional[float] = None   # compressed representation size
    metadata_memory_mb: Optional[float] = None        # codebook indices, norms, scales, etc.
    effective_bits_per_kv_element: Optional[float] = None
    compression_factor: Optional[float] = None        # kv_cache_memory_mb / compressed_kv_memory_mb

    # ------------------------------------------------------------------
    # Runtime metrics
    # ------------------------------------------------------------------
    prefill_tps: Optional[float] = None               # tokens/sec during prefill
    decode_tps: Optional[float] = None                # tokens/sec during decode
    first_token_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    compression_time_ms: Optional[float] = None       # time to compress KV vectors
    decompression_time_ms: Optional[float] = None     # time to decompress for attention
    attention_time_ms: Optional[float] = None         # time for attention computation

    # ------------------------------------------------------------------
    # Candidate-specific optional metrics
    # ------------------------------------------------------------------
    # Residual cache
    residual_memory_mb: Optional[float] = None
    compressed_history_memory_mb: Optional[float] = None
    streaming_logit_cosine: Optional[float] = None
    multi_turn_drift_score: Optional[float] = None

    # SnapKV
    snapkv_vote_time_ms: Optional[float] = None
    snapkv_retention_ratio_actual: Optional[float] = None
    snapkv_selected_tokens: Optional[int] = None
    snapkv_hit_rate: Optional[float] = None           # fraction of selected positions that matched dense attention top-k
    snapkv_memory_saved_mb: Optional[float] = None

    # Prefix cache
    prefix_cache_hit_rate: Optional[float] = None
    prefix_cache_blocks_reused: Optional[int] = None
    prefix_cache_blocks_evicted: Optional[int] = None
    prefix_cache_memory_saved_mb: Optional[float] = None
    prefix_cache_allocator_overhead_ms: Optional[float] = None

    # Sparse JL specific
    sparse_selection_overhead_ms: Optional[float] = None

    # PolarQuant specific
    angle_codebook_kl: Optional[float] = None
    angle_quantization_p95: Optional[float] = None
    radius_relative_error_p95: Optional[float] = None

    # Reconstruction (set by test_a1_reconstruction)
    k_reconstruction_cosine: Optional[float] = None
    v_reconstruction_cosine: Optional[float] = None
    k_mse: Optional[float] = None
    v_mse: Optional[float] = None
    k_snr_db: Optional[float] = None
    v_snr_db: Optional[float] = None

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
    commit_hash: str = ""
    corpus_hash: str = ""
    token_sequence_hash: str = ""
    mlx_version: str = ""
    device: str = ""
    measured_memory: bool = False                   # True if memory was actually measured
    estimated_memory: bool = False                  # True if memory was estimated (not measured)

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
    def from_dict(cls, d: dict[str, Any]) -> "CandidateResult":
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
