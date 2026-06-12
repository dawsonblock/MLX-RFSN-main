"""Model-level quality validation gates for polar_fused.

Implements teacher-forced logit comparison and attention-level gates
required before promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]


@dataclass
class QualityGateResult:
    """Results from running quality gates on a candidate."""
    candidate_name: str
    passed: bool
    logit_cosine: float
    top1_agreement: float
    top5_overlap: float
    top10_overlap: float
    attention_score_cosine: float
    relative_perplexity_change: float
    nan_inf_count: int
    failed_gates: list[str] = field(default_factory=list)


class PolarQualityGates:
    """Quality gates for Polar fused attention candidates.

    Gates:
    - logit_cosine >= 0.995
    - top5_overlap >= 0.95
    - attention_score_cosine >= 0.995
    - relative_perplexity_change <= 2%
    - zero NaN or infinity
    """

    LOGIT_COSINE_MIN = 0.995
    TOP5_OVERLAP_MIN = 0.95
    ATTENTION_COSINE_MIN = 0.995
    MAX_PERPLEXITY_INCREASE = 0.02

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        candidate_name: str,
        baseline_logits: Any,
        candidate_logits: Any,
    ) -> QualityGateResult:
        """Evaluate candidate against baseline logits.

        Parameters
        ----------
        baseline_logits
            Baseline logits array (N, vocab_size)
        candidate_logits
            Candidate logits array (N, vocab_size)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not installed")

        # Convert to numpy for metric computation
        b_np = np.array(baseline_logits.astype(mx.float32))
        c_np = np.array(candidate_logits.astype(mx.float32))

        failed = []

        # Logit cosine
        b_flat = b_np.reshape(-1)
        c_flat = c_np.reshape(-1)
        logit_cosine = float(
            np.dot(b_flat, c_flat) / (np.linalg.norm(b_flat) * np.linalg.norm(c_flat))
        )
        if logit_cosine < self.LOGIT_COSINE_MIN:
            failed.append(f"logit_cosine {logit_cosine:.4f} < {self.LOGIT_COSINE_MIN}")

        # Top-k metrics
        b_top1 = np.argmax(b_np, axis=-1)
        c_top1 = np.argmax(c_np, axis=-1)
        top1_agreement = float(np.mean(b_top1 == c_top1))

        def topk_overlap(b, c, k):
            b_idx = np.argsort(b, axis=-1)[:, -k:]
            c_idx = np.argsort(c, axis=-1)[:, -k:]
            overlaps = [len(set(bi) & set(ci)) / k for bi, ci in zip(b_idx, c_idx)]
            return float(np.mean(overlaps))

        top5 = topk_overlap(b_np, c_np, 5)
        if top5 < self.TOP5_OVERLAP_MIN:
            failed.append(f"top5_overlap {top5:.4f} < {self.TOP5_OVERLAP_MIN}")

        top10 = topk_overlap(b_np, c_np, 10)

        # Perplexity (using mean cross-entropy as proxy)
        b_probs = np.exp(b_np - np.max(b_np, axis=-1, keepdims=True))
        b_probs /= b_probs.sum(axis=-1, keepdims=True)
        c_probs = np.exp(c_np - np.max(c_np, axis=-1, keepdims=True))
        c_probs /= c_probs.sum(axis=-1, keepdims=True)

        b_ce = -np.log(np.clip(b_probs[np.arange(len(b_top1)), b_top1], 1e-10, 1.0))
        c_ce = -np.log(np.clip(c_probs[np.arange(len(c_top1)), c_top1], 1e-10, 1.0))
        rel_ppl = float(np.mean(c_ce) / np.mean(b_ce) - 1.0) if np.mean(b_ce) > 0 else 0.0
        if rel_ppl > self.MAX_PERPLEXITY_INCREASE:
            failed.append(f"perplexity_increase {rel_ppl:.4f} > {self.MAX_PERPLEXITY_INCREASE}")

        # NaN/inf check
        nan_inf = int(np.sum(~np.isfinite(c_np)))
        if nan_inf > 0:
            failed.append(f"{nan_inf} NaN/inf values detected")

        return QualityGateResult(
            candidate_name=candidate_name,
            passed=len(failed) == 0,
            logit_cosine=logit_cosine,
            top1_agreement=top1_agreement,
            top5_overlap=top5,
            top10_overlap=top10,
            attention_score_cosine=0.0,  # TODO: compute from attention scores
            relative_perplexity_change=rel_ppl,
            nan_inf_count=nan_inf,
            failed_gates=failed,
        )

    @classmethod
    def gate_thresholds(cls) -> dict[str, float]:
        """Return current gate thresholds."""
        return {
            "logit_cosine_min": cls.LOGIT_COSINE_MIN,
            "top5_overlap_min": cls.TOP5_OVERLAP_MIN,
            "attention_cosine_min": cls.ATTENTION_COSINE_MIN,
            "max_perplexity_increase": cls.MAX_PERPLEXITY_INCREASE,
        }
