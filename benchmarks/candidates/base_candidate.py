"""Abstract base class for all RFSN benchmark candidates.

Every candidate must implement run_on_model() and is_available().
The base class provides common helpers for memory measurement,
logit quality computation, and result construction.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from benchmarks.schemas import CandidateResult


class BenchmarkCandidate(ABC):
    """Abstract base for benchmark candidates.

    Subclasses must set ``candidate_name`` and implement ``run_on_model()``.
    """

    candidate_name: str = "unnamed"

    @property
    def name(self) -> str:
        """Return the candidate name (alias for candidate_name).
        
        This provides compatibility with code that uses .name instead of .candidate_name.
        """
        return self.candidate_name

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temp: float = 0.0,
    ) -> CandidateResult:
        """Run the candidate with simplified interface for benchmark harness.
        
        This is a convenience wrapper around run_on_model() that provides
        the simpler interface expected by kv_shootout.py.
        
        Args:
            model: Loaded model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temp: Temperature (default 0.0 for greedy)
        
        Returns:
            CandidateResult with benchmark metrics
        """
        model_id = getattr(model, "name_or_path", "unknown")
        prompt_id = "benchmark_prompt"

        return self.run_on_model(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            prompt_id=prompt_id,
            prompt=prompt,
            output_tokens=max_tokens,
            seed=42,  # Fixed seed for reproducibility
        )

    @abstractmethod
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
        """Run the candidate on a loaded model and return a full CandidateResult."""
        ...

    def is_available(self) -> bool:
        """Return True if all dependencies for this candidate are importable."""
        return True

    @property
    def supports_teacher_forced_capture(self) -> bool:
        """Return True if this candidate can capture teacher-forced logprobs.
        
        Subclasses that implement capture_logprobs() should override this.
        """
        return False

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_text: str,
    ) -> list[float] | None:
        """Capture teacher-forced log probabilities for quality comparison.
        
        Args:
            model: Loaded model
            tokenizer: Tokenizer
            prompt: Input prompt
            target_text: Target text to compute logprobs for
            
        Returns:
            List of log probabilities per token, or None if not supported.
        """
        return None

    # ------------------------------------------------------------------
    # Common quality helpers (used by subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_logit_quality(
        baseline_logits: np.ndarray,
        candidate_logits: np.ndarray,
    ) -> dict[str, float | None]:
        """Compute logit-level quality metrics between baseline and candidate.

        Parameters
        ----------
        baseline_logits, candidate_logits
            Shape (T, vocab) float32.
        """
        if baseline_logits.shape != candidate_logits.shape:
            return {k: None for k in (
                "logit_cosine", "top1_match_rate", "top5_overlap",
                "top10_overlap", "perplexity_delta", "visible_output_drift_score",
            )}

        T = baseline_logits.shape[0]

        # Cosine similarity
        dots = np.sum(baseline_logits * candidate_logits, axis=-1)
        norms_b = np.linalg.norm(baseline_logits, axis=-1) + 1e-12
        norms_c = np.linalg.norm(candidate_logits, axis=-1) + 1e-12
        cosines = dots / (norms_b * norms_c)
        logit_cosine = float(np.mean(cosines))

        # Top-k overlaps
        b_top1 = np.argmax(baseline_logits, axis=-1)
        c_top1 = np.argmax(candidate_logits, axis=-1)
        top1_match_rate = float(np.mean(b_top1 == c_top1))

        b_top5 = np.argsort(baseline_logits, axis=-1)[:, -5:]
        c_top5 = np.argsort(candidate_logits, axis=-1)[:, -5:]
        top5_overlap = float(np.mean([
            len(set(b_top5[t]) & set(c_top5[t])) / 5.0 for t in range(T)
        ]))

        b_top10 = np.argsort(baseline_logits, axis=-1)[:, -10:]
        c_top10 = np.argsort(candidate_logits, axis=-1)[:, -10:]
        top10_overlap = float(np.mean([
            len(set(b_top10[t]) & set(c_top10[t])) / 10.0 for t in range(T)
        ]))

        # Perplexity delta
        def _safe_softmax(x: np.ndarray) -> np.ndarray:
            x = x - np.max(x, axis=-1, keepdims=True)
            e = np.exp(x)
            return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

        b_sm = _safe_softmax(baseline_logits)
        c_sm = _safe_softmax(candidate_logits)

        b_lp = np.log(b_sm + 1e-12)
        c_lp = np.log(c_sm + 1e-12)
        b_ppl = float(np.exp(-np.mean(b_lp[np.arange(T), b_top1])))
        c_ppl = float(np.exp(-np.mean(c_lp[np.arange(T), b_top1])))
        perplexity_delta = c_ppl - b_ppl

        # Visible output drift: fraction of top-1 tokens that differ
        visible_output_drift_score = float(np.mean(b_top1 != c_top1))

        return {
            "logit_cosine": logit_cosine,
            "top1_match_rate": top1_match_rate,
            "top5_overlap": top5_overlap,
            "top10_overlap": top10_overlap,
            "perplexity_delta": perplexity_delta,
            "visible_output_drift_score": visible_output_drift_score,
        }

    @staticmethod
    def compute_attention_quality(
        q: np.ndarray,
        k_dense: np.ndarray,
        k_compressed: np.ndarray,
        v_dense: np.ndarray,
        v_compressed: np.ndarray,
        scale: float | None = None,
    ) -> dict[str, float | None]:
        """Compute attention-score quality metrics.

        Computes Q @ K^T for both dense and compressed K, then compares.
        Shape: q, k_*: (T_q or T_kv, D), v_*: (T_kv, D)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(max(q.shape[-1], 1))

        # Dense attention scores: (T_q, T_kv)
        scores_dense = q @ k_dense.T * scale
        scores_comp = q @ k_compressed.T * scale

        # Cosine similarity
        sd_flat = scores_dense.flatten()
        sc_flat = scores_comp.flatten()
        d_norm = np.linalg.norm(sd_flat) + 1e-12
        c_norm = np.linalg.norm(sc_flat) + 1e-12
        attention_score_cosine = float(np.dot(sd_flat, sc_flat) / (d_norm * c_norm))

        # MAE
        attention_score_mae = float(np.mean(np.abs(scores_dense - scores_comp)))

        # Softmax KL
        def _softmax_rows(x: np.ndarray) -> np.ndarray:
            x = x - np.max(x, axis=-1, keepdims=True)
            e = np.exp(x)
            return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

        p_dense = _softmax_rows(scores_dense)
        p_comp = _softmax_rows(scores_comp)
        kl = float(np.mean(np.sum(p_dense * (np.log(p_dense + 1e-12) - np.log(p_comp + 1e-12)), axis=-1)))

        # Top-5 overlap on attention weights (per query)
        T_q = scores_dense.shape[0]
        top5_overlaps = []
        for t in range(T_q):
            b5 = set(np.argsort(scores_dense[t])[-5:])
            c5 = set(np.argsort(scores_comp[t])[-5:])
            top5_overlaps.append(len(b5 & c5) / 5.0)
        attention_top5_overlap = float(np.mean(top5_overlaps))

        return {
            "attention_score_cosine": attention_score_cosine,
            "attention_score_mae": attention_score_mae,
            "attention_top5_overlap": attention_top5_overlap,
            "softmax_kl": kl,
        }

    @staticmethod
    def compute_reconstruction_metrics(
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> dict[str, float]:
        """K/V reconstruction quality metrics.

        Parameters
        ----------
        original, reconstructed : (N, D) float32 arrays.
        """
        # Cosine similarity (per vector, averaged)
        dots = np.sum(original * reconstructed, axis=-1)
        norms_o = np.linalg.norm(original, axis=-1) + 1e-12
        norms_r = np.linalg.norm(reconstructed, axis=-1) + 1e-12
        cosine = float(np.mean(dots / (norms_o * norms_r)))

        # MSE
        mse = float(np.mean((original - reconstructed) ** 2))

        # SNR in dB
        signal_power = float(np.mean(original ** 2))
        noise_power = float(np.mean((original - reconstructed) ** 2))
        snr_db = 10 * math.log10(signal_power / max(noise_power, 1e-30))

        return {"cosine": cosine, "mse": mse, "snr_db": snr_db}

    @staticmethod
    def estimate_kv_memory_mb(model: Any, context_length: int) -> float:
        """Estimate FP16 KV cache memory in MB for a given context."""
        try:
            args = getattr(model, "args", None)
            if args is None:
                return 0.0
            num_layers = getattr(args, "num_hidden_layers", None) or getattr(args, "n_layers", 0)
            num_kv_heads = (
                getattr(args, "num_key_value_heads", None)
                or getattr(args, "num_attention_heads", None)
                or 0
            )
            hidden_size = getattr(args, "hidden_size", None) or getattr(args, "dim", 0)
            num_heads = getattr(args, "num_attention_heads", None) or getattr(args, "n_heads", 1)
            head_dim = hidden_size // max(num_heads, 1)
            kv_bytes = 2 * num_layers * num_kv_heads * head_dim * context_length * 2
            return kv_bytes / (1024 ** 2)
        except Exception:
            return 0.0
