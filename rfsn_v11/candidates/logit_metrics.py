"""Logit metric helpers for comparing baseline and candidate next-token distributions."""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D or 2-D arrays.

    For 2-D arrays of shape (T, vocab), returns the mean cosine over tokens.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 1:
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a)) + 1e-12
        norm_b = float(np.linalg.norm(b)) + 1e-12
        return dot / (norm_a * norm_b)
    # 2-D: per-token cosine, then mean
    dots = np.sum(a * b, axis=-1)
    norms_a = np.linalg.norm(a, axis=-1) + 1e-12
    norms_b = np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(dots / (norms_a * norms_b)))


def kl_divergence_from_logits(baseline_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    """KL(P_baseline || P_candidate) averaged over tokens.

    Parameters
    ----------
    baseline_logits, candidate_logits
        Shape (T, vocab) float32 arrays.
    """
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError(f"Shape mismatch: {baseline_logits.shape} vs {candidate_logits.shape}")
    b_sm = _safe_softmax(baseline_logits)
    c_sm = _safe_softmax(candidate_logits)
    kl_per_token = np.sum(
        b_sm * (np.log(b_sm + 1e-12) - np.log(c_sm + 1e-12)), axis=-1
    )
    return float(np.mean(kl_per_token))


def topk_overlap(baseline_logits: np.ndarray, candidate_logits: np.ndarray, k: int) -> float:
    """Mean overlap fraction between top-k indices of baseline and candidate.

    Parameters
    ----------
    baseline_logits, candidate_logits
        Shape (T, vocab) float32 arrays.
    k
        Number of top items to compare.
    """
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError(f"Shape mismatch: {baseline_logits.shape} vs {candidate_logits.shape}")
    T = baseline_logits.shape[0]
    b_top = np.argsort(baseline_logits, axis=-1)[:, -k:]
    c_top = np.argsort(candidate_logits, axis=-1)[:, -k:]
    overlaps = [
        len(set(b_top[t]) & set(c_top[t])) / k for t in range(T)
    ]
    return float(np.mean(overlaps))


def max_logit_delta(baseline_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    """Maximum absolute difference between corresponding logits."""
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError(f"Shape mismatch: {baseline_logits.shape} vs {candidate_logits.shape}")
    return float(np.max(np.abs(baseline_logits - candidate_logits)))


def first_divergent_token(baseline_tokens: list[int], candidate_tokens: list[int]) -> int | None:
    """Index of first token where sequences differ, or None if identical."""
    for i, (b, c) in enumerate(zip(baseline_tokens, candidate_tokens)):
        if b != c:
            return i
    if len(baseline_tokens) != len(candidate_tokens):
        return min(len(baseline_tokens), len(candidate_tokens))
    return None


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)
