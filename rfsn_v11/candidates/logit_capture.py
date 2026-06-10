"""Capture per-step logits/log-probabilities during MLX-LM generation.

This module provides a lightweight logit-capture path using mlx_lm's
``generate_step`` generator.  It is intentionally separate from the main
generation path so that adding logit capture does not break candidates that
do not support it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .logit_metrics import cosine_similarity
from .quality_gates import (
    KL_DIVERGENCE_MAX,
    LOGIT_COSINE_MIN,
    MAX_LOGIT_DELTA_MAX,
    TOP5_OVERLAP_MIN,
    TOP10_OVERLAP_MIN,
)


def capture_generation_logprobs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 200,
    temp: float = 0.0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
) -> dict[str, Any]:
    """Generate text while capturing per-step log-probability vectors.

    Uses ``mlx_lm.utils.generate_step`` which yields ``(token, log_probs)``
    at each decode step.  The log-probability vectors are collected and
    returned as a numpy array of shape ``(T, vocab)``.

    Parameters
    ----------
    model, tokenizer
        MLX-LM model and tokenizer.
    prompt
        Input prompt string.
    max_tokens
        Maximum tokens to generate.
    temp
        Sampling temperature (0 = greedy).
    kv_bits
        Optional KV-quantization bits for the *baseline* run (used to
        compare baseline-quantized vs candidate).
    kv_group_size
        Group size when ``kv_bits`` is set.

    Returns
    -------
    dict
        ``{"tokens": list[int], "logprobs": np.ndarray, "text": str}``
        or ``{"error": str}`` on failure.
    """
    try:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step

        sampler = make_sampler(temp=temp)
        prompt_ids = tokenizer.encode(prompt)
        prompt_mx = mx.array(prompt_ids)

        tokens = list(prompt_ids)
        logprob_list: list[np.ndarray] = []

        gen_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        if kv_bits is not None:
            gen_kwargs["kv_bits"] = kv_bits
            gen_kwargs["kv_group_size"] = kv_group_size

        for token, log_probs in generate_step(prompt_mx, model, **gen_kwargs):
            tokens.append(int(token.item()))
            # log_probs is shape (vocab,) — convert to numpy
            lp_np = np.array(log_probs)
            logprob_list.append(lp_np)
            if len(tokens) - len(prompt_ids) >= max_tokens:
                break

        if not logprob_list:
            return {"error": "No log-probabilities captured"}

        # Stack to (T, vocab)
        logprobs_arr = np.stack(logprob_list, axis=0)
        generated_text = tokenizer.decode(tokens)

        return {
            "tokens": tokens,
            "logprobs": logprobs_arr,
            "text": generated_text,
        }
    except Exception as exc:
        return {"error": str(exc)}


def compute_logit_metrics_from_logprobs(
    baseline_logprobs: np.ndarray,
    candidate_logprobs: np.ndarray,
) -> dict[str, float | None]:
    """Compute quality metrics between baseline and candidate log-probs.

    Uses the same thresholds as ``quality_gates.evaluate_quality_gate``
    but expects log-probability arrays (output of ``generate_step``)
    instead of raw logits.
    """
    if baseline_logprobs.shape != candidate_logprobs.shape:
        return {
            "logit_cosine": None,
            "kl_divergence": None,
            "top1_match": None,
            "top5_overlap": None,
            "top10_overlap": None,
            "max_logit_delta": None,
            "first_divergent_token": None,
        }

    T, vocab = baseline_logprobs.shape

    # Cosine on log-probs (distributions are already normalised-ish)
    logit_cosine = cosine_similarity(baseline_logprobs, candidate_logprobs)

    # KL divergence — convert log-probs to probabilities first
    b_p = np.exp(
        baseline_logprobs
        - np.max(baseline_logprobs, axis=-1, keepdims=True)
    )
    b_p = b_p / (np.sum(b_p, axis=-1, keepdims=True) + 1e-12)
    c_p = np.exp(
        candidate_logprobs
        - np.max(candidate_logprobs, axis=-1, keepdims=True)
    )
    c_p = c_p / (np.sum(c_p, axis=-1, keepdims=True) + 1e-12)
    kl_per_token = np.sum(
        b_p * (np.log(b_p + 1e-12) - np.log(c_p + 1e-12)), axis=-1
    )
    kl_divergence = float(np.mean(kl_per_token))

    # Top-k overlaps
    b_top1 = np.argmax(baseline_logprobs, axis=-1)
    c_top1 = np.argmax(candidate_logprobs, axis=-1)
    top1_match = float(np.mean(b_top1 == c_top1))

    b_top5 = np.argsort(baseline_logprobs, axis=-1)[:, -5:]
    c_top5 = np.argsort(candidate_logprobs, axis=-1)[:, -5:]
    top5_overlap = float(np.mean([
        len(set(b_top5[t]) & set(c_top5[t])) / 5.0 for t in range(T)
    ]))

    b_top10 = np.argsort(baseline_logprobs, axis=-1)[:, -10:]
    c_top10 = np.argsort(candidate_logprobs, axis=-1)[:, -10:]
    top10_overlap = float(np.mean([
        len(set(b_top10[t]) & set(c_top10[t])) / 10.0 for t in range(T)
    ]))

    # Max logit delta (operates on log-probs here — scale is different
    # from raw logits but still useful for detecting large divergences)
    max_logit_delta = float(
        np.max(np.abs(baseline_logprobs - candidate_logprobs))
    )

    # First divergent token (top-1 differs)
    divergent = np.where(b_top1 != c_top1)[0]
    first_divergent_token = int(divergent[0]) if len(divergent) > 0 else None

    return {
        "logit_cosine": logit_cosine,
        "kl_divergence": kl_divergence,
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "max_logit_delta": max_logit_delta,
        "first_divergent_token": first_divergent_token,
    }


def logit_gate_passed(metrics: dict[str, float | None]) -> bool:
    """Return True if all logit gate thresholds are met."""
    return (
        metrics.get("logit_cosine") is not None
        and metrics["logit_cosine"] >= LOGIT_COSINE_MIN
        and metrics.get("kl_divergence") is not None
        and metrics["kl_divergence"] <= KL_DIVERGENCE_MAX
        and metrics.get("top5_overlap") is not None
        and metrics["top5_overlap"] >= TOP5_OVERLAP_MIN
        and metrics.get("top10_overlap") is not None
        and metrics["top10_overlap"] >= TOP10_OVERLAP_MIN
        and metrics.get("max_logit_delta") is not None
        and metrics["max_logit_delta"] <= MAX_LOGIT_DELTA_MAX
    )
