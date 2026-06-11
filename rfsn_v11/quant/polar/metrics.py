"""Offline validation metrics for PolarQuant K-cache compression.

Computes reconstruction quality and attention-score fidelity against
the original dense FP16 K block.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from .encoder import PolarQuantEncoder
from .decoder import PolarQuantDecoder


def _cosine_similarity(a: mx.array, b: mx.array) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    dot = float(mx.sum(a * b).item())
    norm_a = float(mx.sum(a * a).item()) ** 0.5
    norm_b = float(mx.sum(b * b).item()) ** 0.5
    return dot / (norm_a * norm_b + 1e-12)


def _attention_scores(q: mx.array, k: mx.array) -> mx.array:
    """Compute Q @ K.T for a single head.

    q: (D,)  k: (T, D)  -> scores: (T,)
    """
    return q @ k.T


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of shared top-k indices between two 1-D score arrays."""
    top_a = set(np.argpartition(a, -k)[-k:])
    top_b = set(np.argpartition(b, -k)[-k:])
    return len(top_a & top_b) / k


def evaluate_polar_offline(
    keys: mx.array,
    encoder: PolarQuantEncoder,
    decoder: PolarQuantDecoder,
    query: mx.array | None = None,
) -> dict[str, float | bool]:
    """Run full offline validation on a K block.

    Args:
        keys: K tensor of shape (..., D).
        encoder: PolarQuantEncoder instance.
        decoder: PolarQuantDecoder instance (must match encoder settings).
        query: Optional query vector of shape (D,). If None, a random query
               is generated for attention-score comparison.

    Returns:
        dict with reconstruction_cosine, reconstruction_mse,
        attention_score_cosine, attention_top5_overlap,
        attention_top10_overlap, size_ratio, compression_factor,
        nan_inf_detected.
    """
    block = encoder.encode(keys)
    recon = decoder.decode(block)

    # Reconstruction metrics
    reconstruction_cosine = _cosine_similarity(keys, recon)
    reconstruction_mse = float(mx.mean((keys - recon) ** 2).item())

    # NaN / Inf guard
    nan_inf_detected = bool(
        mx.any(mx.isnan(recon)).item() or mx.any(mx.isinf(recon)).item()
    )

    # Attention-score metrics
    if query is None:
        rng = np.random.RandomState(42)
        query = mx.array(rng.randn(keys.shape[-1]).astype(np.float32))
    else:
        query = query.astype(mx.float32)

    # Flatten K to (T, D) for score computation
    k_flat = keys.reshape(-1, keys.shape[-1])
    r_flat = recon.reshape(-1, recon.shape[-1])

    dense_scores = _attention_scores(query, k_flat)
    recon_scores = _attention_scores(query, r_flat)

    d_np = np.array(dense_scores.astype(mx.float32))
    r_np = np.array(recon_scores.astype(mx.float32))

    attention_score_cosine = float(
        np.sum(d_np * r_np) / (np.linalg.norm(d_np) * np.linalg.norm(r_np) + 1e-12)
    )
    attention_top5_overlap = _topk_overlap(d_np, r_np, 5)
    attention_top10_overlap = _topk_overlap(d_np, r_np, 10)

    # Compression
    orig_bytes = block.original_nbytes()
    comp_bytes = block.compressed_nbytes()
    size_ratio = comp_bytes / orig_bytes if orig_bytes > 0 else 0.0
    compression_factor = orig_bytes / comp_bytes if comp_bytes > 0 else 0.0

    return {
        "reconstruction_cosine": reconstruction_cosine,
        "reconstruction_mse": reconstruction_mse,
        "attention_score_cosine": attention_score_cosine,
        "attention_top5_overlap": attention_top5_overlap,
        "attention_top10_overlap": attention_top10_overlap,
        "size_ratio": size_ratio,
        "compression_factor": compression_factor,
        "nan_inf_detected": nan_inf_detected,
    }
