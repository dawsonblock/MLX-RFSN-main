"""Memory metric helpers for KV-cache compression candidates."""
from __future__ import annotations


def bytes_to_mb(nbytes: int | float) -> float:
    """Convert bytes to megabytes."""
    return float(nbytes) / (1024 * 1024)


def compression_factor(baseline_bytes: int | float, compressed_bytes: int | float) -> float:
    """Baseline size divided by compressed size (higher is better)."""
    if compressed_bytes <= 0:
        raise ValueError("compressed_bytes must be positive")
    return float(baseline_bytes) / float(compressed_bytes)


def size_ratio(baseline_bytes: int | float, compressed_bytes: int | float) -> float:
    """Compressed size divided by baseline size (lower is better)."""
    if baseline_bytes <= 0:
        raise ValueError("baseline_bytes must be positive")
    return float(compressed_bytes) / float(baseline_bytes)
