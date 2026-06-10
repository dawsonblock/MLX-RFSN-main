"""Memory metric helpers for KV-cache compression candidates."""
from __future__ import annotations

from typing import Any


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


def estimate_kv_memory_mb(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generated_tokens: int,
    bits: int = 16,
) -> float | None:
    """Estimate KV-cache memory usage from model configuration.

    Parameters
    ----------
    model
        MLX-LM model object.
    tokenizer
        MLX-LM tokenizer.
    prompt
        Input prompt string.
    generated_tokens
        Number of generated (decode) tokens.
    bits
        Bits per element. 16 for FP16, 8 for 8-bit quantized, etc.

    Returns
    -------
    float | None
        Estimated KV-cache memory in MB, or None if model config cannot be
        read.
    """
    try:
        args = getattr(model, "args", None)
        if args is None:
            return None

        n_layers = getattr(args, "num_hidden_layers", None)
        if n_layers is None:
            n_layers = len(getattr(model, "layers", []))

        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            hidden = getattr(args, "hidden_size", None)
            n_heads = getattr(args, "num_attention_heads", None)
            if hidden and n_heads:
                head_dim = hidden // n_heads
            else:
                return None

        prompt_tokens = len(tokenizer.encode(prompt))
        total_tokens = prompt_tokens + generated_tokens

        # 2 for keys + values
        bytes_per_element = bits / 8.0
        total_bytes = (
            2 * n_layers * n_heads * head_dim * total_tokens * bytes_per_element
        )
        return bytes_to_mb(total_bytes)
    except Exception:
        return None
