"""Memory metric helpers for KV-cache compression candidates."""
from __future__ import annotations

from typing import Any


def bytes_to_mb(nbytes: int | float) -> float:
    """Convert bytes to megabytes."""
    return float(nbytes) / (1024 * 1024)


def compression_factor(
    baseline_bytes: int | float, compressed_bytes: int | float,
) -> float:
    """Baseline size divided by compressed size (higher is better)."""
    if compressed_bytes <= 0:
        raise ValueError("compressed_bytes must be positive")
    return float(baseline_bytes) / float(compressed_bytes)


def size_ratio(
    baseline_bytes: int | float, compressed_bytes: int | float,
) -> float:
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

    Note
    ----
    This is an analytical estimate and does not account for:
    - K8 versus V5 asymmetry
    - Scale tensors
    - Padding
    - Block metadata
    - Staging buffers
    - Dense residuals
    - Temporary reconstructions
    - Scratch space
    - MLX allocator behavior
    """
    try:
        args = getattr(model, "args", None)
        if args is None:
            return None

        n_layers = getattr(args, "num_hidden_layers", None)
        if n_layers is None:
            n_layers = len(getattr(model, "layers", []))

        n_heads = getattr(args, "num_attention_heads", None)
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            hidden = getattr(args, "hidden_size", None)
            if hidden and n_heads:
                head_dim = hidden // n_heads
            else:
                return None

        n_kv_heads = getattr(args, "num_key_value_heads", None)
        if n_kv_heads is None:
            n_kv_heads = n_heads

        prompt_tokens = len(tokenizer.encode(prompt))
        total_tokens = prompt_tokens + generated_tokens

        # 2 for keys + values
        bytes_per_element = bits / 8.0
        total_bytes = (
            2 * n_layers * n_kv_heads * head_dim
            * total_tokens * bytes_per_element
        )
        return bytes_to_mb(total_bytes)
    except Exception:
        return None


def get_actual_kv_memory_mb(runtime_counters: dict[str, Any]) -> dict[str, float]:
    """Get actual KV memory measurements from runtime counters.

    Parameters
    ----------
    runtime_counters
        Dictionary containing runtime instrumentation data.

    Returns
    -------
    dict
        Dictionary with actual memory measurements:
        - estimated_payload_memory_mb: Analytical estimate
        - measured_payload_memory_mb: Actual measured payload
        - measured_staging_memory_mb: Staging buffer memory
        - measured_dense_residual_memory_mb: Dense residual memory
        - measured_scratch_peak_mb: Peak scratch memory
        - measurement_kind: Type of measurement (RUNTIME_COUNTERS or ESTIMATED)
    """
    # Get analytical estimate for comparison
    estimated = runtime_counters.get("estimated_payload_memory_mb", 0.0)

    # Get actual measurements from runtime counters
    measured_payload = runtime_counters.get("packed_bytes_written", 0)
    measured_staging = runtime_counters.get("staging_bytes", 0)
    measured_residual = runtime_counters.get("dense_residual_bytes", 0)
    measured_scratch = runtime_counters.get("scratch_bytes_peak", 0)

    # Convert to MB
    measured_payload_mb = bytes_to_mb(measured_payload)
    measured_staging_mb = bytes_to_mb(measured_staging)
    measured_residual_mb = bytes_to_mb(measured_residual)
    measured_scratch_mb = bytes_to_mb(measured_scratch)

    # Determine measurement kind
    if measured_payload > 0 or measured_staging > 0:
        measurement_kind = "RUNTIME_COUNTERS"
    else:
        measurement_kind = "ESTIMATED"

    return {
        "estimated_payload_memory_mb": estimated,
        "measured_payload_memory_mb": measured_payload_mb,
        "measured_staging_memory_mb": measured_staging_mb,
        "measured_dense_residual_memory_mb": measured_residual_mb,
        "measured_scratch_peak_mb": measured_scratch_mb,
        "measurement_kind": measurement_kind,
    }
