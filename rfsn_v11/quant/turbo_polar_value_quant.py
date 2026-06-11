"""Value quantization scaffolding for TurboPolar (Phase 9 — future).

Current default: dense fp16 values.
Planned options:
  - 8-bit per-token or per-group symmetric quantization
  - 4-bit grouped quantization (after 8-bit proves safe)

Do NOT enable aggressive value quantization until K-only + dense V passes
all teacher-forced logit gates.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ValueQuantConfig:
    """Value quantization configuration for TurboPolar."""

    mode: str = "dense_fp16"  # dense_fp16 | per_token_8bit | grouped_8bit | grouped_4bit
    group_size: int = 64
    bits: int = 8


def quantize_values_dense(values: mx.array) -> mx.array:
    """Store values as dense fp16 (current default)."""
    return values.astype(mx.float16)


def quantize_values_per_token_8bit(values: mx.array) -> tuple[mx.array, mx.array]:
    """Per-token 8-bit symmetric quantization (scaffolding).

    Returns:
        (quantized_codes, scales) where scales is per-token.
    """
    # Find per-token min/max
    vmin = mx.min(values, axis=-1, keepdims=True)
    vmax = mx.max(values, axis=-1, keepdims=True)
    scale = (vmax - vmin) / 255.0
    scale = mx.where(scale == 0, 1.0, scale)
    codes = mx.clip(((values - vmin) / scale).astype(mx.uint8), 0, 255)
    return codes, scale


def dequantize_values_per_token_8bit(codes: mx.array, scales: mx.array, vmin: mx.array) -> mx.array:
    """Dequantize per-token 8-bit values."""
    return codes.astype(mx.float32) * scales + vmin


def quantize_values_grouped_8bit(values: mx.array, group_size: int = 64) -> tuple[mx.array, mx.array, mx.array]:
    """Grouped 8-bit symmetric quantization (scaffolding).

    Returns:
        (codes, scales, zeros)
    """
    # Pad to multiple of group_size
    orig_len = values.shape[-2] if values.ndim >= 2 else values.shape[0]
    pad_len = (group_size - orig_len % group_size) % group_size
    if pad_len:
        pad_shape = list(values.shape)
        pad_shape[-2] = pad_len
        values = mx.concatenate([values, mx.zeros(pad_shape, dtype=values.dtype)], axis=-2)

    # Reshape to groups
    group_shape = values.shape[:-2] + (-1, group_size, values.shape[-1])
    grouped = values.reshape(group_shape)

    vmin = mx.min(grouped, axis=-2, keepdims=True)
    vmax = mx.max(grouped, axis=-2, keepdims=True)
    scale = (vmax - vmin) / 255.0
    scale = mx.where(scale == 0, 1.0, scale)
    codes = mx.clip(((grouped - vmin) / scale).astype(mx.uint8), 0, 255)
    return codes, scale, vmin


def dequantize_values_grouped_8bit(
    codes: mx.array, scales: mx.array, vmin: mx.array
) -> mx.array:
    """Dequantize grouped 8-bit values."""
    return codes.astype(mx.float32) * scales + vmin
