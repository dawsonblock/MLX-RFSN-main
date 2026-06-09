#!/usr/bin/env python3
"""
RFSN v10 — Block-level unpacking and partial dequantization.

Provides selective reconstruction of only the KV blocks needed for a given
attention computation.  This is the core primitive for future sparse-decode and
local-attention paths.
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.compat import mx
import numpy as np

from rfsn_v10.bitpack import BitPackedQuantizer


def _dedupe_and_validate(
    block_indices: mx.array | list[int], num_blocks: int
) -> mx.array:
    """Validate block indices, deduplicate, and return sorted uint32 array."""
    if isinstance(block_indices, list):
        arr = mx.array(block_indices, dtype=mx.int32)
    else:
        arr = block_indices.astype(mx.int32)
    if arr.size == 0:
        raise ValueError("block_indices must not be empty")
    if mx.any(arr < 0).item() or mx.any(arr >= num_blocks).item():
        raise ValueError(
            f"Invalid block index (must be non-negative and < {num_blocks})"
        )
    # Deduplicate via numpy because MLX lacks unique() in all versions
    uniq = np.unique(np.array(arr))
    return mx.array(uniq, dtype=mx.uint32)


def unpack_code_blocks(
    packed: mx.array,
    block_indices: mx.array | list[int],
    bits: int,
    block_size: int,
) -> mx.array:
    """Unpack only the selected blocks from a packed code buffer.

    Args:
        packed: Packed uint32 array from BitPackedQuantizer.
        block_indices: Integer block IDs to extract.
        bits: Bit width per code (2–8 or >8 raw fallback).
        block_size: Number of codes per block.

    Returns:
        Unpacked codes for the selected blocks, concatenated in sorted order.
    """
    codes_per_word = 32 // bits if bits <= 8 else 1
    codes_per_word = max(1, codes_per_word)
    n_values_total = int(packed.size) * codes_per_word
    num_blocks = (n_values_total + block_size - 1) // block_size
    indices = _dedupe_and_validate(block_indices, num_blocks)

    if bits <= 8:
        # Unpack everything, then slice by block
        all_codes = BitPackedQuantizer.unpack(packed, n_values_total, bits)
    else:
        all_codes = packed[:n_values_total]

    selected = []
    for bid in indices.tolist():
        start = int(bid) * block_size
        end = min(start + block_size, n_values_total)
        selected.append(all_codes[start:end])
    return mx.concatenate(selected) if len(selected) > 1 else selected[0]


def unpack_blocks(
    packed: mx.array,
    scales: mx.array,
    *,
    block_ids: list[int],
    block_size: int,
    bits: int,
    group_size: int,
    shape: tuple[int, ...],
) -> mx.array:
    """Unpack and dequantize selected blocks from a packed Cartesian buffer.

    Args:
        packed: Packed uint32 codes of shape suitable for the full tensor.
        scales: Per-group float32 scale factors.
        block_ids: Block indices to reconstruct.
        block_size: Tokens per block (typically 64).
        bits: Quantization bit width.
        group_size: Group size for symmetric dequantization.
        shape: Original full tensor shape (B, H, T, D).

    Returns:
        Reconstructed tensor containing only the selected blocks,
        with the same B, H, D dimensions and T = len(block_ids) * block_size.
    """
    b, h, t, d = shape
    num_blocks = t // block_size
    if t % block_size != 0:
        raise ValueError(f"shape T={t} must be divisible by block_size={block_size}")

    indices = _dedupe_and_validate(block_ids, num_blocks)
    groups_per_block = (block_size * d) // group_size

    # Unpack all codes.  For grouped Cartesian the number of codes equals
    # the number of groups: (B*H*T*D) // group_size.
    codes_per_word = 32 // bits if bits <= 8 else 1
    codes_per_word = max(1, codes_per_word)
    # Use the *scales* shape to determine the true number of groups per head,
    # which avoids over-counting padding words in the packed buffer.
    total_groups_per_head = scales.shape[2]
    total_codes = b * h * total_groups_per_head
    if bits <= 8:
        all_codes = BitPackedQuantizer.unpack(packed, total_codes, bits)
    else:
        all_codes = packed[:total_codes]

    # Reshape to (B, H, total_groups_per_head)
    all_codes = all_codes.reshape(b, h, total_groups_per_head)

    qmax = (1 << (bits - 1)) - 1
    selected_blocks = []
    for bid in indices.tolist():
        bid = int(bid)
        g_start = bid * groups_per_block
        g_end = g_start + groups_per_block
        block_codes = all_codes[:, :, g_start:g_end]
        # Dequantize at group level
        q_signed = block_codes.astype(mx.float32) - float(qmax)
        block_scales = scales[:, :, g_start:g_end]
        restored_groups = q_signed * block_scales
        # Expand each group value to group_size elements via broadcasting
        expanded = restored_groups[:, :, :, None] * mx.ones(
            (b, h, groups_per_block, group_size), dtype=mx.float32
        )
        # reshape (B,H,groups_per_block,group_size) → (B,H,block_size,D)
        selected_blocks.append(expanded.reshape(b, h, block_size, d))

    return mx.concatenate(selected_blocks, axis=2)


def dequantize_k_blocks(
    packet: Any, block_indices: mx.array | list[int]
) -> mx.array:
    """Dequantize only selected K blocks from a QuantizedKVPacket-like object.

    Args:
        packet: An object with at minimum:
            - k_packed (mx.array or packed structure)
            - k_scales (mx.array)
            - k_bits (int)
            - block_size (int)
            - num_blocks (int)
            - k_block_packed_offsets (list[int])
            - k_block_scale_offsets (list[int])
            - k_block_n_values (list[int])
        block_indices: Block IDs to reconstruct.

    Returns:
        Reconstructed K tensor for selected blocks.
    """
    indices = _dedupe_and_validate(block_indices, packet.num_blocks)
    out_blocks = []
    for bid in indices.tolist():
        bid = int(bid)
        poff = packet.k_block_packed_offsets[bid]
        soff = packet.k_block_scale_offsets[bid]
        nval = packet.k_block_n_values[bid]
        if hasattr(packet.k_packed, "packed"):
            sub_packed = packet.k_packed.packed[poff : poff + (nval + 1)]
        else:
            sub_packed = packet.k_packed[poff : poff + (nval + 1)]
        codes = unpack_code_blocks(
            sub_packed, [0], packet.k_bits, packet.block_size
        )
        flat = codes.astype(mx.float32).reshape(-1)
        qmax = (1 << (packet.k_bits - 1)) - 1 if packet.k_bits <= 16 else 127
        q_signed = flat - float(qmax)
        if packet.k_scales.ndim == 0:
            scale = packet.k_scales[soff]
        else:
            scale_end = packet.k_block_scale_offsets[bid + 1]
            scale = packet.k_scales[soff:scale_end]
        restored = q_signed * scale
        out_blocks.append(restored)
    return mx.concatenate(out_blocks)


def dequantize_v_blocks(
    packet: Any, block_indices: mx.array | list[int]
) -> mx.array:
    """Dequantize only selected V blocks from a QuantizedKVPacket-like object.

    Same interface as ``dequantize_k_blocks`` but operates on V fields.
    """
    indices = _dedupe_and_validate(block_indices, packet.num_blocks)
    out_blocks = []
    for bid in indices.tolist():
        bid = int(bid)
        poff = packet.v_block_packed_offsets[bid]
        soff = packet.v_block_scale_offsets[bid]
        nval = packet.v_block_n_values[bid]
        if hasattr(packet.v_packed, "packed"):
            sub_packed = packet.v_packed.packed[poff : poff + (nval + 1)]
        else:
            sub_packed = packet.v_packed[poff : poff + (nval + 1)]
        codes = unpack_code_blocks(
            sub_packed, [0], packet.v_bits, packet.block_size
        )
        flat = codes.astype(mx.float32).reshape(-1)
        qmax = (1 << (packet.v_bits - 1)) - 1 if packet.v_bits <= 16 else 127
        q_signed = flat - float(qmax)
        if packet.v_scales.ndim == 0:
            scale = packet.v_scales[soff]
        else:
            scale_end = packet.v_block_scale_offsets[bid + 1]
            scale = packet.v_scales[soff:scale_end]
        restored = q_signed * scale
        out_blocks.append(restored)
    return mx.concatenate(out_blocks)


def dequantize_kv_blocks(
    k_packet: Any,
    v_packet: Any,
    block_indices: mx.array | list[int],
) -> tuple[mx.array, mx.array]:
    """Dequantize selected K and V blocks from packet-like objects.

    Args:
        k_packet: Object with fields required by ``dequantize_k_blocks``.
        v_packet: Object with fields required by ``dequantize_v_blocks``.
        block_indices: Block IDs to reconstruct for both K and V.

    Returns:
        Tuple of reconstructed (K, V) tensors for selected blocks.
    """
    k_blocks = dequantize_k_blocks(k_packet, block_indices)
    v_blocks = dequantize_v_blocks(v_packet, block_indices)
    return k_blocks, v_blocks


def dequantize_full(
    packed: mx.array,
    scales: mx.array,
    *,
    bits: int,
    group_size: int,
    shape: tuple[int, ...],
    block_size: int | None = None,
) -> mx.array:
    """Fully unpack and dequantize the entire packed buffer.

    Args:
        packed: Packed uint32 codes.
        scales: Per-group float32 scale factors.
        bits: Quantization bit width.
        group_size: Group size for symmetric dequantization.
        shape: Original tensor shape (B, H, T, D).
        block_size: Tokens per block; defaults to *t* (full tensor as one block).

    Returns:
        Fully reconstructed tensor of shape *shape*.
    """
    b, h, t, d = shape
    if block_size is None:
        block_size = t
    num_blocks = t // block_size
    return unpack_blocks(
        packed,
        scales,
        block_ids=list(range(num_blocks)),
        block_size=block_size,
        bits=bits,
        group_size=group_size,
        shape=shape,
    )
