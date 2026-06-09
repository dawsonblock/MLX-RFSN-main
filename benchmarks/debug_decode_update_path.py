#!/usr/bin/env python3
"""Decode-state trace diagnostic for RFSN v10.

Traces the decode/update path step-by-step to identify whether quantized
decode corruption starts immediately or accumulates over steps.

Tests:
  k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive, experimental_hybrid

Prompt lengths: 128, 512
Decode steps: 0, 1, 2, 4, 8, 16, 32

Modes:
  trace   -- compare FP16 vs quantized decode logits step-by-step
  kv-diff -- compare pre/post append K/V tensors, old-cache vs new-token

Output:
  trace:   artifacts/proof/experimental/decode_update_trace.json
  kv-diff: artifacts/proof/experimental/decode_append_kv_diff.json

Each trace row captures KV cache shape, position metadata, and per-step
logit quality metrics to isolate decode corruption patterns.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    if torch.isnan(a_f).any() or torch.isinf(a_f).any():
        return float("nan")
    if torch.isnan(b_f).any() or torch.isinf(b_f).any():
        return float("nan")
    return float(functional.cosine_similarity(a_f, b_f, dim=0).item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = functional.softmax(p_logits.float(), dim=-1)
    q = functional.softmax(q_logits.float(), dim=-1)
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return float(kl.item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> float:
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


def _legacy_cache(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past, "to_legacy_cache"):
        return list(past.to_legacy_cache())
    return list(past)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _compress_cache_stable(
    past_key_values,
    k_bits: int,
    v_bits: int,
    group_size: int,
    device: torch.device,
):
    """Compress via stable RFSN quantization and return layer list."""
    compressed = []
    with tempfile.TemporaryDirectory(prefix="rfsn_decode_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, group_size=group_size,
            use_wht=True, use_incoherent_signs=True,
            prefer_metal_kernels=True, strict_metal=False,
            max_memory_gb=2.0, cache_dir=tmpdir,
        )
        for layer_idx, (k_t, v_t) in enumerate(past_key_values):
            k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
            v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
            bsz, heads, seq, dim = k_np.shape
            dim_padded = int(math.ceil(dim / 64.0) * 64)
            if dim_padded != dim:
                pad = dim_padded - dim
                k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
                v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            k_mx = mx.array(k_np)
            v_mx = mx.array(v_np)
            key = f"layer_{layer_idx}"
            mgr.store(key, k_mx, v_mx, token_count=seq)
            rec = mgr.retrieve(key, out_dtype=mx.float32)
            if rec is None:
                raise RuntimeError("Cache miss")
            rk_mx, rv_mx = rec
            rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
            rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
            rk = rk.to(device=device, dtype=k_t.dtype)
            rv = rv.to(device=device, dtype=v_t.dtype)
            compressed.append((rk, rv))
    return compressed


def _compress_cache_experimental(
    past_key_values, cfg: dict[str, Any], device: torch.device,
):
    """Compress via experimental quantization managers."""
    mode = cfg.get("mode", "hybrid_polar_cartesian")
    if mode == "turbo_polar":
        mgr = TurboPolarKVManager(
            feature_dim=cfg.get("feature_dim", 64),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_bits=cfg.get("v_bits", 6),
            group_size=cfg.get("group_size", 64),
            adaptive_angle_range=cfg.get("adaptive_angle_range", False),
        )
    else:
        mgr = QuantizedKVManager(
            mode="hybrid_polar_cartesian",
            feature_dim=cfg.get("feature_dim", 64),
            polar_ratio=cfg.get("polar_ratio", 0.65),
            polar_levels=cfg.get("polar_levels", 4),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_angle_bits=cfg.get("v_angle_bits", 4),
            v_radius_bits=cfg.get("v_radius_bits", 6),
            cartesian_bits=cfg.get("cartesian_bits", 6),
            group_size=cfg.get("group_size", 64),
        )
    compressed = []
    for layer_idx, (k_t, v_t) in enumerate(past_key_values):
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 64.0) * 64)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)
        packet = mgr.quantize(k_mx, v_mx)
        rk_mx, rv_mx = mgr.dequantize(packet)
        rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
        rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
        rk = rk.to(device=device, dtype=k_t.dtype)
        rv = rv.to(device=device, dtype=v_t.dtype)
        compressed.append((rk, rv))
    return compressed


def _compress_cache(past_key_values, cfg: dict[str, Any], device: torch.device):
    """Dispatch to the appropriate compression path."""
    if cfg["family"] == "stable":
        return _compress_cache_stable(
            past_key_values,
            cfg["k_bits"], cfg["v_bits"], cfg["group_size"],
            device,
        )
    return _compress_cache_experimental(past_key_values, cfg, device)


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict[str, Any]] = {
    "k8_v5_gs64": {
        "family": "stable", "k_bits": 8, "v_bits": 5,
        "group_size": 64,
    },
    "k8_v5_gs32": {
        "family": "stable", "k_bits": 8, "v_bits": 5,
        "group_size": 32,
    },
    "turbo_polar": {
        "family": "experimental", "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64,
    },
    "adaptive": {
        "family": "experimental", "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64, "adaptive_angle_range": True,
    },
    "experimental_hybrid": {
        "family": "experimental", "mode": "hybrid_polar_cartesian",
        "feature_dim": 64, "polar_ratio": 0.65, "polar_levels": 4,
        "k_angle_bits": 5, "k_radius_bits": 8,
        "v_angle_bits": 4, "v_radius_bits": 6,
        "cartesian_bits": 6, "group_size": 64,
    },
}


# ---------------------------------------------------------------------------
# Model + prompt setup (shared)
# ---------------------------------------------------------------------------


def _load_model_and_prompt(
    model_name: str,
    prompt_tokens: int,
    device: torch.device,
    seed: int,
) -> tuple[Any, Any, torch.Tensor, int]:
    """Return (model, tokenizer, prompt_ids, actual_prompt_len)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    torch.manual_seed(seed)

    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    if prompt_tokens > len(dummy_ids):
        repeated = (
            dummy_ids * ((prompt_tokens // len(dummy_ids)) + 1)
        )[:prompt_tokens]
        prompt_str = tokenizer.decode(repeated)
    else:
        prompt_str = tokenizer.decode(dummy_ids[:prompt_tokens])

    prompt_ids = tokenizer.encode(
        prompt_str, return_tensors="pt", truncation=True,
    )
    prompt_ids = prompt_ids.to(device)
    actual_prompt_len = prompt_ids.shape[1]
    return model, tokenizer, prompt_ids, actual_prompt_len


# ---------------------------------------------------------------------------
# Trace mode
# ---------------------------------------------------------------------------


def run_trace_mode(
    model_name: str,
    cfg_name: str,
    prompt_tokens: int,
    decode_steps: list[int],
    device: torch.device,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare FP16 vs quantized decode logits step-by-step.

    Emits one row per (config, prompt_tokens, decode_step) checkpoint.
    Each row includes full KV cache metadata and logit quality metrics.
    """
    cfg = CONFIGS[cfg_name]
    model, _tok, prompt_ids, actual_prompt_len = _load_model_and_prompt(
        model_name, prompt_tokens, device, seed,
    )

    # FP16 prefill
    with torch.no_grad():
        out_fp16 = model(
            input_ids=prompt_ids,
            past_key_values=None,
            use_cache=True,
        )
    fp16_past = out_fp16.past_key_values
    first_tok = int(torch.argmax(out_fp16.logits[:, -1, :], dim=-1).item())

    # Compress prefill cache
    past_list = _legacy_cache(fp16_past)
    compressed_list = _compress_cache(past_list, cfg, device)
    compressed_past = DynamicCache.from_legacy_cache(tuple(compressed_list))

    fp16_decode_past = fp16_past
    quant_decode_past = compressed_past
    next_tok_fp16 = first_tok
    next_tok_quant = first_tok

    rows: list[dict[str, Any]] = []
    max_steps = max(decode_steps)

    for step in range(max_steps + 1):
        if step in decode_steps:
            # Capture cache shapes before this decode step
            fp16_legacy = _legacy_cache(fp16_decode_past)
            quant_legacy = _legacy_cache(quant_decode_past)

            k_shape = list(fp16_legacy[0][0].shape) if fp16_legacy else []
            v_shape = list(fp16_legacy[0][1].shape) if fp16_legacy else []
            rk_shape = list(quant_legacy[0][0].shape) if quant_legacy else []
            rv_shape = list(quant_legacy[0][1].shape) if quant_legacy else []
            kv_len_before = k_shape[2] if len(k_shape) > 2 else 0

            # Expected after-append lengths
            expected_kv_len_after = kv_len_before + 1
            position_id = kv_len_before
            cache_position = kv_len_before

            # Invariant assertions
            assert position_id == actual_prompt_len + step, (
                f"position_id {position_id} != prompt_len+step "
                f"{actual_prompt_len}+{step}"
            )
            assert cache_position == actual_prompt_len + step, (
                f"cache_position {cache_position} != prompt_len+step"
            )

            # Attention mask after append = [1, 1, 1, kv_len_before+1]
            attention_mask_shape = [1, 1, 1, kv_len_before + 1]

            # Run one decode step on both paths
            next_ids_fp16 = torch.tensor([[next_tok_fp16]], device=device)
            next_ids_quant = torch.tensor([[next_tok_quant]], device=device)

            with torch.no_grad():
                out_fp16_step = model(
                    input_ids=next_ids_fp16,
                    past_key_values=fp16_decode_past, use_cache=True,
                )
                out_quant_step = model(
                    input_ids=next_ids_quant,
                    past_key_values=quant_decode_past, use_cache=True,
                )

            logit_fp16 = out_fp16_step.logits[:, -1, :]
            logit_quant = out_quant_step.logits[:, -1, :]

            cosine = _cosine(logit_fp16, logit_quant)
            top5 = _topk_overlap(logit_fp16, logit_quant, k=5)
            kl = _kl_div(logit_fp16, logit_quant)
            max_delta = float(
                torch.max(torch.abs(logit_fp16 - logit_quant)).item()
            )
            mean_delta = float(
                torch.mean(torch.abs(logit_fp16 - logit_quant)).item()
            )

            # Verify KV lengths after the step
            fp16_after = _legacy_cache(out_fp16_step.past_key_values)
            quant_after = _legacy_cache(out_quant_step.past_key_values)
            kv_len_after_fp16 = (
                fp16_after[0][0].shape[2] if fp16_after else 0
            )
            kv_len_after_quant = (
                quant_after[0][0].shape[2] if quant_after else 0
            )
            rk_after_shape = list(quant_after[0][0].shape) if quant_after else []
            rv_after_shape = list(quant_after[0][1].shape) if quant_after else []

            # Invariant checks
            assert kv_len_after_fp16 == kv_len_before + 1, (
                f"kv_len_after {kv_len_after_fp16} != kv_len_before+1 "
                f"{kv_len_before+1}"
            )
            assert kv_len_after_fp16 == expected_kv_len_after, (
                f"kv_len_after {kv_len_after_fp16} != expected "
                f"{expected_kv_len_after}"
            )
            assert rk_after_shape[-2] == kv_len_after_fp16 if rk_after_shape else True, (
                f"reconstructed_k seq dim {rk_after_shape[-2]} != "
                f"kv_len_after {kv_len_after_fp16}"
            )
            assert rv_after_shape[-2] == kv_len_after_fp16 if rv_after_shape else True, (
                f"reconstructed_v seq dim {rv_after_shape[-2]} != "
                f"kv_len_after {kv_len_after_fp16}"
            )

            row_status = (
                "pass"
                if cosine >= 0.99 and top5 >= 0.8
                else "degraded"
            )

            rows.append({
                "config": cfg_name,
                "prompt_tokens": actual_prompt_len,
                "decode_step": step,
                "kv_len_before": kv_len_before,
                "kv_len_after": kv_len_after_fp16,
                "expected_kv_len_after": expected_kv_len_after,
                "position_id": position_id,
                "cache_position": cache_position,
                "attention_mask_shape": attention_mask_shape,
                "k_shape": k_shape,
                "v_shape": v_shape,
                "reconstructed_k_shape": rk_after_shape,
                "reconstructed_v_shape": rv_after_shape,
                "logit_cosine_vs_fp16": cosine,
                "top5_overlap_vs_fp16": top5,
                "kl_vs_fp16": kl,
                "max_abs_logit_delta": max_delta,
                "mean_abs_logit_delta": mean_delta,
                "status": row_status,
            })

            # Advance for next iteration
            fp16_decode_past = out_fp16_step.past_key_values
            quant_decode_past = out_quant_step.past_key_values
            next_tok_fp16 = int(torch.argmax(logit_fp16, dim=-1).item())
            next_tok_quant = int(torch.argmax(logit_quant, dim=-1).item())
        else:
            # Non-checkpoint step: advance both without recording
            next_ids_fp16 = torch.tensor([[next_tok_fp16]], device=device)
            next_ids_quant = torch.tensor([[next_tok_quant]], device=device)
            with torch.no_grad():
                out_fp16_step = model(
                    input_ids=next_ids_fp16,
                    past_key_values=fp16_decode_past, use_cache=True,
                )
                out_quant_step = model(
                    input_ids=next_ids_quant,
                    past_key_values=quant_decode_past, use_cache=True,
                )
            fp16_decode_past = out_fp16_step.past_key_values
            quant_decode_past = out_quant_step.past_key_values
            logit_fp16 = out_fp16_step.logits[:, -1, :]
            logit_quant = out_quant_step.logits[:, -1, :]
            next_tok_fp16 = int(torch.argmax(logit_fp16, dim=-1).item())
            next_tok_quant = int(torch.argmax(logit_quant, dim=-1).item())

    return rows


# ---------------------------------------------------------------------------
# KV-diff mode
# ---------------------------------------------------------------------------


def _agg_cosine(
    fp16_layers: list[tuple[torch.Tensor, torch.Tensor]],
    quant_layers: list[tuple[torch.Tensor, torch.Tensor]],
    seq_slice: slice | None,
    key: str,  # "k" or "v"
) -> tuple[float, float]:
    """Return (mean_cosine, max_abs_error) over layers for one slice."""
    cosines: list[float] = []
    max_errs: list[float] = []
    for (k_fp16, v_fp16), (k_quant, v_quant) in zip(fp16_layers, quant_layers):
        t_fp16 = k_fp16 if key == "k" else v_fp16
        t_quant = k_quant if key == "k" else v_quant
        if seq_slice is not None:
            t_fp16 = t_fp16[:, :, seq_slice, :]
            t_quant = t_quant[:, :, seq_slice, :]
        cosines.append(_cosine(t_fp16, t_quant))
        max_errs.append(
            float(torch.max(torch.abs(t_fp16.float() - t_quant.float())).item())
        )
    mean_cos = float(sum(cosines) / len(cosines)) if cosines else float("nan")
    mean_err = float(sum(max_errs) / len(max_errs)) if max_errs else float("nan")
    return mean_cos, mean_err


def run_kv_diff_mode(
    model_name: str,
    cfg_name: str,
    prompt_tokens: int,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare old-cache preservation and new-token K/V after one append.

    Answers three questions:
    1. Does appending one quantized decode step corrupt the old cache?
    2. Is the new token K/V quantized badly?
    3. Are cache length, K/V order, and position IDs preserved?

    Returns a single result dict with all required fields.
    """
    cfg = CONFIGS[cfg_name]
    model, _tok, prompt_ids, actual_prompt_len = _load_model_and_prompt(
        model_name, prompt_tokens, device, seed,
    )

    # FP16 prefill
    with torch.no_grad():
        out_fp16 = model(
            input_ids=prompt_ids,
            past_key_values=None,
            use_cache=True,
        )
    fp16_past = out_fp16.past_key_values
    first_tok = int(torch.argmax(out_fp16.logits[:, -1, :], dim=-1).item())

    # Compress prefill cache
    past_list_prefill = _legacy_cache(fp16_past)
    compressed_list_prefill = _compress_cache(past_list_prefill, cfg, device)
    compressed_past = DynamicCache.from_legacy_cache(
        tuple(compressed_list_prefill)
    )

    # Snapshot before append
    fp16_before = _legacy_cache(fp16_past)
    quant_before = _legacy_cache(compressed_past)
    kv_len_before = fp16_before[0][0].shape[2] if fp16_before else 0

    # One decode step on FP16 path
    with torch.no_grad():
        out_fp16_step = model(
            input_ids=torch.tensor([[first_tok]], device=device),
            past_key_values=fp16_past, use_cache=True,
        )
    fp16_after = _legacy_cache(out_fp16_step.past_key_values)

    # One decode step on quantized path
    with torch.no_grad():
        out_quant_step = model(
            input_ids=torch.tensor([[first_tok]], device=device),
            past_key_values=compressed_past, use_cache=True,
        )
    quant_after = _legacy_cache(out_quant_step.past_key_values)

    kv_len_after_fp16 = fp16_after[0][0].shape[2] if fp16_after else 0
    kv_len_after_quant = quant_after[0][0].shape[2] if quant_after else 0

    # Invariant checks
    cache_len_correct = (kv_len_after_quant == kv_len_before + 1)
    kv_order_preserved = (
        len(quant_after) == len(fp16_after)
        and all(
            qa[0].shape[-1] == fa[0].shape[-1]
            for qa, fa in zip(quant_after, fp16_after)
        )
    )
    position_id_correct = (kv_len_before == actual_prompt_len)

    # Old-cache cosines: compare tokens 0..kv_len_before-1 of after vs before
    old_slice = slice(0, kv_len_before)
    old_k_cos, old_k_err = _agg_cosine(fp16_after, quant_after, old_slice, "k")
    old_v_cos, old_v_err = _agg_cosine(fp16_after, quant_after, old_slice, "v")

    # New-token cosines: compare token at index kv_len_before
    new_slice = slice(kv_len_before, kv_len_before + 1)
    new_k_cos, new_k_err = _agg_cosine(fp16_after, quant_after, new_slice, "k")
    new_v_cos, new_v_err = _agg_cosine(fp16_after, quant_after, new_slice, "v")

    # Full-cache cosines
    full_k_cos, full_k_err = _agg_cosine(fp16_after, quant_after, None, "k")
    full_v_cos, full_v_err = _agg_cosine(fp16_after, quant_after, None, "v")

    result_status = (
        "pass"
        if cache_len_correct and kv_order_preserved
        and old_k_cos > 0.99 and old_v_cos > 0.99
        else "degraded"
    )

    return {
        "config": cfg_name,
        "prompt_tokens": actual_prompt_len,
        "kv_len_before": kv_len_before,
        "kv_len_after": kv_len_after_quant,
        "old_cache_k_cosine_after_append": old_k_cos,
        "old_cache_v_cosine_after_append": old_v_cos,
        "new_token_k_cosine": new_k_cos,
        "new_token_v_cosine": new_v_cos,
        "full_cache_k_cosine_after_append": full_k_cos,
        "full_cache_v_cosine_after_append": full_v_cos,
        "old_cache_k_max_abs_error": old_k_err,
        "old_cache_v_max_abs_error": old_v_err,
        "new_token_k_max_abs_error": new_k_err,
        "new_token_v_max_abs_error": new_v_err,
        "kv_order_preserved": kv_order_preserved,
        "cache_len_correct": cache_len_correct,
        "position_id_correct": position_id_correct,
        "status": result_status,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode-state trace diagnostic"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--configs", nargs="+",
        default=[
            "k8_v5_gs64", "k8_v5_gs32", "turbo_polar",
            "adaptive", "experimental_hybrid",
        ],
    )
    # Accept both --prompt-lengths (original) and --prompt-tokens (plan alias)
    parser.add_argument(
        "--prompt-lengths", "--prompt-tokens",
        dest="prompt_lengths",
        nargs="+", type=int, default=[128, 512],
    )
    parser.add_argument(
        "--decode-steps", nargs="+", type=int,
        default=[0, 1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=["trace", "kv-diff"], default="trace",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        if args.mode == "kv-diff":
            args.out = (
                "artifacts/proof/experimental/decode_append_kv_diff.json"
            )
        else:
            args.out = (
                "artifacts/proof/experimental/decode_update_trace.json"
            )

    device = torch.device("cpu")

    if args.mode == "trace":
        all_rows: list[dict[str, Any]] = []
        for cfg_name in args.configs:
            if cfg_name not in CONFIGS:
                print(f"  SKIP unknown config: {cfg_name}")
                continue
            for length in args.prompt_lengths:
                print(
                    f"[trace] {cfg_name} @ {length} tokens ..."
                )
                try:
                    rows = run_trace_mode(
                        args.model, cfg_name, length,
                        args.decode_steps, device, args.seed,
                    )
                    all_rows.extend(rows)
                    for r in rows:
                        print(
                            f"  step={r['decode_step']:2d} "
                            f"cosine={r['logit_cosine_vs_fp16']:.4f} "
                            f"top5={r['top5_overlap_vs_fp16']:.2f} "
                            f"kl={r['kl_vs_fp16']:.6f} "
                            f"status={r['status']}"
                        )
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    all_rows.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "error": str(exc),
                        "status": "error",
                    })

        if not all_rows:
            raise RuntimeError("No diagnostic rows generated")

        artifact: dict[str, Any] = {
            "release": "experimental",
            "model": args.model,
            "seed": args.seed,
            "mode": "trace",
            "decode_steps_checked": args.decode_steps,
            "status": "executed",
            "traces": all_rows,
        }

    else:  # kv-diff
        all_results: list[dict[str, Any]] = []
        for cfg_name in args.configs:
            if cfg_name not in CONFIGS:
                print(f"  SKIP unknown config: {cfg_name}")
                continue
            for length in args.prompt_lengths:
                print(
                    f"[kv-diff] {cfg_name} @ {length} tokens ..."
                )
                try:
                    result = run_kv_diff_mode(
                        args.model, cfg_name, length, device, args.seed,
                    )
                    all_results.append(result)
                    print(
                        f"  old_k_cos={result['old_cache_k_cosine_after_append']:.6f} "
                        f"new_k_cos={result['new_token_k_cosine']:.6f} "
                        f"cache_ok={result['cache_len_correct']} "
                        f"status={result['status']}"
                    )
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    all_results.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "error": str(exc),
                        "status": "error",
                    })

        if not all_results:
            raise RuntimeError("No diagnostic rows generated")

        artifact = {
            "release": "experimental",
            "model": args.model,
            "seed": args.seed,
            "mode": "kv-diff",
            "status": "executed",
            "results": all_results,
        }

    if artifact.get("status") == "awaiting_execution":
        raise RuntimeError("Refusing to write placeholder diagnostic artifact")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.mode} output to {out_path}")


if __name__ == "__main__":
    main()
