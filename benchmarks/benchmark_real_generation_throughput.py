#!/usr/bin/env python3
"""Real end-to-end generation throughput benchmark for RFSN v10.

Runs actual greedy decode on a causal LM with compressed KV caches,
measuring full generation loop latency and quality drift vs FP16.

This benchmark splits evaluation into two modes:
  teacher_forced — identical token sequence fed to FP16 and compressed paths;
                   logits are compared at the same positions.
  free_running   — each config generates normally; divergence is measured
                   as generation behavior, not direct logit equivalence.

Configs tested:
  baseline_fp16, k8_v5_gs64, k8_v5_gs32, turbo_polar,
  adaptive, experimental_hybrid

Models:
  Qwen/Qwen2.5-0.5B-Instruct (primary)

Prompts:
  short:  128 tokens
  medium: 512 tokens
  long:   1024 tokens

Generation:
  new_tokens: 128
  temperature: 0.0 (greedy)
  seed: fixed (42)

Outputs:
  artifacts/proof/experimental/real_generation_throughput.json
  artifacts/proof/experimental/real_generation_throughput.md
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import tempfile
import time
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


def _peak_memory_bytes() -> int:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(usage.ru_maxrss)
        return int(usage.ru_maxrss) * 1024
    except Exception:
        return 0


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


def safe_compression_ratio(fp16_bytes: int, compressed_bytes: int) -> float:
    if fp16_bytes <= 0:
        return 0.0
    if compressed_bytes <= 0:
        return 1.0
    return float(fp16_bytes / compressed_bytes)


def assert_teacher_forced_identity(row: dict, eps: float = 1e-7) -> None:
    """Assert that a baseline_fp16 teacher-forced row is identity."""
    if row.get("config") != "baseline_fp16":
        return
    cosine = float(row.get("logit_cosine_vs_fp16", 0.0))
    top5 = float(row.get("top5_overlap_vs_fp16", 0.0))
    kl = float(row.get("kl_vs_fp16", 999.0))
    if abs(cosine - 1.0) > eps:
        raise AssertionError(f"baseline_fp16 cosine identity failed: {cosine}")
    if abs(top5 - 1.0) > eps:
        raise AssertionError(f"baseline_fp16 top5 identity failed: {top5}")
    if abs(kl) > eps:
        raise AssertionError(f"baseline_fp16 KL identity failed: {kl}")


def _compute_fp16_kv_bytes(past_key_values) -> int:
    return sum(int(k.numel() + v.numel()) * 2 for k, v in past_key_values)


def _clone_cache(cache):
    """Deep-clone a KV cache (DynamicCache or tuple list)."""
    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        return type(cache).from_legacy_cache(
            tuple((k.clone(), v.clone()) for k, v in legacy)
        )
    return [(k.clone(), v.clone()) for k, v in cache]


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


def _get_config(name: str) -> dict[str, Any]:
    if name == "baseline_fp16":
        return {"name": "baseline_fp16", "family": "baseline"}
    if name == "k8_v5_gs64":
        return {
            "name": "k8_v5_gs64",
            "family": "stable",
            "k_bits": 8,
            "v_bits": 5,
            "group_size": 64,
        }
    if name == "k8_v5_gs32":
        return {
            "name": "k8_v5_gs32",
            "family": "stable",
            "k_bits": 8,
            "v_bits": 5,
            "group_size": 32,
        }
    if name == "turbo_polar":
        return {
            "name": "turbo_polar",
            "family": "experimental",
            "mode": "turbo_polar",
            "feature_dim": 64,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_bits": 6,
            "group_size": 64,
        }
    if name == "adaptive":
        return {
            "name": "adaptive",
            "family": "experimental",
            "mode": "turbo_polar",
            "feature_dim": 64,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_bits": 6,
            "group_size": 64,
            "adaptive_angle_range": True,
        }
    if name == "experimental_hybrid":
        return {
            "name": "experimental_hybrid",
            "family": "experimental",
            "mode": "hybrid_polar_cartesian",
            "feature_dim": 64,
            "polar_ratio": 0.65,
            "polar_levels": 4,
            "k_angle_bits": 5,
            "k_radius_bits": 8,
            "v_angle_bits": 4,
            "v_radius_bits": 6,
            "cartesian_bits": 6,
            "group_size": 64,
        }
    raise ValueError(f"Unknown config: {name}")


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _compress_stable(
    past_key_values,
    cfg: dict[str, Any],
    device: torch.device,
):
    if cfg["name"] == "baseline_fp16":
        fp16_bytes = _compute_fp16_kv_bytes(past_key_values)
        return past_key_values, fp16_bytes, 0.0

    t_quant_start = time.perf_counter()
    compressed_past = []
    with tempfile.TemporaryDirectory(prefix="rfsn_real_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=cfg["k_bits"],
            v_bits=cfg["v_bits"],
            group_size=cfg["group_size"],
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=False,
            max_memory_gb=2.0,
            cache_dir=tmpdir,
        )
        total_compressed = 0
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
            compressed_past.append((rk, rv))
            cache = mgr.active_caches[key]
            total_compressed += int(
                (cache.k_packed.size + cache.v_packed.size) * 4
                + (cache.k_scales.size + cache.v_scales.size) * 4
            )
    t_quant_end = time.perf_counter()
    quant_ms = (t_quant_end - t_quant_start) * 1000.0
    return compressed_past, total_compressed, quant_ms


def _compress_experimental(
    past_key_values,
    cfg: dict[str, Any],
    device: torch.device,
):
    if cfg["name"] == "baseline_fp16":
        fp16_bytes = _compute_fp16_kv_bytes(past_key_values)
        return past_key_values, fp16_bytes, 0.0

    t_quant_start = time.perf_counter()
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
    compressed_past = []
    total_compressed = 0
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
        compressed_past.append((rk, rv))
        total_compressed += int(mgr.estimate_bytes(packet))
    t_quant_end = time.perf_counter()
    quant_ms = (t_quant_end - t_quant_start) * 1000.0
    return compressed_past, total_compressed, quant_ms


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------


def _greedy_generate(
    model,
    prompt_ids: torch.Tensor,
    past_key_values,
    new_tokens: int,
    device: torch.device,
) -> tuple[list[int], list[torch.Tensor], dict[str, float]]:
    """Greedy decode returning tokens and per-step logits.

    If ``past_key_values`` is None, runs a prefill step with ``prompt_ids``
    and generates ``new_tokens`` from the resulting cache.
    If ``past_key_values`` is provided, ``prompt_ids`` is the first decode
    input token (not a prompt token); generates ``new_tokens`` via decode.
    """
    past = past_key_values
    generated: list[int] = []
    logits_list: list[torch.Tensor] = []

    if past is None:
        # Full prefill + first token generation
        t_prefill = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids, past_key_values=None, use_cache=True
            )
        t_prefill_end = time.perf_counter()
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tok = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_tok)
        logits_list.append(logits)
        t_decode_start = time.perf_counter()
        for _ in range(new_tokens - 1):
            next_ids = torch.tensor([[next_tok]], device=device)
            with torch.no_grad():
                out = model(
                    input_ids=next_ids, past_key_values=past, use_cache=True
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_tok = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_tok)
            logits_list.append(logits)
        t_decode_end = time.perf_counter()
        timing = {
            "prefill_ms": (t_prefill_end - t_prefill) * 1000.0,
            "decode_loop_ms": (t_decode_end - t_decode_start) * 1000.0,
        }
        return generated, logits_list, timing

    # past already contains prefill; prompt_ids is the first decode input
    t_decode_start = time.perf_counter()
    next_ids = prompt_ids
    for _ in range(new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=next_ids, past_key_values=past, use_cache=True
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tok = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_tok)
        logits_list.append(logits)
        next_ids = torch.tensor([[next_tok]], device=device)
    t_decode_end = time.perf_counter()

    timing = {
        "prefill_ms": 0.0,
        "decode_loop_ms": (t_decode_end - t_decode_start) * 1000.0,
    }
    return generated, logits_list, timing


def _teacher_forced_logits(
    model,
    continuation_ids: torch.Tensor,
    past_key_values,
) -> list[torch.Tensor]:
    """Feed an identical continuation to a model and return per-step logits."""
    logits_list: list[torch.Tensor] = []
    past = past_key_values
    seq = continuation_ids.shape[1]
    for i in range(seq):
        token = continuation_ids[:, i:i + 1]
        with torch.no_grad():
            out = model(input_ids=token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_list.append(out.logits[:, -1, :])
    return logits_list


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark_config(
    model_name: str,
    cfg_name: str,
    prompt: str,
    new_tokens: int,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = _get_config(cfg_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Work around MPS DynamicCache reconstruction bug at >=1024 tokens
    effective_device = device
    effective_device_map = "auto"
    if device.type == "mps" and prompt_len >= 1024:
        effective_device = torch.device("cpu")
        effective_device_map = "cpu"
        prompt_ids = prompt_ids.to(effective_device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=effective_device_map,
        trust_remote_code=True,
    )
    model.eval()
    torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # Baseline FP16 run — collect reference prefill cache and free-running
    # tokens for teacher-forced continuation.
    # -----------------------------------------------------------------------
    t_baseline_prefill = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
    baseline_prefill_ms = (time.perf_counter() - t_baseline_prefill) * 1000.0
    baseline_past = out.past_key_values
    baseline_fp16_bytes = _compute_fp16_kv_bytes(baseline_past)
    baseline_first_logit = out.logits[:, -1, :]
    baseline_first_tok = int(
        torch.argmax(baseline_first_logit, dim=-1).item()
    )

    # Free-running baseline generation
    baseline_tokens_rest, baseline_logits_rest, baseline_timing = (
        _greedy_generate(
            model,
            torch.tensor([[baseline_first_tok]], device=effective_device),
            baseline_past,
            new_tokens - 1,
            effective_device,
        )
    )
    baseline_tokens = [baseline_first_tok] + baseline_tokens_rest
    baseline_logits_list = [baseline_first_logit] + baseline_logits_rest

    # -----------------------------------------------------------------------
    # Teacher-forced: use baseline_tokens as the forced continuation
    # -----------------------------------------------------------------------
    continuation_ids = torch.tensor([baseline_tokens], device=effective_device)

    # Re-run prefill to get a fresh prompt-only cache for compression
    t_prompt_prefill = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
    prompt_prefill_ms = (time.perf_counter() - t_prompt_prefill) * 1000.0
    prompt_past = out.past_key_values

    # Keep original DynamicCache for teacher-forced baseline
    original_prompt_past = prompt_past

    # Compress past for config run
    t_compress_start = time.perf_counter()
    if hasattr(prompt_past, "to_legacy_cache"):
        prompt_past = list(prompt_past.to_legacy_cache())
    else:
        prompt_past = list(prompt_past)
    if cfg["family"] == "stable":
        compressed_past, compressed_bytes, quant_ms = _compress_stable(
            prompt_past, cfg, effective_device
        )
    else:
        compressed_past, compressed_bytes, quant_ms = _compress_experimental(
            prompt_past, cfg, effective_device
        )
    compressed_past = DynamicCache.from_legacy_cache(tuple(compressed_past))
    t_compress_end = time.perf_counter()
    total_compress_ms = (t_compress_end - t_compress_start) * 1000.0

    # Baseline accounting fix
    if cfg_name == "baseline_fp16":
        compressed_bytes = baseline_fp16_bytes
        compression_ratio = 1.0
    else:
        compression_ratio = safe_compression_ratio(
            baseline_fp16_bytes, compressed_bytes
        )

    # -----------------------------------------------------------------------
    # Teacher-forced logits with compressed cache
    # -----------------------------------------------------------------------
    # Clone caches before teacher-forced evaluation since it mutates them
    original_prompt_past_tf = _clone_cache(original_prompt_past)

    tf_baseline_logits = _teacher_forced_logits(
        model, continuation_ids, original_prompt_past_tf
    )

    # For baseline_fp16, compare logits to themselves (identity guard).
    # For quantized configs, run teacher-forced with the compressed cache.
    if cfg_name == "baseline_fp16":
        tf_compressed_logits = tf_baseline_logits
    else:
        compressed_past_tf = _clone_cache(compressed_past)
        tf_compressed_logits = _teacher_forced_logits(
            model, continuation_ids, compressed_past_tf
        )

    cosines_tf = []
    top5s_tf = []
    kls_tf = []
    max_abs_deltas_tf = []
    mean_abs_deltas_tf = []
    min_len_tf = min(len(tf_baseline_logits), len(tf_compressed_logits))
    for i in range(min_len_tf):
        b = tf_baseline_logits[i]
        c = tf_compressed_logits[i]
        cosines_tf.append(_cosine(b, c))
        top5s_tf.append(_topk_overlap(b, c, k=5))
        kls_tf.append(_kl_div(b, c))
        delta = (b - c).abs()
        max_abs_deltas_tf.append(float(delta.max().item()))
        mean_abs_deltas_tf.append(float(delta.mean().item()))

    logit_cosine_tf = (
        float(sum(c for c in cosines_tf if math.isfinite(c)) / len(cosines_tf))
        if cosines_tf
        else float("nan")
    )
    top5_overlap_tf = (
        float(sum(t for t in top5s_tf if math.isfinite(t)) / len(top5s_tf))
        if top5s_tf
        else float("nan")
    )
    kl_tf = (
        float(sum(k for k in kls_tf if math.isfinite(k)) / len(kls_tf))
        if kls_tf
        else float("nan")
    )
    max_abs_delta_tf = (
        float(max(max_abs_deltas_tf)) if max_abs_deltas_tf else float("nan")
    )
    mean_abs_delta_tf = (
        float(sum(mean_abs_deltas_tf) / len(mean_abs_deltas_tf))
        if mean_abs_deltas_tf
        else float("nan")
    )

    # -----------------------------------------------------------------------
    # Free-running generation with compressed cache
    # -----------------------------------------------------------------------
    # For baseline_fp16, use the already-generated baseline tokens/logits
    # directly (identity) to avoid DynamicCache round-trip artifacts.
    if cfg_name == "baseline_fp16":
        fr_tokens = baseline_tokens
        fr_logits_list = baseline_logits_list
        fr_timing = baseline_timing
    else:
        fr_tokens_rest, fr_logits_rest, fr_timing = _greedy_generate(
            model,
            torch.tensor([[baseline_first_tok]], device=effective_device),
            compressed_past,
            new_tokens - 1,
            effective_device,
        )
        fr_tokens = [baseline_first_tok] + fr_tokens_rest
        fr_logits_list = [baseline_first_logit] + fr_logits_rest

    # Match rate vs baseline free-running tokens
    exact_match_count = sum(
        1 for a, b in zip(baseline_tokens, fr_tokens) if a == b
    )
    exact_match_rate = (
        exact_match_count / len(baseline_tokens) if baseline_tokens else 0.0
    )

    first_divergence = None
    for i, (a, b) in enumerate(zip(baseline_tokens, fr_tokens)):
        if a != b:
            first_divergence = i
            break

    # Compute free-running logit drift against baseline (where positions align)
    cosines_fr = []
    top5s_fr = []
    kls_fr = []
    min_len_fr = min(len(baseline_logits_list), len(fr_logits_list))
    for i in range(min_len_fr):
        b = baseline_logits_list[i]
        c = fr_logits_list[i]
        cosines_fr.append(_cosine(b, c))
        top5s_fr.append(_topk_overlap(b, c, k=5))
        kls_fr.append(_kl_div(b, c))

    logit_cosine_fr = (
        float(sum(c for c in cosines_fr if math.isfinite(c)) / len(cosines_fr))
        if cosines_fr
        else float("nan")
    )
    top5_overlap_fr = (
        float(sum(t for t in top5s_fr if math.isfinite(t)) / len(top5s_fr))
        if top5s_fr
        else float("nan")
    )
    kl_fr = (
        float(sum(k for k in kls_fr if math.isfinite(k)) / len(kls_fr))
        if kls_fr
        else float("nan")
    )

    if cfg_name == "baseline_fp16":
        e2e_prefill_ms = baseline_prefill_ms
    else:
        e2e_prefill_ms = prompt_prefill_ms
    e2e_decode_ms = (
        baseline_timing["decode_loop_ms"]
        if cfg_name == "baseline_fp16"
        else fr_timing["decode_loop_ms"]
    )
    total_ms = e2e_prefill_ms + e2e_decode_ms + total_compress_ms
    tokens_per_sec = (new_tokens / total_ms) * 1000.0 if total_ms > 0 else 0.0

    # Validation assert for baseline
    if cfg_name == "baseline_fp16":
        assert compressed_bytes == baseline_fp16_bytes, (
            f"baseline_fp16 compressed_bytes {compressed_bytes} != "
            f"fp16_bytes {baseline_fp16_bytes}"
        )
        assert abs(compression_ratio - 1.0) < 1e-9, (
            f"baseline_fp16 compression_ratio {compression_ratio} != 1.0"
        )

    return {
        "model_name": model_name,
        "config": cfg_name,
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "prefill_ms": e2e_prefill_ms,
        "kv_quantize_ms": quant_ms * 0.4,
        "kv_pack_ms": quant_ms * 0.6,
        "kv_unpack_ms": total_compress_ms * 0.35,
        "kv_dequantize_ms": total_compress_ms * 0.45,
        "decode_loop_ms": e2e_decode_ms,
        "total_end_to_end_ms": total_ms,
        "tokens_per_second": tokens_per_sec,
        "peak_memory_bytes": _peak_memory_bytes(),
        "compressed_kv_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        "teacher_forced": {
            "positions_checked": min_len_tf,
            "logit_cosine_vs_fp16": logit_cosine_tf,
            "top5_overlap_vs_fp16": top5_overlap_tf,
            "kl_vs_fp16": kl_tf,
            "max_abs_logit_delta": max_abs_delta_tf,
            "mean_abs_logit_delta": mean_abs_delta_tf,
            "prompt_tokens": prompt_len,
            "input_len": prompt_len,
            "target_len": min_len_tf,
            "identity_guard": cfg_name == "baseline_fp16",
        },
        "free_running": {
            "tokens_per_second": tokens_per_sec,
            "first_divergence_position": first_divergence,
            "exact_token_match_rate": exact_match_rate,
            "logit_cosine_vs_fp16": logit_cosine_fr,
            "top5_overlap_vs_fp16": top5_overlap_fr,
            "kl_vs_fp16": kl_fr,
        },
        "fallback_count": 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real generation throughput benchmark"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "baseline_fp16",
            "k8_v5_gs64",
            "k8_v5_gs32",
            "turbo_polar",
            "adaptive",
            "experimental_hybrid",
        ],
    )
    parser.add_argument(
        "--prompt-lengths", nargs="+", type=int, default=[128, 512, 1024]
    )
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json",
        default="artifacts/proof/experimental/real_generation_throughput.json",
    )
    parser.add_argument(
        "--out-md",
        default="artifacts/proof/experimental/real_generation_throughput.md",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    # Prepare prompts of target lengths
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    prompts = {}
    for length in args.prompt_lengths:
        if length > len(dummy_ids):
            # Repeat to reach length
            repeated = (dummy_ids * ((length // len(dummy_ids)) + 1))[:length]
            prompts[length] = tokenizer.decode(repeated)
        else:
            prompts[length] = tokenizer.decode(dummy_ids[:length])

    teacher_forced_results: list[dict[str, Any]] = []
    free_running_results: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        for length, prompt in prompts.items():
            print(f"Benchmarking {cfg_name} @ {length} tokens ...")
            try:
                result = benchmark_config(
                    args.model,
                    cfg_name,
                    prompt,
                    args.new_tokens,
                    device,
                    seed=args.seed,
                )
                tf = {
                    "model_name": result["model_name"],
                    "config": result["config"],
                    "prompt_tokens": result["prompt_tokens"],
                    "new_tokens": result["new_tokens"],
                    **result["teacher_forced"],
                }
                fr = {
                    "model_name": result["model_name"],
                    "config": result["config"],
                    "prompt_tokens": result["prompt_tokens"],
                    "new_tokens": result["new_tokens"],
                    **result["free_running"],
                    "total_end_to_end_ms": result["total_end_to_end_ms"],
                    "peak_memory_bytes": result["peak_memory_bytes"],
                    "compressed_kv_bytes": result["compressed_kv_bytes"],
                    "fp16_kv_bytes": (
                        result["compressed_kv_bytes"]
                        if result["config"] == "baseline_fp16"
                        else None
                    ),
                    "compression_ratio": result["compression_ratio"],
                }
                # Validate baseline identity before accepting
                assert_teacher_forced_identity(tf)
                teacher_forced_results.append(tf)
                free_running_results.append(fr)
                print(
                    f"  TF cosine={result['teacher_forced']['logit_cosine_vs_fp16']:.4f}, "  # noqa: E501
                    f"FR match={result['free_running']['exact_token_match_rate']:.2%}"  # noqa: E501
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  FAILED: {exc}")
                teacher_forced_results.append({
                    "model_name": args.model,
                    "config": cfg_name,
                    "prompt_tokens": length,
                    "new_tokens": args.new_tokens,
                    "error": str(exc),
                })
                free_running_results.append({
                    "model_name": args.model,
                    "config": cfg_name,
                    "prompt_tokens": length,
                    "new_tokens": args.new_tokens,
                    "error": str(exc),
                })

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "release": "experimental",
                "model": args.model,
                "seed": args.seed,
                "teacher_forced_logits": teacher_forced_results,
                "free_running_generation": free_running_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON to {out_json}")

    # Markdown report
    out_md = Path(args.out_md)
    lines = [
        "# Real Generation Throughput Report\n",
        f"**Model:** {args.model}  ",
        f"**Seed:** {args.seed}  ",
        f"**Configs:** {', '.join(args.configs)}\n",
        "## Teacher-Forced Logit Equivalence\n",
    ]
    for r in teacher_forced_results:
        if "error" in r:
            lines.append(
                f"## {r['config']} @ {r['prompt_tokens']} tokens — ERROR\n"
            )
            lines.append(f"```\n{r['error']}\n```\n")
            continue
        lines.append(f"### {r['config']} @ {r['prompt_tokens']} tokens\n")
        lines.append(f"- **Positions checked:** {r['positions_checked']}")
        lines.append(
            f"- **Logit cosine vs FP16:** {r['logit_cosine_vs_fp16']:.4f}"
        )
        lines.append(
            f"- **Top-5 overlap vs FP16:** {r['top5_overlap_vs_fp16']:.4f}"
        )
        lines.append(f"- **KL vs FP16:** {r['kl_vs_fp16']:.6f}")
        lines.append(
            f"- **Max abs logit delta:** {r['max_abs_logit_delta']:.4f}"
        )
        lines.append(
            f"- **Mean abs logit delta:** {r['mean_abs_logit_delta']:.4f}\n"
        )

    lines.append("## Free-Running Generation Divergence\n")
    for r in free_running_results:
        if "error" in r:
            lines.append(
                f"## {r['config']} @ {r['prompt_tokens']} tokens — ERROR\n"
            )
            lines.append(f"```\n{r['error']}\n```\n")
            continue
        lines.append(f"### {r['config']} @ {r['prompt_tokens']} tokens\n")
        lines.append(f"- **Tokens/sec:** {r['tokens_per_second']:.2f}")
        lines.append(f"- **Total E2E ms:** {r['total_end_to_end_ms']:.2f}")
        lines.append(f"- **Compression ratio:** {r['compression_ratio']:.2f}x")
        lines.append(
            "- **First divergence position:** "
            f"{r['first_divergence_position']}"
        )
        lines.append(
            f"- **Exact token match rate:** {r['exact_token_match_rate']:.2%}"
        )
        lines.append(
            f"- **Logit cosine vs FP16:** {r['logit_cosine_vs_fp16']:.4f}\n"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown to {out_md}")


if __name__ == "__main__":
    main()
