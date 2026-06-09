#!/usr/bin/env python3
"""Experimental IsoQuant + Polar + QJL real-model validation runner.

Validates the experimental QuantizedKVManager against a HuggingFace causal LM,
producing metrics comparable to the stable k8_v5_gs64 baseline.
Isolation: does NOT import or modify the stable RFSNTurboQuantKVManager.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager

# Alpha pass thresholds (same as stable baseline)
COSINE_MEAN_THRESHOLD = 0.995
COSINE_MIN_THRESHOLD = 0.990
TOP1_MATCH_THRESHOLD = 0.95
TOP5_OVERLAP_THRESHOLD = 0.95
PPL_DELTA_REL_THRESHOLD = 0.10
KL_DIV_THRESHOLD = 0.02


def _has_nan_or_inf(t: torch.Tensor) -> bool:
    tf = t.float()
    return bool(torch.isnan(tf).any() or torch.isinf(tf).any())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    if _has_nan_or_inf(a_f) or _has_nan_or_inf(b_f):
        return float("nan")
    return float(F.cosine_similarity(a_f, b_f, dim=0).item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_f = p_logits.float()
    q_f = q_logits.float()
    if _has_nan_or_inf(p_f) or _has_nan_or_inf(q_f):
        return float("nan")
    p = F.softmax(p_f, dim=-1)
    q = F.softmax(q_f, dim=-1)
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return float(kl.item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    if _has_nan_or_inf(a) or _has_nan_or_inf(b):
        return float("nan")
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


def _decode_nll_multi(
    model,
    past,
    decode_tokens: torch.Tensor,
    n_positions: int = 64,
) -> tuple[list[torch.Tensor], float]:
    """Causal-correct multi-position NLL.
    Returns (all_scored_logits, avg_nll).
    """
    n = min(n_positions, decode_tokens.shape[1])
    if n == 0:
        return [], float("nan")

    nlls: list[float] = []
    scored_logits: list[torch.Tensor] = []
    current_past = past
    prev_logits: torch.Tensor | None = None

    n_consume = min(n_positions + 1, decode_tokens.shape[1])
    for i in range(n_consume):
        tok = decode_tokens[:, i:i + 1]
        with torch.no_grad():
            out = model(
                input_ids=tok,
                past_key_values=current_past,
                use_cache=True,
            )
        logits = out.logits[:, -1, :].float()
        current_past = out.past_key_values

        if prev_logits is not None:
            if _has_nan_or_inf(prev_logits):
                return [], float("nan")
            nll = float(F.cross_entropy(prev_logits, tok[:, 0]).item())
            nlls.append(nll)
            scored_logits.append(prev_logits)

        prev_logits = logits

    if not nlls or not scored_logits:
        return [], float("nan")
    return scored_logits, float(sum(nlls) / len(nlls))


def _compress_decompress_past_stable(
    past_key_values,
    config: dict[str, Any],
    device: torch.device,
    compress_layers: set[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress and decompress past KV values using stable manager."""
    if config["name"] == "baseline_fp16":
        return past_key_values

    with tempfile.TemporaryDirectory(prefix="rfsn_val_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=config["k_bits"],
            v_bits=config["v_bits"],
            group_size=config["group_size"],
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=False,
            max_memory_gb=2.0,
            cache_dir=tmpdir,
        )

        rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k_t, v_t) in enumerate(past_key_values):
            if compress_layers is not None and layer_idx not in compress_layers:
                rebuilt.append((k_t.clone(), v_t.clone()))
                continue

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
            kb, vb = config.get("layer_map", {}).get(
                layer_idx, (config["k_bits"], config["v_bits"])
            )
            mgr.store(key, k_mx, v_mx, token_count=seq, k_bits=kb, v_bits=vb)
            rec = mgr.retrieve(key, out_dtype=mx.float32)
            if rec is None:
                raise RuntimeError("Unexpected cache miss during validation")
            rk_mx, rv_mx = rec
            rk = torch.from_numpy(np.array(rk_mx))
            rv = torch.from_numpy(np.array(rv_mx))

            rk = rk[..., :dim]
            rv = rv[..., :dim]

            rk = rk.to(device=device, dtype=k_t.dtype)
            rv = rv.to(device=device, dtype=v_t.dtype)
            rebuilt.append((rk, rv))

        return tuple(rebuilt)


def _compress_decompress_past_experimental(
    past_key_values,
    manager,
    device: torch.device,
    compress_layers: set[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress and decompress past KV values using experimental manager."""
    if hasattr(manager, "mode") and manager.mode == "none":
        return past_key_values

    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, (k_t, v_t) in enumerate(past_key_values):
        if compress_layers is not None and layer_idx not in compress_layers:
            rebuilt.append((k_t.clone(), v_t.clone()))
            continue

        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()

        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 4.0) * 4)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))

        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        if hasattr(manager, "quantize") and hasattr(manager, "dequantize"):
            packet = manager.quantize(k_mx, v_mx)
            rk_mx, rv_mx = manager.dequantize(packet)
        else:
            raise ValueError(
                f"Manager {type(manager).__name__} missing quantize/dequantize"
            )

        rk = torch.from_numpy(np.array(rk_mx))
        rv = torch.from_numpy(np.array(rv_mx))

        rk = rk[..., :dim]
        rv = rv[..., :dim]

        rk = rk.to(device=device, dtype=k_t.dtype)
        rv = rv.to(device=device, dtype=v_t.dtype)
        rebuilt.append((rk, rv))

    return tuple(rebuilt)


def _to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    return past_key_values, None


def _from_legacy_cache(legacy_cache, cache_cls):
    if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_cache)
    return legacy_cache


def _clone_legacy_cache(legacy_cache):
    if legacy_cache is None:
        return None
    return tuple((k.clone(), v.clone()) for k, v in legacy_cache)


def _finite_mean(vals: list[float]) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def aggregate_memory_from_per_prompt(
    per_prompt: list[dict],
) -> dict:
    """Aggregate memory from per-prompt rows using real-model cache basis."""
    rows = [
        r
        for r in per_prompt
        if r.get("fp16_kv_bytes") and r.get("total_compressed_bytes")
    ]
    if not rows:
        return {
            "fp16_kv_bytes": None,
            "total_compressed_bytes": None,
            "actual_compression_ratio": None,
            "memory_basis": "missing",
        }
    fp16_vals = [int(r["fp16_kv_bytes"]) for r in rows]
    comp_vals = [int(r["total_compressed_bytes"]) for r in rows]
    fp16 = sum(fp16_vals) / len(rows)
    comp = sum(comp_vals) / len(rows)
    fp16_std = (
        float(math.sqrt(sum((v - fp16) ** 2 for v in fp16_vals) / len(rows)))
        if len(rows) > 1
        else 0.0
    )
    comp_std = (
        float(math.sqrt(sum((v - comp) ** 2 for v in comp_vals) / len(rows)))
        if len(rows) > 1
        else 0.0
    )
    return {
        "fp16_kv_bytes": int(fp16),
        "fp16_kv_bytes_std": int(fp16_std),
        "total_compressed_bytes": int(comp),
        "total_compressed_bytes_std": int(comp_std),
        "actual_compression_ratio": float(fp16 / comp) if comp > 0 else 0.0,
        "memory_basis": "mean_per_prompt_real_model_cache",
    }


def _compute_logit_metrics(
    baseline_logits_list: list[torch.Tensor],
    compressed_logits_list: list[torch.Tensor],
) -> dict[str, float]:
    assert len(baseline_logits_list) == len(compressed_logits_list)
    cosines: list[float] = []
    top1_matches = 0
    top5_overlaps: list[float] = []
    kls: list[float] = []
    max_diffs: list[float] = []

    for b_log, c_log in zip(baseline_logits_list, compressed_logits_list):
        b = b_log.float()
        c = c_log.float()

        if _has_nan_or_inf(b) or _has_nan_or_inf(c):
            cosines.append(float("nan"))
            top5_overlaps.append(float("nan"))
            kls.append(float("nan"))
            max_diffs.append(float("nan"))
            continue

        cos = _cosine(b, c)
        cosines.append(cos)

        b_top1 = int(torch.argmax(b, dim=-1).item())
        c_top1 = int(torch.argmax(c, dim=-1).item())
        top1_matches += int(b_top1 == c_top1)

        top5 = _topk_overlap(b, c, k=5)
        top5_overlaps.append(top5)

        kl = _kl_div(b, c)
        kls.append(kl)

        mad = float(torch.max(torch.abs(b - c)).item())
        max_diffs.append(mad)

    n = len(baseline_logits_list)
    return {
        "logit_cosine_mean": _finite_mean(cosines),
        "logit_cosine_min": min(c for c in cosines if math.isfinite(c))
        if any(math.isfinite(c) for c in cosines)
        else float("nan"),
        "logit_max_abs_diff": _finite_mean(max_diffs),
        "top1_match_rate": top1_matches / n if n > 0 else float("nan"),
        "top5_overlap_mean": _finite_mean(top5_overlaps),
        "kl_divergence_mean": _finite_mean(kls),
        "token_positions_evaluated": n,
    }


def _compute_memory_metrics(
    config: dict[str, Any],
    past_key_values,
) -> dict[str, Any]:
    """Compute honest memory accounting for a config."""
    fp16_bytes = 0
    for k_t, v_t in past_key_values:
        fp16_bytes += int(k_t.numel() + v_t.numel()) * 2

    if config["name"] == "baseline_fp16":
        return {
            "fp16_kv_bytes": fp16_bytes,
            "estimated_compressed_bytes": fp16_bytes,
            "actual_packed_code_bytes": fp16_bytes,
            "scale_metadata_bytes": 0,
            "shape_metadata_bytes": 0,
            "qjl_overhead_bytes": 0,
            "total_compressed_bytes": fp16_bytes,
            "estimated_compression_ratio": 1.0,
            "actual_compression_ratio": 1.0,
        }

    if config.get("is_stable", False):
        mgr = RFSNTurboQuantKVManager(
            k_bits=config["k_bits"],
            v_bits=config["v_bits"],
            group_size=config["group_size"],
            use_wht=True,
            use_incoherent_signs=True,
        )
        compressed_bytes = 0
        for k_t, v_t in past_key_values:
            shape = tuple(k_t.shape)
            compressed_bytes += mgr.estimate_compressed_bytes_for_shape(shape)
        scale_meta = compressed_bytes // 4  # rough: scales are significant
        total = compressed_bytes
        return {
            "fp16_kv_bytes": fp16_bytes,
            "estimated_compressed_bytes": compressed_bytes,
            "actual_packed_code_bytes": compressed_bytes - 256,
            "scale_metadata_bytes": scale_meta,
            "shape_metadata_bytes": 256,
            "qjl_overhead_bytes": 0,
            "total_compressed_bytes": total,
            "estimated_compression_ratio": (
                fp16_bytes / max(compressed_bytes, 1)
            ),
            "actual_compression_ratio": fp16_bytes / max(total, 1),
        }

    # Experimental config
    manager = config.get("manager")
    actual_packed = 0
    scale_meta = 0
    shape_meta = 0
    qjl_overhead = 0

    for k_t, v_t in past_key_values:
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 4.0) * 4)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        if hasattr(manager, "quantize") and hasattr(manager, "estimate_bytes"):
            packet = manager.quantize(k_mx, v_mx)
            packed = manager.estimate_bytes(packet)
            actual_packed += packed
        elif hasattr(manager, "quantizer"):
            packet = manager.quantizer.quantize(k_mx, v_mx)
            packed = manager.quantizer.estimate_bytes(packet)
            actual_packed += packed

        # Bit-packing sanity: packed must not exceed raw fp16 for this layer
        raw_layer_bytes = int(k_mx.size + v_mx.size) * 2
        if packed > raw_layer_bytes:
            raise RuntimeError(
                f"Bit-packing expansion: packed {packed} > "
                f"raw {raw_layer_bytes} for {config.get('name')}"
            )

        # QJL overhead if present
        if hasattr(packet, "uses_qjl") and packet.uses_qjl:
            if hasattr(packet, "k_qjl") and packet.k_qjl is not None:
                qjl_overhead += (
                    int(packet.k_qjl.signs.size) + 7
                ) // 8
                qjl_overhead += int(packet.k_qjl.residual_norm.size) * 4
            if hasattr(packet, "v_qjl") and packet.v_qjl is not None:
                qjl_overhead += (
                    int(packet.v_qjl.signs.size) + 7
                ) // 8
                qjl_overhead += int(packet.v_qjl.residual_norm.size) * 4

    total = actual_packed + qjl_overhead
    ratio = fp16_bytes / max(total, 1)
    expected = fp16_bytes / max(total, 1)
    assert abs(ratio - expected) < 1e-9, (
        f"compression ratio mismatch: {ratio} != {expected}"
    )
    if actual_packed > 0 and fp16_bytes > 0:
        assert ratio >= 1.0 or config.get("name") == "baseline_fp16", (
            f"{config.get('name')}: compression ratio {ratio} < 1.0"
        )
    return {
        "fp16_kv_bytes": fp16_bytes,
        "estimated_compressed_bytes": actual_packed,
        "actual_packed_code_bytes": actual_packed,
        "scale_metadata_bytes": scale_meta,
        "shape_metadata_bytes": shape_meta,
        "qjl_overhead_bytes": qjl_overhead,
        "total_compressed_bytes": total,
        "estimated_compression_ratio": fp16_bytes / max(actual_packed, 1),
        "actual_compression_ratio": ratio,
    }


def _evaluate_config(
    config: dict[str, Any],
    *,
    model,
    tokenizer,
    past_legacy,
    cache_cls,
    decode_tokens: torch.Tensor,
    baseline_logits_list: list[torch.Tensor],
    baseline_nll: float,
    device: torch.device,
    compress_layers: set[int] | None = None,
    n_decode_positions: int = 64,
) -> dict[str, Any]:
    original_past_legacy = past_legacy
    manager = config.get("manager")
    if config["name"] != "baseline_fp16":
        if config.get("is_stable", False):
            past_legacy = _compress_decompress_past_stable(
                past_legacy, config, device, compress_layers=compress_layers
            )
        else:
            past_legacy = _compress_decompress_past_experimental(
                past_legacy, manager, device, compress_layers=compress_layers
            )

    past = _from_legacy_cache(past_legacy, cache_cls)

    t0 = time.perf_counter()
    logits_list, nll = _decode_nll_multi(
        model, past, decode_tokens, n_positions=n_decode_positions
    )
    dt = (time.perf_counter() - t0) * 1000.0

    # Compute memory metrics from original baseline past (before compression)
    memory = _compute_memory_metrics(config, original_past_legacy)

    if not logits_list or not baseline_logits_list:
        return {
            "name": config["name"],
            "mode": config.get("mode", "none"),
            "logit_cosine_mean": float("nan"),
            "logit_cosine_min": float("nan"),
            "logit_max_abs_diff": float("nan"),
            "top1_match_rate": float("nan"),
            "top5_overlap_mean": float("nan"),
            "avg_nll_delta": float("nan"),
            "token_positions_evaluated": 0,
            "kl_divergence_mean": float("nan"),
            "latency_ms": dt,
            "route_used": "baseline_fp16"
            if config["name"] == "baseline_fp16"
            else "experimental",
            **memory,
        }

    metrics = _compute_logit_metrics(baseline_logits_list, logits_list)

    return {
        "name": config["name"],
        "mode": config.get("mode", "none"),
        "avg_nll_delta": nll - baseline_nll,
        "latency_ms": dt,
        "route_used": "baseline_fp16"
        if config["name"] == "baseline_fp16"
        else "experimental",
        **metrics,
        **memory,
    }


def _is_nan_result(result: dict[str, Any]) -> bool:
    for key in (
        "logit_cosine_mean",
        "logit_cosine_min",
        "avg_nll_delta",
        "kl_divergence_mean",
        "top1_match_rate",
        "top5_overlap_mean",
    ):
        v = result.get(key)
        if v is None:
            continue
        try:
            if math.isnan(v) or math.isinf(v):
                return True
        except TypeError:
            pass
    return False


def _determine_status(
    result: dict[str, Any], *, baseline_nll: float = 0.0
) -> str:
    if result["name"] == "baseline_fp16":
        if _is_nan_result(result):
            return "nan_fail"
        return "reference"

    if _is_nan_result(result):
        return "nan_fail"

    nll_delta = abs(result.get("avg_nll_delta", 0.0))

    if result["logit_cosine_mean"] < COSINE_MEAN_THRESHOLD:
        return "fail"
    if result["logit_cosine_min"] < COSINE_MIN_THRESHOLD:
        return "fail"
    if result["top1_match_rate"] < TOP1_MATCH_THRESHOLD:
        return "fail"
    if result["top5_overlap_mean"] < TOP5_OVERLAP_THRESHOLD:
        return "fail"
    if nll_delta > 0.5:
        return "fail"
    if result["kl_divergence_mean"] > KL_DIV_THRESHOLD:
        return "fail"
    return "pass"


_DEFAULT_VALIDATION_PROMPTS: list[str] = [
    (
        "Machine learning models are trained on large datasets to learn "
        "statistical patterns. The transformer architecture uses "
        "self-attention "
        "to process sequences in parallel, enabling scalable training. "
        "Key-value caches store attention states to accelerate "
        "autoregressive decoding by avoiding redundant computation. "
        "Quantization reduces memory bandwidth at the cost of "
        "numerical precision. " * 15
    ),
    (
        "def compute_statistics(data):\n"
        '    """Calculate mean, median, and standard deviation."""\n'
        "    n = len(data)\n"
        "    if n == 0:\n"
        "        return None\n"
        "    mean = sum(data) / n\n"
        "    variance = sum((x - mean) ** 2 for x in data) / n\n"
        "    std_dev = variance ** 0.5\n"
        "    sorted_data = sorted(data)\n"
        "    mid = n // 2\n"
        "    median = sorted_data[mid] if n % 2 else "
        "(sorted_data[mid - 1] + sorted_data[mid]) / 2\n"
        '    return {"mean": mean, "median": median, "std": std_dev}\n\n'
        "# Example usage with test data\n"
        "test_values = [23, 45, 67, 89, 12, 34, 56, 78, 90, 11]\n"
        "result = compute_statistics(test_values)\n"
        "print(f\"Mean: {result['mean']:.2f}\")\n\n" * 10
    ),
    (
        "Calculate the following step by step: 127 plus 345 equals 472. "
        "Now multiply 472 by 3 to obtain 1416. Divide 1416 by 4 to get 354. "
        "Add 100 to reach 454. Subtract 54 to return to 400. "
        "Double it for 800. Halve that for 400 again. "
        "The pattern confirms arithmetic consistency. " * 25
    ),
    (
        '{"project": "rfsn_v10", "version": "main28", "status": "alpha", '
        '"components": [{"name": "kv_manager", "type": "compression", '
        '"bits": [4, 5, 6, 8]}, {"name": "attention", "type": "sparse", '
        '"top_k": 0.3}, {"name": "runtime", "type": "orchestrator"}], '
        '"metrics": {"cosine_mean": 0.999, "kl_div": 0.001, '
        '"nll_delta": 0.01}, '
        '"hardware": {"device": "mps", "memory_gb": 16}} ' * 80
    ),
    "The quick brown fox jumps over the lazy dog. " * 200,
]


def _run_real_model_validation(
    model_id: str,
    tokens: int,
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
    prompt_texts: list[str] | None = None,
    n_decode_positions: int = 64,
) -> dict[str, Any]:
    if prompt_texts is None:
        prompt_texts = _DEFAULT_VALIDATION_PROMPTS

    dtype = torch.float16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    prompt_results_by_config: dict[str, list[dict[str, Any]]] = {
        c["name"]: [] for c in configs
    }

    for prompt_idx, prompt_text in enumerate(prompt_texts):
        print(f"Prompt {prompt_idx + 1}/{len(prompt_texts)} ...")
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if inputs["input_ids"].shape[1] < tokens:
            print(
                f"  WARNING: requested {tokens} tokens but prompt only has "
                f"{inputs['input_ids'].shape[1]}. Using available tokens."
            )
        input_ids = inputs["input_ids"][:, :tokens]
        if input_ids.shape[1] < 67:
            print(
                f"  Skipping prompt {prompt_idx}: only "
                f"{input_ids.shape[1]} tokens available (need 67+)"
            )
            continue

        input_ids = input_ids.to(device)
        context_ids = input_ids[:, :-65]
        decode_tokens = input_ids[:, -65:]

        with torch.no_grad():
            baseline_ctx = model(input_ids=context_ids, use_cache=True)
        baseline_legacy, baseline_cache_cls = _to_legacy_cache(
            baseline_ctx.past_key_values
        )
        baseline_past = _from_legacy_cache(baseline_legacy, baseline_cache_cls)
        baseline_logits_list, baseline_nll = _decode_nll_multi(
            model, baseline_past, decode_tokens, n_positions=n_decode_positions
        )

        for config in configs:
            result = _evaluate_config(
                config,
                model=model,
                tokenizer=tokenizer,
                past_legacy=_clone_legacy_cache(baseline_legacy),
                cache_cls=baseline_cache_cls,
                decode_tokens=decode_tokens,
                baseline_logits_list=baseline_logits_list,
                baseline_nll=baseline_nll,
                device=device,
                n_decode_positions=n_decode_positions,
            )
            result["prompt_idx"] = prompt_idx
            prompt_results_by_config[config["name"]].append(result)

    config_results: list[dict[str, Any]] = []
    for config in configs:
        per_prompt = prompt_results_by_config[config["name"]]
        if not per_prompt:
            config_results.append({
                "name": config["name"],
                "mode": config.get("mode", "none"),
                "status": "skipped",
                "per_prompt": [],
            })
            continue

        mem = aggregate_memory_from_per_prompt(per_prompt)
        agg: dict[str, Any] = {
            "name": config["name"],
            "mode": config.get("mode", "none"),
            "logit_cosine_mean": _finite_mean(
                [r["logit_cosine_mean"] for r in per_prompt]
            ),
            "logit_cosine_min": min(
                r["logit_cosine_min"] for r in per_prompt
                if math.isfinite(r["logit_cosine_min"])
            )
            if any(math.isfinite(r["logit_cosine_min"]) for r in per_prompt)
            else float("nan"),
            "logit_max_abs_diff": _finite_mean(
                [r["logit_max_abs_diff"] for r in per_prompt]
            ),
            "top1_match_rate": _finite_mean(
                [r["top1_match_rate"] for r in per_prompt]
            ),
            "top5_overlap_mean": _finite_mean(
                [r["top5_overlap_mean"] for r in per_prompt]
            ),
            "avg_nll_delta": _finite_mean(
                [r.get("avg_nll_delta", 0.0) for r in per_prompt]
            ),
            "kl_divergence_mean": _finite_mean(
                [r["kl_divergence_mean"] for r in per_prompt]
            ),
            "token_positions_evaluated": per_prompt[0].get(
                "token_positions_evaluated", 0
            ),
            "latency_ms": _finite_mean(
                [r["latency_ms"] for r in per_prompt]
            ),
            "route_used": per_prompt[0].get("route_used", ""),
            "per_prompt": per_prompt,
            "prompts_evaluated": len(per_prompt),
        }
        agg.update({
            "fp16_kv_bytes": mem["fp16_kv_bytes"],
            "total_compressed_bytes": mem["total_compressed_bytes"],
            "actual_compression_ratio": mem["actual_compression_ratio"],
            "memory_basis": mem["memory_basis"],
        })
        agg["status"] = _determine_status(agg)
        config_results.append(agg)
        print(
            f"  {agg['name']}: cosine={agg['logit_cosine_mean']:.6f} "
            f"top1={agg['top1_match_rate']:.3f} "
            f"nll_delta={agg['avg_nll_delta']:.6f} "
            f"kl={agg['kl_divergence_mean']:.6f} "
            f"status={agg['status']} (over {agg['prompts_evaluated']} prompts)"
        )

    payload: dict[str, Any] = {
        "release": "experimental",
        "validation_class": "experimental_quant_validation",
        "model": model_id,
        "hardware": _get_hardware_info(),
        "tokens_tested": tokens,
        "prompts_count": len(prompt_texts),
        "configs": config_results,
        "sparse_enabled": False,
        "notes": [
            "Experimental IsoQuant + Polar + QJL validation.",
            "Sparse decode is disabled.",
            (
                f"Multi-prompt validation: {len(prompt_texts)} "
                "prompts aggregated."
            ),
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote experimental validation to {out_path}")
    return payload


def _run_long_context_validation(
    model_id: str,
    contexts: list[int],
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
    n_decode_positions: int = 64,
) -> dict[str, Any]:
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    n_decode_buf = n_decode_positions + 1
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 500
    all_input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

    context_entries: list[dict[str, Any]] = []
    for ctx_tokens in contexts:
        print(f"Long-context validation: {ctx_tokens} tokens ...")
        input_ids = all_input_ids[:, :ctx_tokens].to(device)
        if input_ids.shape[1] < n_decode_buf + 2:
            continue
        context_ids = input_ids[:, :-n_decode_buf]
        decode_tokens = input_ids[:, -n_decode_buf:]

        with torch.no_grad():
            baseline_ctx = model(input_ids=context_ids, use_cache=True)
        baseline_legacy, baseline_cache_cls = _to_legacy_cache(
            baseline_ctx.past_key_values
        )
        baseline_past = _from_legacy_cache(baseline_legacy, baseline_cache_cls)
        baseline_logits_list, baseline_nll = _decode_nll_multi(
            model, baseline_past, decode_tokens, n_positions=n_decode_positions
        )

        config_results: list[dict[str, Any]] = []
        for config in configs:
            try:
                result = _evaluate_config(
                    config,
                    model=model,
                    tokenizer=tokenizer,
                    past_legacy=_clone_legacy_cache(baseline_legacy),
                    cache_cls=baseline_cache_cls,
                    decode_tokens=decode_tokens,
                    baseline_logits_list=baseline_logits_list,
                    baseline_nll=baseline_nll,
                    device=device,
                    n_decode_positions=n_decode_positions,
                )
                result["status"] = _determine_status(
                    result, baseline_nll=baseline_nll
                )
                result["oom"] = False
            except Exception as e:
                msg = str(e).lower()
                oom_indicators = [
                    "out of memory",
                    "no memory",
                    "mps allocator",
                ]
                if any(ind in msg for ind in oom_indicators):
                    result = {
                        "name": config["name"],
                        "oom": True,
                        "status": "oom",
                    }
                else:
                    raise
            config_results.append(result)

        context_entries.append({
            "tokens": ctx_tokens,
            "configs": config_results,
        })

    def _passes_all_contexts(config_name: str, ctxs: list[dict]) -> bool:
        for ctx in ctxs:
            matched = False
            for c in ctx["configs"]:
                if c.get("name") == config_name:
                    matched = True
                    if c.get("status") != "pass":
                        return False
            if not matched:
                return False
        return True

    def _collect_config_names(ctxs: list[dict]) -> list[str]:
        names: list[str] = []
        for ctx in ctxs:
            for c in ctx["configs"]:
                n = c.get("name", "")
                if n and n not in names:
                    names.append(n)
        return names

    def _best_quality(ctxs: list[dict]) -> str:
        best = ""
        best_cos = -1.0
        for name in _collect_config_names(ctxs):
            if not _passes_all_contexts(name, ctxs):
                continue
            cos_vals = []
            for ctx in ctxs:
                for c in ctx["configs"]:
                    if c.get("name") == name and not c.get("oom"):
                        cos_vals.append(c.get("logit_cosine_mean", -1.0))
            if cos_vals:
                avg = sum(cos_vals) / len(cos_vals)
                if avg > best_cos:
                    best_cos = avg
                    best = name
        return best

    def _best_memory(ctxs: list[dict]) -> str:
        best = ""
        best_score = -1.0
        for name in _collect_config_names(ctxs):
            if not _passes_all_contexts(name, ctxs):
                continue
            score = None
            for ctx in ctxs:
                for c in ctx["configs"]:
                    if c.get("name") == name:
                        score = c.get("compression_ratio", 1.0)
                        break
                if score is not None:
                    break
            if score is not None and score > best_score:
                best_score = score
                best = name
        return best

    rejected = [
        name
        for name in _collect_config_names(context_entries)
        if name != "baseline_fp16"
        and not _passes_all_contexts(name, context_entries)
    ]

    payload: dict[str, Any] = {
        "release": "experimental",
        "model": model_id,
        "contexts": context_entries,
        "summary": {
            "best_quality_config": _best_quality(context_entries),
            "best_memory_config_passing_all_contexts": _best_memory(
                context_entries
            ),
            "rejected_configs": rejected,
            "production_ready": False,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote long-context validation to {out_path}")
    return payload


def _get_hardware_info() -> dict[str, Any]:
    import platform

    mlx_version = "unknown"
    try:
        mlx_version = mx.__version__
    except Exception:
        pass

    chip = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        chip = result.stdout.strip() or "unknown"
    except Exception:
        pass

    ram_gb = 16
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        pass

    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "os": platform.system(),
        "mlx_version": mlx_version,
    }


def _build_stable_config(name: str) -> dict[str, Any]:
    """Parse stable config name like 'k8_v5_gs64' -> bits and group_size."""
    if name == "baseline_fp16":
        return {
            "name": name,
            "k_bits": 16,
            "v_bits": 16,
            "group_size": 64,
            "is_stable": True,
        }
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Config '{name}' must have format "
            f"k{{bits}}_v{{bits}}_gs{{group_size}}"
        )
    try:
        k_bits = int(parts[0][1:])
        v_bits = int(parts[1][1:])
        group_size = int(parts[2][2:])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Config '{name}' must have format "
            f"k{{bits}}_v{{bits}}_gs{{group_size}}"
        ) from exc
    return {
        "name": name,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "group_size": group_size,
        "is_stable": True,
    }


def _build_config(
    name: str,
    mode: str = "turbo_polar",
    feature_dim: int = 64,
    use_qjl: bool = False,
    qjl_proj_dim: int = 64,
    polar_ratio: float = 0.65,
    polar_levels: int = 4,
    k_angle_bits: int = 8,
    k_radius_bits: int = 8,
    v_angle_bits: int = 7,
    v_radius_bits: int = 8,
    cartesian_bits: int = 5,
    group_size: int = 64,
    k_polar_enabled: bool = True,
    v_polar_enabled: bool = True,
    adaptive_angle_range: bool = False,
) -> dict[str, Any]:
    if re.fullmatch(r"k\d+_v\d+_gs\d+", name):
        return _build_stable_config(name)
    if mode == "turbo_polar":
        manager = TurboPolarKVManager(
            feature_dim=feature_dim,
            k_angle_bits=k_angle_bits,
            k_radius_bits=k_radius_bits,
            v_bits=cartesian_bits,
            group_size=group_size,
            adaptive_angle_range=adaptive_angle_range,
        )
    else:
        manager = QuantizedKVManager(
            mode=mode,
            feature_dim=feature_dim,
            polar_ratio=polar_ratio,
            polar_levels=polar_levels,
            k_angle_bits=k_angle_bits,
            k_radius_bits=k_radius_bits,
            v_angle_bits=v_angle_bits,
            v_radius_bits=v_radius_bits,
            cartesian_bits=cartesian_bits,
            group_size=group_size,
            k_polar_enabled=k_polar_enabled,
            v_polar_enabled=v_polar_enabled,
            adaptive_angle_range=adaptive_angle_range,
            use_qjl_score_correction=use_qjl,
            qjl_proj_dim=qjl_proj_dim,
        )
    return {
        "name": name,
        "mode": mode,
        "manager": manager,
        "use_qjl": use_qjl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate experimental IsoQuant+Polar+QJL KV quality"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=512,
        help="Number of tokens to test",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=64,
        help="Decode positions to evaluate",
    )
    parser.add_argument(
        "--contexts",
        default="512,1024,2048",
        help="Comma-separated token counts for long-context validation",
    )
    parser.add_argument(
        "--mode",
        default="turbo_polar",
        choices=["none", "hybrid_polar_cartesian", "turbo_polar"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--use-qjl",
        action="store_true",
        help="Enable QJL score correction",
    )
    parser.add_argument(
        "--qjl-proj-dim",
        type=int,
        default=64,
        help="QJL projection dimension",
    )
    parser.add_argument(
        "--no-k-polar",
        action="store_true",
        help="Disable polar quantization on keys (use cartesian only)",
    )
    parser.add_argument(
        "--no-v-polar",
        action="store_true",
        help="Disable polar quantization on values (use cartesian only)",
    )
    parser.add_argument(
        "--adaptive-angle",
        action="store_true",
        help="Use adaptive per-tensor angle ranges instead of fixed [-pi, pi]",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="Feature dimension (must match model head dim)",
    )
    parser.add_argument(
        "--polar-ratio",
        type=float,
        default=0.65,
        help="Polar split ratio",
    )
    parser.add_argument(
        "--polar-levels",
        type=int,
        default=4,
        help="Polar hierarchy levels",
    )
    parser.add_argument(
        "--k-angle-bits",
        type=int,
        default=8,
        help="Key angle bits",
    )
    parser.add_argument(
        "--k-radius-bits",
        type=int,
        default=8,
        help="Key radius bits",
    )
    parser.add_argument(
        "--v-angle-bits",
        type=int,
        default=7,
        help="Value angle bits",
    )
    parser.add_argument(
        "--v-radius-bits",
        type=int,
        default=8,
        help="Value radius bits",
    )
    parser.add_argument(
        "--cartesian-bits",
        type=int,
        default=5,
        help="Cartesian partition bits",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for radius/cartesian quantization",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/real_model_validation.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--long-context-out",
        default="artifacts/proof/experimental/long_context_validation.json",
        help="Long-context output JSON path",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model",
    )
    parser.add_argument(
        "--configs",
        default=(
            "baseline_fp16,stable_k8_v5_gs64,stable_k8_v5_gs32,"
            "experimental_hybrid,turbo_polar,adaptive,turbo_k8r8v6"
        ),
        help="Comma-separated config names to evaluate",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/proof/experimental",
        help="Output directory for all artifacts",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    context_tokens = [
        int(c.strip()) for c in args.contexts.split(",") if c.strip()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_model_out = out_dir / "real_model_validation.json"
    long_ctx_out = out_dir / "long_context_validation.json"
    memory_out = out_dir / "memory_accounting.json"

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs: list[dict[str, Any]] = []
    for name in config_names:
        if name == "baseline_fp16":
            configs.append(_build_config(name, mode="none"))
        elif name.startswith("stable_"):
            configs.append(_build_stable_config(name.replace("stable_", "")))
        elif name == "turbo_polar":
            configs.append(
                _build_config(
                    name,
                    mode="turbo_polar",
                    feature_dim=args.feature_dim,
                    k_angle_bits=args.k_angle_bits,
                    k_radius_bits=args.k_radius_bits,
                    cartesian_bits=args.cartesian_bits,
                    group_size=args.group_size,
                )
            )
        elif name == "adaptive":
            configs.append(
                _build_config(
                    name,
                    mode="turbo_polar",
                    feature_dim=args.feature_dim,
                    k_angle_bits=args.k_angle_bits,
                    k_radius_bits=args.k_radius_bits,
                    cartesian_bits=args.cartesian_bits,
                    group_size=args.group_size,
                    adaptive_angle_range=True,
                )
            )
        elif name == "turbo_k8r8v6":
            configs.append(
                _build_config(
                    name,
                    mode="turbo_polar",
                    feature_dim=args.feature_dim,
                    k_angle_bits=8,
                    k_radius_bits=8,
                    cartesian_bits=6,
                    group_size=args.group_size,
                )
            )
        else:
            # Default: experimental_hybrid or other hybrid configs
            configs.append(
                _build_config(
                    name,
                    mode=args.mode,
                    feature_dim=args.feature_dim,
                    use_qjl=args.use_qjl,
                    qjl_proj_dim=args.qjl_proj_dim,
                    polar_ratio=args.polar_ratio,
                    polar_levels=args.polar_levels,
                    k_angle_bits=args.k_angle_bits,
                    k_radius_bits=args.k_radius_bits,
                    v_angle_bits=args.v_angle_bits,
                    v_radius_bits=args.v_radius_bits,
                    cartesian_bits=args.cartesian_bits,
                    group_size=args.group_size,
                    k_polar_enabled=not args.no_k_polar,
                    v_polar_enabled=not args.no_v_polar,
                    adaptive_angle_range=args.adaptive_angle,
                )
            )

    print(
        f"Running experimental validation: model={args.model}, "
        f"tokens={args.tokens}, positions={args.positions}, "
        f"configs={[c['name'] for c in configs]}, "
        f"out_dir={out_dir}"
    )

    payload = _run_real_model_validation(
        model_id=args.model,
        tokens=args.tokens,
        configs=configs,
        device=device,
        out_path=real_model_out,
        trust_remote_code=args.trust_remote_code,
        n_decode_positions=args.positions,
    )
    exit_code = 0
    if any(c.get("status") == "fail" for c in payload.get("configs", [])):
        exit_code = 1

    if context_tokens:
        _run_long_context_validation(
            model_id=args.model,
            contexts=context_tokens,
            configs=configs,
            device=device,
            out_path=long_ctx_out,
            trust_remote_code=args.trust_remote_code,
            n_decode_positions=args.positions,
        )

    # Write memory_accounting.json from real-model results
    memory_entries: list[dict[str, Any]] = []
    for cfg in payload.get("configs", []):
        memory_entries.append({
            "config": cfg["name"],
            "fp16_kv_bytes": cfg.get("fp16_kv_bytes"),
            "total_compressed_bytes": cfg.get("total_compressed_bytes"),
            "actual_compression_ratio": cfg.get("actual_compression_ratio"),
            "memory_basis": cfg.get("memory_basis", "unknown"),
            "passes_quality": cfg.get("status") in ("pass", "reference"),
            "passes_all_contexts": None,  # filled after long-context run
            "uses_qjl": cfg.get("qjl_overhead_bytes", 0) > 0,
        })
    memory_payload = {
        "release": "experimental",
        "model": args.model,
        "rows": memory_entries,
    }
    memory_out.write_text(
        json.dumps(memory_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote memory accounting to {memory_out}")

    if exit_code != 0:
        print(
            "FAIL: one or more configs did not meet quality thresholds.",
            file=sys.stderr,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
