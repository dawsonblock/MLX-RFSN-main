#!/usr/bin/env python3
"""Direct KV roundtrip quality validation by prompt length for RFSN v10.

Validates that quantizer roundtrip quality is consistent across context
lengths without involving the full generation loop.

Prompt lengths: 32, 64, 128, 256, 512, 1024
Configs: k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive, experimental_hybrid

Metrics: K cosine, V cosine, K max/MAE, V max/MAE, attention score KL
Output: artifacts/proof/experimental/kv_roundtrip_by_context.json
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

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(functional.cosine_similarity(a_f, b_f, dim=0).item())


def _attention_score_metrics(k_fp16, v_fp16, k_quant, v_quant) -> dict[str, float]:
    """Compute comprehensive attention score metrics between FP16 and quantized.

    Returns raw score MAE/RMSE, stable softmax KL, cosine, and top-k overlap.
    """
    # Use a single query vector (last position of first head)
    dim = k_fp16.shape[-1]
    scale = float(dim) ** -0.5
    q = k_fp16[0, 0, -1:, :].float()

    raw_scores_fp16 = torch.matmul(q, k_fp16[0, 0, :, :].transpose(-2, -1).float()) * scale
    raw_scores_quant = torch.matmul(q, k_quant[0, 0, :, :].transpose(-2, -1).float()) * scale

    # Raw score metrics
    score_diff = (raw_scores_fp16 - raw_scores_quant).abs()
    score_mae = float(score_diff.mean().item())
    score_rmse = float(score_diff.pow(2).mean().sqrt().item())
    score_cosine = float(functional.cosine_similarity(
        raw_scores_fp16.reshape(-1), raw_scores_quant.reshape(-1), dim=0
    ).item())

    # Stable softmax KL
    eps = 1e-8
    p = functional.softmax(raw_scores_fp16, dim=-1)
    q_dist = functional.softmax(raw_scores_quant, dim=-1)
    p_clamped = p.clamp(min=eps)
    q_clamped = q_dist.clamp(min=eps)
    # Renormalize after clamping
    p_clamped = p_clamped / p_clamped.sum(dim=-1, keepdim=True)
    q_clamped = q_clamped / q_clamped.sum(dim=-1, keepdim=True)
    softmax_kl = float(torch.sum(p_clamped * (p_clamped.log() - q_clamped.log())).item())

    # Top-k attention overlap
    seq_len = raw_scores_fp16.shape[-1]
    k_val = min(5, seq_len)
    topk_fp16 = set(torch.topk(raw_scores_fp16, k=k_val, dim=-1).indices[0].tolist())
    topk_quant = set(torch.topk(raw_scores_quant, k=k_val, dim=-1).indices[0].tolist())
    topk_overlap = float(len(topk_fp16 & topk_quant) / len(topk_fp16)) if topk_fp16 else 0.0

    # Query and key norms for context
    q_norm = float(q.norm().item())
    k_norm_fp16 = float(k_fp16[0, 0, :, :].float().norm(dim=-1).mean().item())

    return {
        "score_mae": score_mae,
        "score_rmse": score_rmse,
        "score_cosine": score_cosine,
        "softmax_kl": softmax_kl,
        "topk_attention_overlap": topk_overlap,
        "scale_used": scale,
        "query_norm": q_norm,
        "key_norm_mean": k_norm_fp16,
    }


def _attention_score_kl(k_fp16, v_fp16, k_quant, v_quant) -> float:
    """Compute stable KL between attention score distributions."""
    metrics = _attention_score_metrics(k_fp16, v_fp16, k_quant, v_quant)
    return metrics["softmax_kl"]


def _get_config(name: str) -> dict[str, Any]:
    if name == "k8_v5_gs64":
        return {"name": "k8_v5_gs64", "family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 64}
    if name == "k8_v5_gs32":
        return {"name": "k8_v5_gs32", "family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 32}
    if name == "turbo_polar":
        return {"name": "turbo_polar", "family": "experimental", "mode": "turbo_polar", "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8, "v_bits": 6, "group_size": 64}
    if name == "adaptive":
        return {"name": "adaptive", "family": "experimental", "mode": "turbo_polar", "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8, "v_bits": 6, "group_size": 64, "adaptive_angle_range": True}
    if name == "experimental_hybrid":
        return {"name": "experimental_hybrid", "family": "experimental", "mode": "hybrid_polar_cartesian", "feature_dim": 64, "polar_ratio": 0.65, "polar_levels": 4, "k_angle_bits": 5, "k_radius_bits": 8, "v_angle_bits": 4, "v_radius_bits": 6, "cartesian_bits": 6, "group_size": 64}
    raise ValueError(f"Unknown config: {name}")


def _compress_stable(past_key_values, cfg: dict[str, Any], device: torch.device):
    with tempfile.TemporaryDirectory(prefix="rfsn_rt_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=cfg["k_bits"], v_bits=cfg["v_bits"], group_size=cfg["group_size"],
            use_wht=True, use_incoherent_signs=True, prefer_metal_kernels=True,
            strict_metal=False, max_memory_gb=2.0, cache_dir=tmpdir,
        )
        out = []
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
            out.append((rk, rv))
    return out


def _compress_experimental(past_key_values, cfg: dict[str, Any], device: torch.device):
    mode = cfg.get("mode", "hybrid_polar_cartesian")
    if mode == "turbo_polar":
        mgr = TurboPolarKVManager(
            feature_dim=cfg.get("feature_dim", 64), k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8), v_bits=cfg.get("v_bits", 6),
            group_size=cfg.get("group_size", 64), adaptive_angle_range=cfg.get("adaptive_angle_range", False),
        )
    else:
        mgr = QuantizedKVManager(
            mode="hybrid_polar_cartesian", feature_dim=cfg.get("feature_dim", 64),
            polar_ratio=cfg.get("polar_ratio", 0.65), polar_levels=cfg.get("polar_levels", 4),
            k_angle_bits=cfg.get("k_angle_bits", 5), k_radius_bits=cfg.get("k_radius_bits", 8),
            v_angle_bits=cfg.get("v_angle_bits", 4), v_radius_bits=cfg.get("v_radius_bits", 6),
            cartesian_bits=cfg.get("cartesian_bits", 6), group_size=cfg.get("group_size", 64),
        )
    out = []
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
        out.append((rk, rv))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="KV roundtrip by context")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--configs", nargs="+", default=["k8_v5_gs64", "k8_v5_gs32", "turbo_polar", "adaptive", "experimental_hybrid"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--out", default="artifacts/proof/experimental/kv_roundtrip_by_context.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto" if device.type != "mps" else "mps", trust_remote_code=True,
    )
    model.eval()

    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)

    results: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        cfg = _get_config(cfg_name)
        for length in args.contexts:
            if length > len(dummy_ids):
                repeated = (dummy_ids * ((length // len(dummy_ids)) + 1))[:length]
                prompt = tokenizer.decode(repeated)
            else:
                prompt = tokenizer.decode(dummy_ids[:length])

            prompt_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            prompt_ids = prompt_ids.to(device)

            print(f"Roundtrip {cfg_name} @ {length} tokens ...")
            with torch.no_grad():
                out = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
            past = out.past_key_values

            if cfg["family"] == "stable":
                quant_past = _compress_stable(list(past), cfg, device)
            else:
                quant_past = _compress_experimental(list(past), cfg, device)

            k_cosines = []
            v_cosines = []
            k_max_errs = []
            v_max_errs = []
            k_maes = []
            v_maes = []
            att_kls = []

            for (k_fp16, v_fp16), (k_q, v_q) in zip(past, quant_past):
                k_cosines.append(_cosine(k_fp16, k_q))
                v_cosines.append(_cosine(v_fp16, v_q))
                k_max_errs.append(float((k_fp16 - k_q).abs().max().item()))
                v_max_errs.append(float((v_fp16 - v_q).abs().max().item()))
                k_maes.append(float((k_fp16 - k_q).abs().mean().item()))
                v_maes.append(float((v_fp16 - v_q).abs().mean().item()))
                try:
                    att_kls.append(_attention_score_kl(k_fp16, v_fp16, k_q, v_q))
                except Exception:
                    att_kls.append(float("nan"))

            results.append({
                "config": cfg_name,
                "prompt_tokens": length,
                "k_cosine": float(sum(k_cosines) / len(k_cosines)) if k_cosines else float("nan"),
                "v_cosine": float(sum(v_cosines) / len(v_cosines)) if v_cosines else float("nan"),
                "k_max_error": float(max(k_max_errs)) if k_max_errs else float("nan"),
                "v_max_error": float(max(v_max_errs)) if v_max_errs else float("nan"),
                "k_mean_abs_error": float(sum(k_maes) / len(k_maes)) if k_maes else float("nan"),
                "v_mean_abs_error": float(sum(v_maes) / len(v_maes)) if v_maes else float("nan"),
                "attention_score_kl": float(sum(att_kls) / len(att_kls)) if att_kls else float("nan"),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"model": args.model, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
