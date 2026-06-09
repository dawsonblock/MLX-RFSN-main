#!/usr/bin/env python3
"""Short-prompt focused drift debug benchmark for RFSN v10.

Runs baseline_fp16, k8_v5_gs64, and k8_v5_gs32 across short prompts
and records step-by-step logit drift to isolate where corruption starts.

Prompt lengths: 32, 64, 128, 256, 512
New tokens: 1, 8, 32, 128

Output: artifacts/proof/experimental/short_prompt_drift_trace.json
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    if torch.isnan(a_f).any() or torch.isinf(a_f).any():
        return float("nan")
    if torch.isnan(b_f).any() or torch.isinf(b_f).any():
        return float("nan")
    return float(functional.cosine_similarity(a_f, b_f, dim=0).item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> float:
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = functional.softmax(p_logits.float(), dim=-1)
    q = functional.softmax(q_logits.float(), dim=-1)
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return float(kl.item())


def _compress_past(past_key_values, cfg: dict[str, Any], device: torch.device):
    if cfg["name"] == "baseline_fp16":
        return list(past_key_values)

    compressed = []
    with tempfile.TemporaryDirectory(prefix="rfsn_short_") as tmpdir:
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


def _trace_config(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    cfg_name: str,
    new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return a trace row for every decode step."""
    from benchmark_real_generation_throughput import _get_config

    cfg = _get_config(cfg_name)
    prompt_len = prompt_ids.shape[1]

    # Baseline prefill
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
    baseline_past = out.past_key_values
    baseline_logits = out.logits[:, -1, :]

    # Compress prefill cache
    if hasattr(baseline_past, "to_legacy_cache"):
        compressed_past = list(baseline_past.to_legacy_cache())
    else:
        compressed_past = list(baseline_past)
    compressed_past = _compress_past(compressed_past, cfg, device)
    compressed_past = DynamicCache.from_legacy_cache(tuple(compressed_past))

    trace: list[dict[str, Any]] = []

    # Decode step 0 (first token after prefill)
    first_tok = int(torch.argmax(baseline_logits, dim=-1).item())
    next_ids = torch.tensor([[first_tok]], device=device)

    # We need to generate step by step with both caches and compare logits
    bp = baseline_past
    cp = compressed_past

    for step in range(new_tokens):
        with torch.no_grad():
            out_b = model(
                input_ids=next_ids, past_key_values=bp, use_cache=True
            )
            out_c = model(
                input_ids=next_ids, past_key_values=cp, use_cache=True
            )

        bp = out_b.past_key_values
        cp = out_c.past_key_values
        logit_b = out_b.logits[:, -1, :]
        logit_c = out_c.logits[:, -1, :]

        # Extract cache metadata from baseline (FP16) cache
        # For transformers DynamicCache, shape is (bsz, heads, seq, dim)
        k0, v0 = bp[0] if bp else (None, None)
        kv_cache_len = int(k0.shape[2]) if k0 is not None else 0
        rope_position = prompt_len + step

        trace.append({
            "config": cfg_name,
            "prompt_tokens": prompt_len,
            "decode_step": step,
            "kv_cache_len": kv_cache_len,
            "rope_position": rope_position,
            "logit_cosine": _cosine(logit_b, logit_c),
            "top5_overlap": _topk_overlap(logit_b, logit_c, k=5),
            "kl": _kl_div(logit_b, logit_c),
            "max_abs_logit_delta": float(
                (logit_b - logit_c).abs().max().item()
            ),
            "mean_abs_logit_delta": float(
                (logit_b - logit_c).abs().mean().item()
            ),
        })

        # Use baseline token for next step so divergence doesn't
        # contaminate the comparison.
        next_tok = int(torch.argmax(logit_b, dim=-1).item())
        next_ids = torch.tensor([[next_tok]], device=device)

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-prompt drift debug")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline_fp16", "k8_v5_gs64", "k8_v5_gs32"],
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--new-tokens",
        nargs="+",
        type=int,
        default=[1, 8, 32, 128],
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/short_prompt_drift_trace.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    prompts = {}
    for length in args.prompt_lengths:
        if length > len(dummy_ids):
            repeated = (dummy_ids * ((length // len(dummy_ids)) + 1))[:length]
            prompts[length] = tokenizer.decode(repeated)
        else:
            prompts[length] = tokenizer.decode(dummy_ids[:length])

    trace_rows: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        for length, prompt in prompts.items():
            # Work around MPS DynamicCache bug at >=1024 tokens
            effective_device = device
            effective_device_map = (
                "auto" if device.type != "mps" else "mps"
            )
            if device.type == "mps" and length >= 1024:
                effective_device = torch.device("cpu")
                effective_device_map = "cpu"

            prompt_ids = tokenizer.encode(
                prompt, return_tensors="pt", truncation=True
            )
            prompt_ids = prompt_ids.to(effective_device)

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float16,
                device_map=effective_device_map,
                trust_remote_code=True,
            )
            model.eval()

            for new_tokens in args.new_tokens:
                print(
                    f"Tracing {cfg_name} @ {length} tokens, "
                    f"{new_tokens} steps ..."
                )
                try:
                    rows = _trace_config(
                        model, tokenizer, prompt_ids,
                        cfg_name, new_tokens, effective_device,
                    )
                    trace_rows.extend(rows)
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    trace_rows.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "new_tokens": new_tokens,
                        "error": str(exc),
                    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"model": args.model, "trace": trace_rows}, indent=2
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote trace to {out_path}")


if __name__ == "__main__":
    main()
