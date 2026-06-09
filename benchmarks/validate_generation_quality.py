#!/usr/bin/env python3
"""Generation quality validation for RFSN v10 experimental configs.

Checks output stability, not just logit metrics.

Prompts:
  factual prompt
  code prompt
  long-context recall prompt
  reasoning prompt

Compare:
  baseline_fp16, k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive

Metrics:
  exact token match rate
  first divergence position
  semantic similarity if available
  repetition rate
  invalid token rate
  average logprob delta

Usage:
  python benchmarks/validate_generation_quality.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --configs baseline_fp16 k8_v5_gs64 turbo_polar adaptive
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


PROMPTS = {
    "factual": (
        "The capital of France is"
    ),
    "code": (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
    ),
    "long_context": (
        "Alice had a cat named Whiskers. Bob had a dog named Rex. "
        "One day Alice and Bob met at the park. "
        "Alice said her cat liked to chase birds. "
        "Bob said his dog liked to chase cats. "
        "They laughed and decided to keep their pets on leashes. "
        "Later that week, Alice took Whiskers to the vet. "
        "The vet said Whiskers was very healthy. "
        "Bob took Rex for a long hike in the mountains. "
        "Rex was tired but happy. "
        "At the end of the month, Alice and Bob met again. "
        "Question: What is the name of Alice's cat?"
    ),
    "reasoning": (
        "If a train travels 60 miles per hour and needs to cover 180 miles, "
        "how many hours will the trip take?"
    ),
}


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
    raise ValueError(f"Unknown config: {name}")


def _compress_past_stable(past, cfg, device):
    if cfg["name"] == "baseline_fp16":
        return past
    with tempfile.TemporaryDirectory(prefix="rfsn_val_") as tmpdir:
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
        rebuilt = []
        for layer_idx, (k_t, v_t) in enumerate(past):
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
            rebuilt.append((rk, rv))
        return tuple(rebuilt)


def _compress_past_experimental(past, cfg, device):
    if cfg["name"] == "baseline_fp16":
        return past
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
    rebuilt = []
    for k_t, v_t in past:
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
        packet = mgr.quantize(k_mx, v_mx)
        rk_mx, rv_mx = mgr.dequantize(packet)
        rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
        rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
        rk = rk.to(device=device, dtype=k_t.dtype)
        rv = rv.to(device=device, dtype=v_t.dtype)
        rebuilt.append((rk, rv))
    return tuple(rebuilt)


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    cfg: dict[str, Any],
    device: torch.device,
    max_new_tokens: int = 64,
    seed: int = 42,
) -> tuple[str, list[float]]:
    torch.manual_seed(seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token = int(torch.argmax(next_token_logits, dim=-1).item())
    generated_ids = [next_token]
    logprobs: list[float] = []
    logprobs.append(float(torch.max(torch.softmax(next_token_logits, dim=-1)).item()))

    # Compress past
    if cfg["family"] == "stable":
        past = _compress_past_stable(past, cfg, device)
    elif cfg["family"] == "experimental":
        past = _compress_past_experimental(past, cfg, device)

    for _ in range(max_new_tokens - 1):
        input_tensor = torch.tensor([[next_token]], device=device)
        with torch.no_grad():
            out = model(input_ids=input_tensor, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token)
        logprobs.append(float(torch.max(probs).item()))

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, logprobs


def compute_metrics(
    tokenizer: AutoTokenizer,
    baseline_text: str,
    baseline_ids: list[int],
    candidate_text: str,
    candidate_ids: list[int],
    baseline_logprobs: list[float],
    candidate_logprobs: list[float],
) -> dict[str, Any]:
    # Exact token match rate
    min_len = min(len(baseline_ids), len(candidate_ids))
    matches = sum(1 for i in range(min_len) if baseline_ids[i] == candidate_ids[i])
    exact_match_rate = matches / min_len if min_len > 0 else 0.0

    # First divergence
    first_div = min_len
    for i in range(min_len):
        if baseline_ids[i] != candidate_ids[i]:
            first_div = i
            break

    # Repetition rate (simple: repeated tokens)
    if len(candidate_ids) > 1:
        repeats = sum(1 for i in range(1, len(candidate_ids)) if candidate_ids[i] == candidate_ids[i - 1])
        repetition_rate = repeats / (len(candidate_ids) - 1)
    else:
        repetition_rate = 0.0

    # Invalid token rate (placeholder: tokens that decode to empty)
    invalid = sum(1 for t in candidate_ids if t == 0 or t == tokenizer.pad_token_id)
    invalid_rate = invalid / len(candidate_ids) if candidate_ids else 0.0

    # Average logprob delta
    min_lp = min(len(baseline_logprobs), len(candidate_logprobs))
    lp_deltas = [
        abs(baseline_logprobs[i] - candidate_logprobs[i])
        for i in range(min_lp)
    ]
    avg_logprob_delta = sum(lp_deltas) / len(lp_deltas) if lp_deltas else 0.0

    return {
        "exact_token_match_rate": exact_match_rate,
        "first_divergence_position": first_div,
        "repetition_rate": repetition_rate,
        "invalid_token_rate": invalid_rate,
        "average_logprob_delta": avg_logprob_delta,
        "baseline_length": len(baseline_ids),
        "candidate_length": len(candidate_ids),
    }


def main() -> None:
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--configs", nargs="+", default=[
        "baseline_fp16", "k8_v5_gs64", "k8_v5_gs32", "turbo_polar", "adaptive",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/proof/experimental/generation_quality.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Baseline run first
    baseline_outputs: dict[str, tuple[str, list[int], list[float]]] = {}
    print("Running baseline_fp16 ...")
    for prompt_name, prompt_text in PROMPTS.items():
        text, logprobs = generate_with_config(
            model, tokenizer, prompt_text, _get_config("baseline_fp16"),
            device, max_new_tokens=args.max_new_tokens, seed=args.seed,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        baseline_outputs[prompt_name] = (text, ids, logprobs)

    results: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        if cfg_name == "baseline_fp16":
            continue
        print(f"Running {cfg_name} ...")
        cfg = _get_config(cfg_name)
        for prompt_name, prompt_text in PROMPTS.items():
            try:
                text, logprobs = generate_with_config(
                    model, tokenizer, prompt_text, cfg,
                    device, max_new_tokens=args.max_new_tokens, seed=args.seed,
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                baseline_text, baseline_ids, baseline_logprobs = baseline_outputs[prompt_name]
                metrics = compute_metrics(
                    tokenizer,
                    baseline_text, baseline_ids,
                    text, ids,
                    baseline_logprobs, logprobs,
                )
                results.append({
                    "config": cfg_name,
                    "prompt": prompt_name,
                    "metrics": metrics,
                    "generated_text": text,
                })
                print(f"  {prompt_name}: match_rate={metrics['exact_token_match_rate']:.2%}")
            except Exception as exc:
                print(f"  {prompt_name}: FAILED: {exc}")
                results.append({
                    "config": cfg_name,
                    "prompt": prompt_name,
                    "error": str(exc),
                })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "release": "experimental",
            "model": args.model,
            "seed": args.seed,
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
