#!/usr/bin/env python3
"""
RFSN v10 — Teacher-forced step trace diagnostic.

Isolates exactly where teacher-forced logit drift starts and whether it
accumulates or is present from step 0.

Methodology
-----------
1. Run FP16 prefill on a fixed prompt. Greedy-decode new_tokens steps to
   collect the "baseline token sequence" (same sequence used by
   benchmark_real_generation_throughput.py for teacher-forced evaluation).
2. Re-run FP16 prefill to get a fresh prompt-only cache.
   Compress the cache for each config.
3. For each step, feed the *same baseline forced token* to both the FP16
   reference path and the compressed path. Record per-step logit metrics
   and per-step distribution-shift diagnostics.

This lets us answer:
  - Does drift appear at step 0 (first decode token after a compressed
    prefill) or does it accumulate over steps?
  - What is the rank of the FP16 argmax token in the compressed logits?
  - How does entropy change?
  - How large is the logprob delta for the selected token?

Continuation mode: teacher_forced (all paths use identical forced tokens).
Compare mode: compressed vs FP16-baseline-with-same-forced-tokens.

Shared trace fields per row
---------------------------
  config              str
  prompt_tokens       int
  step                int
  forced_token_id     int
  continuation_mode   "teacher_forced"
  kv_len_before       int   (cache length before this decode step)
  kv_len_after        int   (cache length after this decode step)
  position_id         int
  cache_position      int
  logit_cosine_vs_fp16    float
  top5_overlap_vs_fp16    float
  kl_vs_fp16              float
  max_abs_logit_delta     float
  mean_abs_logit_delta    float
  argmax_fp16_token_id    int
  argmax_quant_token_id   int
  rank_of_fp16_argmax_in_quant   int  (0-based rank in compressed logits)
  logprob_forced_token_fp16      float
  logprob_forced_token_quant     float
  logprob_forced_token_delta     float
  entropy_fp16            float
  entropy_quant           float
  entropy_delta           float
  status                  "pass" | "degraded"

Usage
-----
    python benchmarks/benchmark_teacher_forced_step_trace.py
    python benchmarks/benchmark_teacher_forced_step_trace.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompt-tokens 128 512 \\
        --new-tokens 32 \\
        --out artifacts/proof/experimental/teacher_forced_step_trace.json
"""
from __future__ import annotations

import argparse
import json
import math
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager

_DEFAULT_OUT = Path(
    "artifacts/proof/experimental/teacher_forced_step_trace.json"
)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_fp16": {"name": "baseline_fp16", "family": "baseline"},
    "k8_v5_gs64": {
        "name": "k8_v5_gs64", "family": "stable",
        "k_bits": 8, "v_bits": 5, "group_size": 64,
    },
    "k8_v5_gs32": {
        "name": "k8_v5_gs32", "family": "stable",
        "k_bits": 8, "v_bits": 5, "group_size": 32,
    },
    "turbo_polar": {
        "name": "turbo_polar", "family": "experimental",
        "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64,
    },
    "adaptive": {
        "name": "adaptive", "family": "experimental",
        "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64, "adaptive_angle_range": True,
    },
    "experimental_hybrid": {
        "name": "experimental_hybrid", "family": "experimental",
        "mode": "hybrid_polar_cartesian",
        "feature_dim": 64, "polar_ratio": 0.65, "polar_levels": 4,
        "k_angle_bits": 5, "k_radius_bits": 8,
        "v_angle_bits": 4, "v_radius_bits": 6,
        "cartesian_bits": 6, "group_size": 64,
    },
}

# ---------------------------------------------------------------------------
# Metric helpers
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
    return float(len(ai & bi) / len(ai)) if ai else 0.0


def _entropy(logits: torch.Tensor) -> float:
    p = functional.softmax(logits.float(), dim=-1)
    eps = 1e-10
    h = -torch.sum(p * torch.log(p + eps))
    return float(h.item())


def _logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    lp = functional.log_softmax(logits.float(), dim=-1)
    return float(lp[0, token_id].item())


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    """0-based rank of token_id in descending logit order."""
    sorted_ids = torch.argsort(logits[0], descending=True)
    rank = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    return int(rank[0].item()) if len(rank) > 0 else -1


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _legacy_cache(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past is None:
        return []
    if hasattr(past, "to_legacy_cache"):
        return list(past.to_legacy_cache())
    return list(past)


def _compress_stable(
    past_list: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if cfg["family"] == "baseline":
        return past_list
    compressed = []
    with tempfile.TemporaryDirectory(prefix="rfsn_tf_") as tmpdir:
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
        for layer_idx, (k_t, v_t) in enumerate(past_list):
            k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
            v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
            _bsz, _heads, _seq, dim = k_np.shape
            dim_padded = int(math.ceil(dim / 64.0) * 64)
            if dim_padded != dim:
                pad = dim_padded - dim
                k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
                v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            k_mx = mx.array(k_np)
            v_mx = mx.array(v_np)
            key = f"layer_{layer_idx}"
            mgr.store(key, k_mx, v_mx, token_count=_seq)
            rec = mgr.retrieve(key, out_dtype=mx.float32)
            if rec is None:
                raise RuntimeError("Cache miss on retrieve")
            rk_mx, rv_mx = rec
            rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
            rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
            rk = rk.to(device=device, dtype=k_t.dtype)
            rv = rv.to(device=device, dtype=v_t.dtype)
            compressed.append((rk, rv))
    return compressed


def _compress_experimental(
    past_list: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if cfg["family"] == "baseline":
        return past_list
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
    for k_t, v_t in past_list:
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
        _bsz, _heads, _seq, dim = k_np.shape
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


def _compress(
    past_list: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if cfg["family"] == "stable":
        return _compress_stable(past_list, cfg, device)
    if cfg["family"] == "experimental":
        return _compress_experimental(past_list, cfg, device)
    return past_list


# ---------------------------------------------------------------------------
# Per-config step trace
# ---------------------------------------------------------------------------


def run_teacher_forced_step_trace(
    model_name: str,
    cfg_name: str,
    prompt_tokens: int,
    new_tokens: int,
    device: torch.device,
    seed: int = 42,
    text_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Return one row per decode step for (config, prompt_tokens).

    Args:
        text_prompt: If provided, encode this text to get prompt IDs instead
            of using random token IDs. Must match or exceed prompt_tokens in
            length after encoding. This lets the step trace use the same
            exact prompt as benchmark_real_generation_throughput.py.
    """
    cfg = CONFIGS[cfg_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if text_prompt is not None:
        # Use the provided text prompt, truncated/padded to prompt_tokens
        encoded = tokenizer.encode(
            text_prompt, add_special_tokens=False
        )
        if len(encoded) < prompt_tokens:
            # Repeat if too short
            encoded = (encoded * ((prompt_tokens // len(encoded)) + 1))[
                :prompt_tokens
            ]
        else:
            encoded = encoded[:prompt_tokens]
        prompt_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    else:
        # Build a reproducible prompt of approximately the right length
        torch.manual_seed(seed)
        np.random.seed(seed)
        vocab_size = tokenizer.vocab_size or 32000
        ids = np.random.randint(1000, min(vocab_size - 1, 30000), size=prompt_tokens)
        prompt_ids = torch.tensor([ids.tolist()], dtype=torch.long, device=device)

    actual_prompt_len = prompt_ids.shape[1]

    # Work around MPS DynamicCache bug at >=1024 tokens
    if device.type == "mps" and actual_prompt_len >= 1024:
        device = torch.device("cpu")
        prompt_ids = prompt_ids.to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto" if device.type != "cpu" else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    # -----------------------------------------------------------------------
    # Step 1: FP16 prefill → collect baseline greedy tokens (forced sequence)
    # -----------------------------------------------------------------------
    with torch.no_grad():
        out_base = model(
            input_ids=prompt_ids, past_key_values=None, use_cache=True
        )
    base_past = out_base.past_key_values
    base_first_tok = int(torch.argmax(out_base.logits[:, -1, :], dim=-1).item())

    forced_tokens: list[int] = [base_first_tok]
    past = base_past
    next_tok = base_first_tok
    for _ in range(new_tokens - 1):
        next_ids = torch.tensor([[next_tok]], device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_ids, past_key_values=past, use_cache=True
            )
        past = out.past_key_values
        next_tok = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        forced_tokens.append(next_tok)

    # -----------------------------------------------------------------------
    # Step 2: Re-run prefill → compress cache for this config
    # -----------------------------------------------------------------------
    with torch.no_grad():
        out_prefill = model(
            input_ids=prompt_ids, past_key_values=None, use_cache=True
        )
    fp16_prefill_past = out_prefill.past_key_values

    # FP16 reference path (uncompressed)
    fp16_past = fp16_prefill_past

    # Compressed path
    past_list = _legacy_cache(fp16_prefill_past)
    compressed_list = _compress(past_list, cfg, device)
    compressed_past = DynamicCache.from_legacy_cache(tuple(compressed_list))

    # -----------------------------------------------------------------------
    # Step 3: Teacher-forced step loop — same token fed to both paths
    # -----------------------------------------------------------------------
    rows: list[dict[str, Any]] = []

    for step, forced_token_id in enumerate(forced_tokens):
        forced_ids = torch.tensor([[forced_token_id]], device=device)

        # Record kv_len_before from FP16 path (both should match)
        fp16_legacy_before = _legacy_cache(fp16_past)
        kv_len_before = (
            fp16_legacy_before[0][0].shape[2]
            if fp16_legacy_before else actual_prompt_len
        )
        position_id = kv_len_before
        cache_position = kv_len_before

        with torch.no_grad():
            out_fp16 = model(
                input_ids=forced_ids,
                past_key_values=fp16_past,
                use_cache=True,
            )
            out_quant = model(
                input_ids=forced_ids,
                past_key_values=compressed_past,
                use_cache=True,
            )

        logit_fp16 = out_fp16.logits[:, -1, :]
        logit_quant = out_quant.logits[:, -1, :]

        fp16_past = out_fp16.past_key_values
        compressed_past = out_quant.past_key_values

        fp16_legacy_after = _legacy_cache(fp16_past)
        kv_len_after = (
            fp16_legacy_after[0][0].shape[2]
            if fp16_legacy_after else kv_len_before + 1
        )

        cosine = _cosine(logit_fp16, logit_quant)
        top5 = _topk_overlap(logit_fp16, logit_quant)
        kl = _kl_div(logit_fp16, logit_quant)
        max_delta = float(torch.max(torch.abs(logit_fp16 - logit_quant)).item())
        mean_delta = float(torch.mean(torch.abs(logit_fp16 - logit_quant)).item())

        argmax_fp16 = int(torch.argmax(logit_fp16, dim=-1).item())
        argmax_quant = int(torch.argmax(logit_quant, dim=-1).item())
        rank_fp16_in_quant = _rank_of_token(logit_quant, argmax_fp16)

        lp_forced_fp16 = _logprob_of_token(logit_fp16, forced_token_id)
        lp_forced_quant = _logprob_of_token(logit_quant, forced_token_id)
        lp_delta = lp_forced_quant - lp_forced_fp16

        ent_fp16 = _entropy(logit_fp16)
        ent_quant = _entropy(logit_quant)
        ent_delta = ent_quant - ent_fp16

        status = "pass" if cosine >= 0.99 and top5 >= 0.8 else "degraded"

        rows.append({
            "config": cfg_name,
            "prompt_tokens": actual_prompt_len,
            "step": step,
            "forced_token_id": forced_token_id,
            "continuation_mode": "teacher_forced",
            "kv_len_before": kv_len_before,
            "kv_len_after": kv_len_after,
            "position_id": position_id,
            "cache_position": cache_position,
            "logit_cosine_vs_fp16": cosine,
            "top5_overlap_vs_fp16": top5,
            "kl_vs_fp16": kl,
            "max_abs_logit_delta": max_delta,
            "mean_abs_logit_delta": mean_delta,
            "argmax_fp16_token_id": argmax_fp16,
            "argmax_quant_token_id": argmax_quant,
            "rank_of_fp16_argmax_in_quant": rank_fp16_in_quant,
            "logprob_forced_token_fp16": lp_forced_fp16,
            "logprob_forced_token_quant": lp_forced_quant,
            "logprob_forced_token_delta": lp_delta,
            "entropy_fp16": ent_fp16,
            "entropy_quant": ent_quant,
            "entropy_delta": ent_delta,
            "status": status,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Teacher-forced step trace diagnostic"
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    ap.add_argument(
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
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--text-prompt",
        default=None,
        help=(
            "Use this text string as the prompt (encoded to prompt_tokens). "
            "Defaults to None, which uses random token IDs. "
            "Pass 'foxdog' to use the same repeated 'quick brown fox' text "
            "as benchmark_real_generation_throughput.py."
        ),
    )
    ap.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    all_traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Resolve text_prompt shorthand
    text_prompt: str | None = args.text_prompt
    if text_prompt == "foxdog":
        text_prompt = "The quick brown fox jumps over the lazy dog. " * 200

    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            print(f"  [SKIP] Unknown config: {cfg_name}", flush=True)
            continue
        for pt in args.prompt_tokens:
            prompt_mode = (
                f"text={text_prompt[:20]!r}…"
                if text_prompt else "random-ids"
            )
            print(
                f"  Running: {cfg_name} @ {pt} prompt tokens "
                f"({prompt_mode}) / {args.new_tokens} forced steps …",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                rows = run_teacher_forced_step_trace(
                    model_name=args.model,
                    cfg_name=cfg_name,
                    prompt_tokens=pt,
                    new_tokens=args.new_tokens,
                    device=device,
                    seed=args.seed,
                    text_prompt=text_prompt,
                )
                all_traces.extend(rows)
                elapsed = time.perf_counter() - t0
                n_pass = sum(1 for r in rows if r.get("status") == "pass")
                print(
                    f"    done in {elapsed:.1f}s — "
                    f"{len(rows)} steps, {n_pass}/{len(rows)} pass",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"    ERROR: {exc}", flush=True)
                errors.append({
                    "config": cfg_name,
                    "prompt_tokens": pt,
                    "error": str(exc),
                    "elapsed_s": elapsed,
                })

    if not all_traces and not errors:
        raise SystemExit(
            "No traces generated — no configs ran successfully. "
            "Check that at least one config and prompt length was specified."
        )

    output = {
        "status": "executed",
        "model": args.model,
        "new_tokens": args.new_tokens,
        "seed": args.seed,
        "continuation_mode": "teacher_forced",
        "prompt_source": (
            "text_foxdog"
            if text_prompt and "quick brown fox" in text_prompt
            else ("text_custom" if text_prompt else "random_ids")
        ),
        "total_traces": len(all_traces),
        "total_errors": len(errors),
        "traces": all_traces,
        "errors": errors,
    }

    if output.get("status") == "awaiting_execution":
        raise SystemExit("BUG: output still marked awaiting_execution")
    if not output["traces"]:
        raise SystemExit("No trace rows generated — check for errors above")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"\nWrote {len(all_traces)} trace rows "
        f"({len(errors)} errors) to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
