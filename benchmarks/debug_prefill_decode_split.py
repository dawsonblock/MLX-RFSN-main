#!/usr/bin/env python3
"""Prefill-vs-decode split isolation benchmark for RFSN v10.

Runs four configurations to isolate where corruption starts:
  A. FP16 prefill + FP16 decode
  B. Quantized prefill + FP16 decode
  C. FP16 prefill + quantized decode
  D. Quantized prefill + quantized decode

Output: artifacts/proof/experimental/prefill_decode_split.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from benchmark_real_generation_throughput import (
    _compress_experimental,
    _compress_stable,
    _get_config,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _run_prefill_decode_split(
    model,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    cfg: dict[str, Any],
    device: torch.device,
    mode: str,  # "A", "B", "C", "D"
) -> dict[str, Any]:
    """mode: A=FP16/FP16, B=Quant/FP16, C=FP16/Quant, D=Quant/Quant"""

    def _prefill(ids, use_quant: bool):
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=None, use_cache=True)
        past = out.past_key_values
        if use_quant:
            if hasattr(past, "to_legacy_cache"):
                past = list(past.to_legacy_cache())
            else:
                past = list(past)
            if cfg["family"] == "stable":
                past, _, _ = _compress_stable(past, cfg, device)
            else:
                past, _, _ = _compress_experimental(past, cfg, device)
            past = DynamicCache.from_legacy_cache(tuple(past))
        return past, out.logits[:, -1, :]

    def _decode_step(ids, past, use_quant: bool):
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=past, use_cache=True)
        new_past = out.past_key_values
        if use_quant:
            if hasattr(new_past, "to_legacy_cache"):
                new_past = list(new_past.to_legacy_cache())
            else:
                new_past = list(new_past)
            if cfg["family"] == "stable":
                new_past, _, _ = _compress_stable(new_past, cfg, device)
            else:
                new_past, _, _ = _compress_experimental(new_past, cfg, device)
            new_past = DynamicCache.from_legacy_cache(tuple(new_past))
        return new_past, out.logits[:, -1, :]

    quant_prefill = mode in ("B", "D")
    quant_decode = mode in ("C", "D")

    past, first_logit = _prefill(prompt_ids, use_quant=quant_prefill)
    next_tok = int(torch.argmax(first_logit, dim=-1).item())

    logits_list = [first_logit]
    for _ in range(new_tokens - 1):
        next_ids = torch.tensor([[next_tok]], device=device)
        past, logit = _decode_step(next_ids, past, use_quant=quant_decode)
        logits_list.append(logit)
        next_tok = int(torch.argmax(logit, dim=-1).item())

    return {
        "mode": mode,
        "quant_prefill": quant_prefill,
        "quant_decode": quant_decode,
        "logits_count": len(logits_list),
        "logits_list": logits_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill-decode split debug")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["k8_v5_gs64", "k8_v5_gs32", "turbo_polar", "adaptive",
                 "experimental_hybrid"],
    )
    parser.add_argument(
        "--prompt-tokens", nargs="+", type=int, default=[128, 512]
    )
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument(
        "--out",
        default="artifacts/proof/experimental/prefill_decode_split.json",
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto" if device.type != "mps" else "mps",
        trust_remote_code=True,
    )
    model.eval()

    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)

    results: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        cfg = _get_config(cfg_name)
        for length in args.prompt_tokens:
            if length > len(dummy_ids):
                factor = (length // len(dummy_ids)) + 1
                repeated = (dummy_ids * factor)[:length]
                prompt = tokenizer.decode(repeated)
            else:
                prompt = tokenizer.decode(dummy_ids[:length])

            prompt_ids = tokenizer.encode(
                prompt, return_tensors="pt", truncation=True
            )
            prompt_ids = prompt_ids.to(device)

            print(f"Split {cfg_name} @ {length} ...")

            # Run mode A first to collect reference logits
            try:
                run_a = _run_prefill_decode_split(
                    model, prompt_ids, args.new_tokens,
                    cfg, device, "A",
                )
                ref_logits = run_a["logits_list"]
                results.append({
                    "config": cfg_name,
                    "prompt_tokens": length,
                    "new_tokens": args.new_tokens,
                    "mode": "A",
                    "continuation_mode": "free_running_greedy",
                    "token_sequence_source": "mode_A_greedy",
                    "quant_prefill": run_a["quant_prefill"],
                    "quant_decode": run_a["quant_decode"],
                    "logits_count": run_a["logits_count"],
                    "status": "ok",
                })
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                print(f"  FAILED mode A: {exc}")
                for mode in ("A", "B", "C", "D"):
                    results.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "new_tokens": args.new_tokens,
                        "mode": mode,
                        "status": "error",
                        "error": str(exc),
                    })
                continue

            for mode in ("B", "C", "D"):
                try:
                    run = _run_prefill_decode_split(
                        model, prompt_ids, args.new_tokens,
                        cfg, device, mode,
                    )
                    cosines = []
                    top5s = []
                    min_len = min(len(ref_logits), len(run["logits_list"]))
                    for i in range(min_len):
                        b = ref_logits[i]
                        c = run["logits_list"][i]
                        cosines.append(
                            float(torch.nn.functional.cosine_similarity(
                                b.reshape(-1), c.reshape(-1), dim=0
                            ).item())
                        )
                        ai = set(torch.topk(b, k=5, dim=-1).indices[0].tolist())
                        bi = set(torch.topk(c, k=5, dim=-1).indices[0].tolist())
                        top5s.append(float(len(ai & bi) / len(ai)) if ai else 0.0)
                    results.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "new_tokens": args.new_tokens,
                        "mode": mode,
                        "continuation_mode": "free_running_greedy",
                        "token_sequence_source": f"mode_{mode}_greedy",
                        "quant_prefill": run["quant_prefill"],
                        "quant_decode": run["quant_decode"],
                        "logits_count": run["logits_count"],
                        "logit_cosine_vs_a": (
                            sum(cosines) / len(cosines) if cosines else float("nan")
                        ),
                        "top5_overlap_vs_a": (
                            sum(top5s) / len(top5s) if top5s else float("nan")
                        ),
                        "status": "ok",
                    })
                except (RuntimeError, ValueError, OSError, TypeError) as exc:
                    print(f"  FAILED mode {mode}: {exc}")
                    results.append({
                        "config": cfg_name,
                        "prompt_tokens": length,
                        "new_tokens": args.new_tokens,
                        "mode": mode,
                        "status": "error",
                        "error": str(exc),
                    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "model": args.model,
            "methodology_note": (
                "Each mode (A/B/C/D) uses its own independent greedy "
                "decode (free_running_greedy). Logits are averaged across "
                "all decode steps, not teacher-forced. "
                "Modes share the same prompt but diverge after the first "
                "decode token. Compare with teacher_forced_step_trace.json "
                "for teacher-forced per-step quality."
            ),
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
