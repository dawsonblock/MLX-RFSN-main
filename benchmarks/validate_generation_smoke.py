#!/usr/bin/env python3
"""Generation smoke test for RFSN v10 Main 28.

Runs greedy decode for baseline and compressed KV configs, then checks:
  - Token match rate vs baseline (first N tokens)
  - Normalised edit distance between generated strings
  - Repetition rate (fraction of duplicate n-grams)
  - No NaN/Inf in logits at any decode step

Produces a JSON summary at --out (default: artifacts/proof/main28/generation_smoke.json).

Usage:
  python benchmarks/validate_generation_smoke.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --tokens 128 --decode 32 \\
      --out artifacts/proof/main28/generation_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """Fraction of n-gram positions that are duplicates of an earlier n-gram."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    seen: set[tuple[int, ...]] = set()
    dup = 0
    for g in ngrams:
        if g in seen:
            dup += 1
        seen.add(g)
    return dup / len(ngrams)


def _edit_distance(a: str, b: str) -> int:
    """Standard DP edit distance (Levenshtein)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = curr
    return prev[-1]


def _normalised_edit_distance(a: str, b: str) -> float:
    dist = _edit_distance(a, b)
    denom = max(len(a), len(b), 1)
    return dist / denom


# ---------------------------------------------------------------------------
# KV cache helpers (mirrors validate_real_model_kv.py)
# ---------------------------------------------------------------------------

def _to_legacy_cache(pkv: Any) -> tuple[list, type]:
    if pkv is None:
        return [], type(None)
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache(), type(pkv)
    return list(pkv), type(pkv)


def _from_legacy_cache(legacy: list, cache_cls: type) -> Any:
    if not legacy:
        return None
    if cache_cls is type(None) or cache_cls is list:
        return legacy
    try:
        return cache_cls.from_legacy_cache(legacy)
    except Exception:
        return legacy


def _clone_legacy_cache(legacy: list) -> list:
    cloned = []
    for layer in legacy:
        if isinstance(layer, (tuple, list)):
            cloned.append(tuple(t.clone() for t in layer))
        else:
            cloned.append(layer)
    return cloned


def _compress_past(past_legacy: list, config: dict, device: torch.device) -> list:
    """Compress KV cache via RFSN TurboQuant and decompress back."""
    from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
    import mlx.core as mx
    import numpy as np

    k_bits = config["k_bits"]
    v_bits = config["v_bits"]
    group_size = config["group_size"]

    compressed = []
    layer_map = config.get("layer_map", {})
    for layer_idx, (k, v) in enumerate(past_legacy):
        k_np = k.float().cpu().numpy()
        v_np = v.float().cpu().numpy()
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, group_size=group_size,
            cache_dir=str(Path.home() / ".rfsn_cache"),
        )
        token_count = k.shape[2]  # [B, H, T, D]
        kb, vb = layer_map.get(layer_idx, (k_bits, v_bits))
        mgr.store(
            skill_pattern="smoke_test",
            keys=mx.array(k_np),
            values=mx.array(v_np),
            token_count=token_count,
            k_bits=kb, v_bits=vb,
        )
        rk, rv = mgr.retrieve(skill_pattern="smoke_test")
        k_out = torch.from_numpy(np.array(rk)).to(device=device, dtype=k.dtype)
        v_out = torch.from_numpy(np.array(rv)).to(device=device, dtype=v.dtype)
        compressed.append((k_out, v_out))
    return compressed


# ---------------------------------------------------------------------------
# Core greedy decode
# ---------------------------------------------------------------------------

def _greedy_decode(
    model,
    context_ids: torch.Tensor,
    n_decode: int,
    past_legacy: list | None = None,
    cache_cls: type | None = None,
) -> tuple[list[int], list[torch.Tensor], bool]:
    """Return (token_ids, logits_list, had_nan)."""
    if past_legacy is not None:
        past = _from_legacy_cache(past_legacy, cache_cls)
        input_ids = context_ids
    else:
        past = None
        input_ids = context_ids

    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)

    past = out.past_key_values
    generated: list[int] = []
    logits_list: list[torch.Tensor] = []
    had_nan = False

    for _ in range(n_decode):
        logits = out.logits[:, -1, :].float()
        if _has_nan_or_inf(logits):
            had_nan = True
        logits_list.append(logits)
        next_tok = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_tok)
        next_ids = torch.tensor([[next_tok]], device=context_ids.device)
        with torch.no_grad():
            out = model(
                input_ids=next_ids, past_key_values=past, use_cache=True
            )
        past = out.past_key_values

    return generated, logits_list, had_nan


# ---------------------------------------------------------------------------
# Config parsing (mirrors validate_real_model_kv.py)
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "baseline_fp16": {"name": "baseline_fp16", "k_bits": 16, "v_bits": 16, "group_size": 64},
    "mixed_L0-1k8v4_restk6v4_gs64": {"name": "mixed_L0-1k8v4_restk6v4_gs64", "k_bits": 6, "v_bits": 4, "group_size": 64, "layer_map": {0: (8, 4), 1: (8, 4)}},
    "mixed_L0k8v4_restk6v4_gs64": {"name": "mixed_L0k8v4_restk6v4_gs64", "k_bits": 6, "v_bits": 4, "group_size": 64, "layer_map": {0: (8, 4)}},
    "k8_v3_gs64": {"name": "k8_v3_gs64", "k_bits": 8, "v_bits": 3, "group_size": 64},
    "k8_v4_gs64": {"name": "k8_v4_gs64", "k_bits": 8, "v_bits": 4, "group_size": 64},
    "k8_v5_gs64": {"name": "k8_v5_gs64", "k_bits": 8, "v_bits": 5, "group_size": 64},
    "k8_v4_gs32": {"name": "k8_v4_gs32", "k_bits": 8, "v_bits": 4, "group_size": 32},
    "k8_v5_gs32": {"name": "k8_v5_gs32", "k_bits": 8, "v_bits": 5, "group_size": 32},
    "k6_v6_gs64": {"name": "k6_v6_gs64", "k_bits": 6, "v_bits": 6, "group_size": 64},
    "k4_v4_gs64": {"name": "k4_v4_gs64", "k_bits": 4, "v_bits": 4, "group_size": 64},
}


def _parse_config(name: str) -> dict[str, Any]:
    if name in _CONFIG_REGISTRY:
        return _CONFIG_REGISTRY[name]
    parts = name.split("_")
    cfg: dict[str, Any] = {"name": name, "k_bits": 8, "v_bits": 4, "group_size": 64}
    for p in parts:
        if p.startswith("k") and p[1:].isdigit():
            cfg["k_bits"] = int(p[1:])
        elif p.startswith("v") and p[1:].isdigit():
            cfg["v_bits"] = int(p[1:])
        elif p.startswith("gs") and p[2:].isdigit():
            cfg["group_size"] = int(p[2:])
    return cfg


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

_SMOKE_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The early bird catches the worm. "
)


def _run_smoke(
    model_id: str,
    tokens: int,
    n_decode: int,
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.to(device)
    model.eval()

    prompt = _SMOKE_PROMPT * max(1, tokens // len(_SMOKE_PROMPT.split()) + 1)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :tokens].to(device)
    if input_ids.shape[1] < 4:
        raise ValueError(f"Need at least 4 context tokens, got {input_ids.shape[1]}")

    print(f"Context tokens: {input_ids.shape[1]}, decode steps: {n_decode}")

    # Baseline greedy decode
    t0 = time.perf_counter()
    baseline_toks, baseline_logits, baseline_nan = _greedy_decode(
        model, input_ids, n_decode
    )
    baseline_dt = (time.perf_counter() - t0) * 1000.0
    baseline_text = tokenizer.decode(baseline_toks, skip_special_tokens=True)

    # Build baseline past from all-but-last context token.
    # The last token will be fed once below to avoid duplication.
    with torch.no_grad():
        pre_out = model(input_ids=input_ids[:, :-1], use_cache=True)
    pre_legacy, pre_cache_cls = _to_legacy_cache(pre_out.past_key_values)

    results: list[dict[str, Any]] = []

    for config in configs:
        if config["name"] == "baseline_fp16":
            token_match = 1.0
            ned = 0.0
            rep_rate = _repetition_rate(baseline_toks)
            had_nan = baseline_nan
            gen_toks = baseline_toks
            gen_text = baseline_text
            latency_ms = baseline_dt
        else:
            compressed = _compress_past(
                _clone_legacy_cache(pre_legacy), config, device
            )
            t0 = time.perf_counter()

            # Feed last context token ONCE through compressed past
            compressed_past = _from_legacy_cache(compressed, pre_cache_cls)
            with torch.no_grad():
                first_out = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=compressed_past,
                    use_cache=True,
                )

            # Continue greedy decode
            first_logits = first_out.logits[:, -1, :].float()
            had_nan = _has_nan_or_inf(first_logits)
            first_tok = int(torch.argmax(first_logits, dim=-1).item())
            gen_toks = [first_tok]
            past = first_out.past_key_values

            for _ in range(n_decode - 1):
                next_ids = torch.tensor(
                    [[gen_toks[-1]]], device=device
                )
                with torch.no_grad():
                    out = model(
                        input_ids=next_ids,
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()
                if _has_nan_or_inf(logits):
                    had_nan = True
                next_tok = int(torch.argmax(logits, dim=-1).item())
                gen_toks.append(next_tok)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)
            n = min(len(gen_toks), len(baseline_toks))
            matches = sum(
                g == b for g, b in zip(gen_toks[:n], baseline_toks[:n])
            )
            token_match = matches / n if n > 0 else float("nan")
            ned = _normalised_edit_distance(gen_text, baseline_text)
            rep_rate = _repetition_rate(gen_toks)

        if not math.isfinite(token_match):
            status = "nan_fail"
        elif had_nan:
            status = "nan_fail"
        elif token_match < 0.5:
            status = "fail"
        elif ned > 0.5:
            status = "fail"
        elif rep_rate > 0.8:
            status = "fail"
        else:
            status = "pass"

        entry: dict[str, Any] = {
            "name": config["name"],
            "token_match_rate": token_match,
            "normalised_edit_distance": ned,
            "repetition_rate_4gram": rep_rate,
            "had_nan_logits": had_nan,
            "latency_ms": latency_ms,
            "status": status,
        }
        results.append(entry)
        print(
            f"  {config['name']}: token_match={token_match:.3f} "
            f"ned={ned:.3f} rep={rep_rate:.3f} nan={had_nan} "
            f"status={status}"
        )

    payload: dict[str, Any] = {
        "release": "main28",
        "analysis": "generation_smoke",
        "model": model_id,
        "context_tokens": int(input_ids.shape[1]),
        "decode_steps": n_decode,
        "baseline_text": baseline_text[:200],
        "configs": results,
        "all_pass": all(r["status"] == "pass" for r in results),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote generation smoke to {out_path}")
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RFSN v10 Main 28 generation smoke test"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--tokens", type=int, default=128,
        help="Number of context tokens",
    )
    parser.add_argument(
        "--decode", type=int, default=32,
        help="Number of greedy decode steps",
    )
    parser.add_argument(
        "--configs",
        default=(
            "baseline_fp16,mixed_L0-1k8v4_restk6v4_gs64,"
            "k8_v4_gs64,k8_v5_gs64,k8_v3_gs64,"
            "k6_v6_gs64,k8_v4_gs32,k8_v5_gs32,k4_v4_gs64"
        ),
        help="Comma-separated config names",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main28/generation_smoke.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code from HuggingFace",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [_parse_config(n) for n in config_names]

    payload = _run_smoke(
        model_id=args.model,
        tokens=args.tokens,
        n_decode=args.decode,
        configs=configs,
        device=device,
        out_path=Path(args.out),
        trust_remote_code=args.trust_remote_code,
    )

    if not payload["all_pass"]:
        failed = [r["name"] for r in payload["configs"] if r["status"] != "pass"]
        print(f"FAIL: {failed}", file=sys.stderr)
        sys.exit(1)

    print("generation smoke: OK")


if __name__ == "__main__":
    main()
