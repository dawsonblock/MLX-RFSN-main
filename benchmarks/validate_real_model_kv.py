#!/usr/bin/env python3
"""Main 28 real-model KV validation runner.

Runs a HuggingFace causal LM (auto-downloaded) in baseline and compressed
modes. Compresses KV past tensors via RFSN TurboQuant, then decodes and
compares logits.

Supports:
- --model: HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
- --configs: comma-separated config names (e.g., k8_v3_gs64,k4_v4_gs64)
- --contexts: comma-separated token counts for long-context validation
- --tokens: number of tokens to test (default 512)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


# Alpha pass thresholds
COSINE_MEAN_THRESHOLD = 0.995
COSINE_MIN_THRESHOLD = 0.990
TOP1_MATCH_THRESHOLD = 0.95
TOP5_OVERLAP_THRESHOLD = 0.95
PPL_DELTA_REL_THRESHOLD = 0.10
KL_DIV_THRESHOLD = 0.02

# Mixed-precision config registry: per-layer bit overrides on top of default
_MIXED_CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "mixed_L0-1k8v4_restk6v4_gs64": {
        "name": "mixed_L0-1k8v4_restk6v4_gs64",
        "k_bits": 6,
        "v_bits": 4,
        "group_size": 64,
        "layer_map": {0: (8, 4), 1: (8, 4)},
    },
    "mixed_L0k8v4_restk6v4_gs64": {
        "name": "mixed_L0k8v4_restk6v4_gs64",
        "k_bits": 6,
        "v_bits": 4,
        "group_size": 64,
        "layer_map": {0: (8, 4)},
    },
}


def _parse_config(name: str) -> dict[str, Any]:
    """Parse config name like 'k8_v3_gs64' -> bits and group_size."""
    if name == "baseline_fp16":
        return {"name": name, "k_bits": 16, "v_bits": 16, "group_size": 64}
    if name in _MIXED_CONFIG_REGISTRY:
        return _MIXED_CONFIG_REGISTRY[name].copy()
    # Expected format: k{bits}_v{bits}_gs{group_size}
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
    }


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


def _decode_nll(
    model, past, decode_token: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """Decode one token and return (logits_fp32, past_key_values).

    The returned logits are produced AFTER feeding decode_token, so they
    predict the token that would follow decode_token.
    """
    with torch.no_grad():
        out = model(
            input_ids=decode_token, past_key_values=past, use_cache=True
        )
    logits = out.logits[:, -1, :].float()
    return logits, out.past_key_values


def _decode_nll_multi(
    model,
    past,
    decode_tokens: torch.Tensor,
    n_positions: int = 64,
) -> tuple[list[torch.Tensor], float]:
    """Causal-correct multi-position NLL; return (all_scored_logits, avg_nll).

    Scoring rule (causal LM):
      - logits produced by feeding token t predict token t+1.
      - To score decode_tokens[i], we use the logits that were live
        *before* decode_tokens[i] was fed (i.e. after feeding
        decode_tokens[i-1], or the context forward pass for i==0).

    Uses shifted-window scoring: token 0 advances the cache but is NOT
    scored.  Tokens 1..n are each scored against the logits produced by
    the previous forward pass.  This yields `n_positions` scored logits.

    Returns:
        all_scored_logits: list of logits tensors, one per scored position.
                         Length == number of positions actually scored.
        avg_nll: mean NLL over all scored positions.
    """
    n = min(n_positions, decode_tokens.shape[1])
    if n == 0:
        return [], float("nan")

    nlls: list[float] = []
    scored_logits: list[torch.Tensor] = []
    current_past = past
    prev_logits: torch.Tensor | None = None

    # Shifted-window: consume n_positions+1 tokens to score n_positions.
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
            # Score tok[i] against logits produced BEFORE feeding tok[i]
            if _has_nan_or_inf(prev_logits):
                return [], float("nan")
            nll = float(F.cross_entropy(prev_logits, tok[:, 0]).item())
            nlls.append(nll)
            scored_logits.append(prev_logits)

        prev_logits = logits

    if not nlls or not scored_logits:
        return [], float("nan")
    return scored_logits, float(sum(nlls) / len(nlls))


def _compress_decompress_past(
    past_key_values,
    config: dict[str, Any],
    device: torch.device,
    compress_layers: set[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress and decompress past KV values using RFSN manager.

    If compress_layers is provided, only those layer indices are compressed;
    all others are returned unchanged (fp16 baseline).
    """
    if config["name"] == "baseline_fp16":
        return past_key_values

    mgr = RFSNTurboQuantKVManager(
        k_bits=config["k_bits"],
        v_bits=config["v_bits"],
        group_size=config["group_size"],
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=True,
        strict_metal=False,
        max_memory_gb=2.0,
        cache_dir=str(Path.home() / ".rfsn_cache"),
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
        kb, vb = config.get("layer_map", {}).get(layer_idx, (config["k_bits"], config["v_bits"]))
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


def _to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    return past_key_values, None


def _from_legacy_cache(legacy_cache, cache_cls):
    if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_cache)
    return legacy_cache


def _clone_legacy_cache(legacy_cache):
    """Deep-clone tensors in legacy cache to avoid in-place mutation."""
    if legacy_cache is None:
        return None
    return tuple((k.clone(), v.clone()) for k, v in legacy_cache)


def _finite_mean(vals: list[float]) -> float:
    """Return mean of finite values, or NaN if none."""
    finite = [v for v in vals if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _compute_logit_metrics(
    baseline_logits_list: list[torch.Tensor],
    compressed_logits_list: list[torch.Tensor],
) -> dict[str, float]:
    """Compute cosine, top1, top5, KL across all scored positions."""
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
        "logit_cosine_min": min(
            c for c in cosines if math.isfinite(c)
        ) if any(math.isfinite(c) for c in cosines) else float("nan"),
        "logit_max_abs_diff": _finite_mean(max_diffs),
        "top1_match_rate": top1_matches / n if n > 0 else float("nan"),
        "top5_overlap_mean": _finite_mean(top5_overlaps),
        "kl_divergence_mean": _finite_mean(kls),
        "token_positions_evaluated": n,
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
    if config["name"] != "baseline_fp16":
        past_legacy = _compress_decompress_past(
            past_legacy, config, device, compress_layers=compress_layers
        )

    past = _from_legacy_cache(past_legacy, cache_cls)

    t0 = time.perf_counter()
    logits_list, nll = _decode_nll_multi(
        model, past, decode_tokens, n_positions=n_decode_positions
    )
    dt = (time.perf_counter() - t0) * 1000.0

    if not logits_list or not baseline_logits_list:
        return {
            "name": config["name"],
            "k_bits": config["k_bits"],
            "v_bits": config["v_bits"],
            "group_size": config["group_size"],
            "logit_cosine_mean": float("nan"),
            "logit_cosine_min": float("nan"),
            "logit_max_abs_diff": float("nan"),
            "top1_match_rate": float("nan"),
            "top5_overlap_mean": float("nan"),
            "avg_nll_delta": float("nan"),
            "token_positions_evaluated": 0,
            "kl_divergence_mean": float("nan"),
            "latency_ms": dt,
            "route_used": "retrieve"
            if config["name"] != "baseline_fp16"
            else "baseline_fp16",
        }

    metrics = _compute_logit_metrics(baseline_logits_list, logits_list)

    return {
        "name": config["name"],
        "k_bits": config["k_bits"],
        "v_bits": config["v_bits"],
        "group_size": config["group_size"],
        "avg_nll_delta": nll - baseline_nll,
        "latency_ms": dt,
        "route_used": "retrieve"
        if config["name"] != "baseline_fp16"
        else "baseline_fp16",
        **metrics,
    }


def _is_nan_result(result: dict[str, Any]) -> bool:
    """Return True if any primary metric is NaN or Inf."""
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


def _determine_status(result: dict[str, Any], *, baseline_nll: float = 0.0) -> str:
    """Apply alpha pass thresholds honestly."""
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
    # 1. Technical explanation (self-referential to the system being tested)
    (
        "Machine learning models are trained on large datasets to learn "
        "statistical patterns. The transformer architecture uses self-attention "
        "to process sequences in parallel, enabling scalable training. "
        "Key-value caches store attention states to accelerate "
        "autoregressive decoding by avoiding redundant computation. "
        "Quantization reduces memory bandwidth at the cost of "
        "numerical precision. " * 15
    ),
    # 2. Code completion pattern
    (
        "def compute_statistics(data):\n"
        "    \"\"\"Calculate mean, median, and standard deviation.\"\"\"\n"
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
        "    return {\"mean\": mean, \"median\": median, \"std\": std_dev}\n\n"
        "# Example usage with test data\n"
        "test_values = [23, 45, 67, 89, 12, 34, 56, 78, 90, 11]\n"
        "result = compute_statistics(test_values)\n"
        "print(f\"Mean: {result['mean']:.2f}\")\n\n" * 10
    ),
    # 3. Number reasoning / arithmetic reasoning
    (
        "Calculate the following step by step: 127 plus 345 equals 472. "
        "Now multiply 472 by 3 to obtain 1416. Divide 1416 by 4 to get 354. "
        "Add 100 to reach 454. Subtract 54 to return to 400. "
        "Double it for 800. Halve that for 400 again. "
        "The pattern confirms arithmetic consistency. " * 25
    ),
    # 4. JSON-like structured data continuation
    (
        '{"project": "rfsn_v10", "version": "main28", "status": "alpha", '
        '"components": [{"name": "kv_manager", "type": "compression", '
        '"bits": [4, 5, 6, 8]}, {"name": "attention", "type": "sparse", '
        '"top_k": 0.3}, {"name": "runtime", "type": "orchestrator"}], '
        '"metrics": {"cosine_mean": 0.999, "kl_div": 0.001, "nll_delta": 0.01}, '
        '"hardware": {"device": "mps", "memory_gb": 16}} ' * 80
    ),
    # 5. Long repeated context (original staple)
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

    # Per-prompt results accumulator: config_name -> list of single-prompt
    # results
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

    # Aggregate: mean each metric across prompts per config
    config_results: list[dict[str, Any]] = []
    for config in configs:
        per_prompt = prompt_results_by_config[config["name"]]
        if not per_prompt:
            skipped: dict[str, Any] = {
                "name": config["name"],
                "k_bits": config["k_bits"],
                "v_bits": config["v_bits"],
                "group_size": config["group_size"],
                "status": "skipped",
                "per_prompt": [],
            }
            if "layer_map" in config:
                skipped["layer_map"] = {str(k): v for k, v in config["layer_map"].items()}
            config_results.append(skipped)
            continue

        agg: dict[str, Any] = {
            "name": config["name"],
            "k_bits": config["k_bits"],
            "v_bits": config["v_bits"],
            "group_size": config["group_size"],
            "logit_cosine_mean": _finite_mean(
                [r["logit_cosine_mean"] for r in per_prompt]
            ),
            "logit_cosine_min": min(
                r["logit_cosine_min"] for r in per_prompt
                if math.isfinite(r["logit_cosine_min"])
            ) if any(
                math.isfinite(r["logit_cosine_min"]) for r in per_prompt
            ) else float("nan"),
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
        if "layer_map" in config:
            agg["layer_map"] = {str(k): v for k, v in config["layer_map"].items()}
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
        "release": "main28",
        "validation_class": "real_non_random_model_validation",
        "model": model_id,
        "hardware": _get_hardware_info(),
        "tokens_tested": tokens,
        "prompts_count": len(prompt_texts),
        "configs": config_results,
        "sparse_enabled": False,
        "notes": [
            "Real non-random model validation executed.",
            "Sparse decode is disabled by default.",
            f"Multi-prompt validation: {len(prompt_texts)} prompts aggregated.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote real model validation to {out_path}")
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
    # Use float32 on MPS to avoid fp16 overflow at longer sequence lengths
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
                    "out of memory", "no memory", "mps allocator"
                ]
                if any(ind in msg for ind in oom_indicators):
                    result = {"name": config["name"], "oom": True, "status": "oom"}
                else:
                    raise
            config_results.append(result)

        context_entries.append({
            "tokens": ctx_tokens,
            "configs": config_results,
        })

    # Determine best configs from results — all-context-passing logic
    def _passes_all_contexts(config_name: str, ctxs: list[dict]) -> bool:
        """Return True only if config passes in EVERY tested context."""
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
        # Best mean cosine among configs passing ALL contexts
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
        # Prefer highest compression (lowest bits) among configs passing ALL contexts
        best = ""
        best_score = float("inf")
        for name in _collect_config_names(ctxs):
            if not _passes_all_contexts(name, ctxs):
                continue
            # Get score from first occurrence (same config has same k_bits/v_bits)
            score = None
            for ctx in ctxs:
                for c in ctx["configs"]:
                    if c.get("name") == name:
                        layer_map = c.get("layer_map")
                        if layer_map and c["name"] in _MIXED_CONFIG_REGISTRY:
                            n_layers = 24
                            kb, vb = c["k_bits"], c["v_bits"]
                            total = (kb + vb) * n_layers
                            for layer_idx, (okb, ovb) in layer_map.items():
                                total -= (kb + vb)
                                total += (okb + ovb)
                            score = total / n_layers
                        else:
                            score = c.get("k_bits", 16) + c.get("v_bits", 16)
                        break
                if score is not None:
                    break
            if score is not None and score < best_score:
                best_score = score
                best = name
        return best

    def _recommended(ctxs: list[dict]) -> str:
        for prefer in (
            "mixed_L0-1k8v4_restk6v4_gs64",
            "k8_v4_gs64",
            "k8_v4_gs32",
            "k8_v5_gs64",
            "k8_v5_gs32",
            "k8_v3_gs64",
        ):
            if _passes_all_contexts(prefer, ctxs):
                return prefer
        return "baseline_fp16"

    def _compression_estimate(ctxs: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        n = 64 * 64
        def _b(nv, bits): return ((nv + (32 // bits) - 1) // (32 // bits)) * 4
        sc = ((n + 64 - 1) // 64) * 4
        def _comp(k, v): return (n * 2 * 2) / (_b(n, k) + _b(n, v) + sc * 2)
        for ctx in ctxs:
            for c in ctx["configs"]:
                name = c["name"]
                if name in out:
                    continue
                layer_map = c.get("layer_map")
                if layer_map and name in _MIXED_CONFIG_REGISTRY:
                    total = 0.0
                    kb, vb = c["k_bits"], c["v_bits"]
                    for _ in range(24):
                        total += _comp(kb, vb)
                    for layer_idx, (okb, ovb) in layer_map.items():
                        total -= _comp(kb, vb)
                        total += _comp(okb, ovb)
                    out[name] = total / 24
                else:
                    out[name] = _comp(c["k_bits"], c["v_bits"])
        return out

    rejected = [
        name for name in _collect_config_names(context_entries)
        if name != "baseline_fp16" and not _passes_all_contexts(name, context_entries)
    ]

    payload: dict[str, Any] = {
        "release": "main28",
        "model": model_id,
        "contexts": context_entries,
        "summary": {
            "best_quality_config": _best_quality(context_entries),
            "best_memory_config_passing_all_contexts": _best_memory(context_entries),
            "recommended_default": _recommended(context_entries),
            "compression_estimate_x": _compression_estimate(context_entries),
            "rejected_configs": rejected,
            "production_ready": False,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote long-context validation to {out_path}")
    return payload


def _run_per_layer_sensitivity(
    model_id: str,
    tokens: int,
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Compress each layer individually to identify sensitivity."""
    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.to(device)
    model.eval()

    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :tokens].to(device)
    if input_ids.shape[1] < 67:
        raise ValueError(
            "Need at least 67 tokens for per-layer sensitivity "
            "(context + 65 decode tokens)"
        )
    context_ids = input_ids[:, :-65]
    decode_tokens = input_ids[:, -65:]

    with torch.no_grad():
        baseline_ctx = model(input_ids=context_ids, use_cache=True)
    baseline_legacy, baseline_cache_cls = _to_legacy_cache(
        baseline_ctx.past_key_values
    )
    baseline_past = _from_legacy_cache(baseline_legacy, baseline_cache_cls)
    baseline_logits_list, baseline_nll = _decode_nll_multi(
        model, baseline_past, decode_tokens, n_positions=64
    )

    num_layers = len(baseline_legacy)
    # Use the first non-baseline config for sensitivity testing
    test_configs = [c for c in configs if c["name"] != "baseline_fp16"]
    if not test_configs:
        test_configs = [configs[0]]

    all_results: list[dict[str, Any]] = []
    for config in test_configs:
        print(f"Per-layer sensitivity: {config['name']} ...")
        layer_results: list[dict[str, Any]] = []
        for layer_idx in range(num_layers):
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
                compress_layers={layer_idx},
                n_decode_positions=64,
            )
            layer_results.append({
                "layer": layer_idx,
                "cosine": result["logit_cosine_mean"],
                "top1_match": result["top1_match_rate"],
                "top5_overlap": result["top5_overlap_mean"],
                "nll_delta": result.get("avg_nll_delta", 0.0),
                "kl": result["kl_divergence_mean"],
            })
        all_results.append({
            "config_name": config["name"],
            "num_layers": num_layers,
            "layers": layer_results,
        })

    # Build summary across all tested configs
    def _worst_n_by(key: str, layer_rows: list[dict], n: int = 4) -> list[int]:
        finite = [
            r for r in layer_rows
            if isinstance(r.get(key), float) and math.isfinite(r[key])
        ]
        if key in ("cosine", "top1_match", "top5_overlap"):
            finite.sort(key=lambda r: r[key])
        else:
            finite.sort(key=lambda r: abs(r[key]), reverse=True)
        return [r["layer"] for r in finite[:n]]

    def _unique_sorted(layers: list[int]) -> list[int]:
        return sorted(set(layers))

    all_worst_cosine: list[int] = []
    all_worst_kl: list[int] = []
    all_worst_nll: list[int] = []
    for res in all_results:
        rows = res["layers"]
        all_worst_cosine.extend(_worst_n_by("cosine", rows, 4))
        all_worst_kl.extend(_worst_n_by("kl", rows, 4))
        all_worst_nll.extend(_worst_n_by("nll_delta", rows, 4))

    worst_cosine = _unique_sorted(all_worst_cosine)[:4]
    worst_kl = _unique_sorted(all_worst_kl)[:4]
    worst_nll = _unique_sorted(all_worst_nll)[:4]
    recommended_protected = _unique_sorted(
        worst_cosine + worst_kl + worst_nll
    )[:4]

    sensitivity_summary = {
        "worst_layers_by_cosine": worst_cosine,
        "worst_layers_by_kl": worst_kl,
        "worst_layers_by_nll_delta": worst_nll,
        "recommended_protected_layers": recommended_protected,
    }

    payload: dict[str, Any] = {
        "release": "main28",
        "analysis": "per_layer_sensitivity",
        "model": model_id,
        "tokens_tested": tokens,
        "configs": all_results,
        "summary": sensitivity_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote per-layer sensitivity to {out_path}")
    return payload


def _run_targeted_layer_protection(
    model_id: str,
    tokens: int,
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
    sensitivity_path: Path | None = None,
) -> dict[str, Any]:
    """Test targeted layer protection sets derived from sensitivity analysis.

    Scenarios tested:
      - protect_first_4: keep layers 0-3 at fp16
      - protect_16: keep layer 16 at fp16
      - protect_16_20: keep layers 16, 20 at fp16
      - protect_16_20_21_23: keep layers 16, 20, 21, 23 at fp16
      - protect_worst_4: keep top-4 worst layers from sensitivity (if available)
    """
    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.to(device)
    model.eval()

    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :tokens].to(device)
    if input_ids.shape[1] < 67:
        raise ValueError(
            "Need at least 67 tokens for targeted layer protection "
            "(context + 65 decode tokens)"
        )
    context_ids = input_ids[:, :-65]
    decode_tokens = input_ids[:, -65:]

    with torch.no_grad():
        baseline_ctx = model(input_ids=context_ids, use_cache=True)
    baseline_legacy, baseline_cache_cls = _to_legacy_cache(
        baseline_ctx.past_key_values
    )
    baseline_past = _from_legacy_cache(baseline_legacy, baseline_cache_cls)
    baseline_logits_list, baseline_nll = _decode_nll_multi(
        model, baseline_past, decode_tokens, n_positions=64
    )

    num_layers = len(baseline_legacy)
    test_configs = [c for c in configs if c["name"] != "baseline_fp16"]
    if not test_configs:
        test_configs = [configs[0]]

    # Load worst-layer recommendations from sensitivity analysis if available
    worst_4_from_sensitivity: list[int] = [16, 20, 21, 23]  # safe default
    if sensitivity_path is not None and sensitivity_path.exists():
        try:
            sens = json.loads(sensitivity_path.read_text(encoding="utf-8"))
            rec = sens.get("summary", {}).get("recommended_protected_layers", [])
            if rec:
                worst_4_from_sensitivity = rec[:4]
        except Exception:
            pass

    # Build named protection scenarios
    all_scenario_layers = []
    for layer in worst_4_from_sensitivity:
        if layer < num_layers:
            all_scenario_layers.append(layer)

    scenarios: list[tuple[str, list[int]]] = [
        ("protect_first_4", list(range(min(4, num_layers)))),
        ("protect_16", [16] if 16 < num_layers else []),
        ("protect_16_20", [idx for idx in [16, 20] if idx < num_layers]),
        (
            "protect_16_20_21_23",
            [idx for idx in [16, 20, 21, 23] if idx < num_layers],
        ),
        ("protect_worst_4", all_scenario_layers),
    ]

    all_results: list[dict[str, Any]] = []
    for config in test_configs:
        print(f"Targeted layer protection: {config['name']} ...")
        protection_results: list[dict[str, Any]] = []
        for scenario_name, protect_layers in scenarios:
            if not protect_layers:
                continue
            protect_set = set(protect_layers)
            compress_set = set(range(num_layers)) - protect_set
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
                compress_layers=compress_set,
                n_decode_positions=64,
            )
            protection_results.append({
                "scenario": scenario_name,
                "protected_layers": protect_layers,
                "cosine": result["logit_cosine_mean"],
                "top1_match": result["top1_match_rate"],
                "top5_overlap": result["top5_overlap_mean"],
                "nll_delta": result.get("avg_nll_delta", 0.0),
                "kl": result["kl_divergence_mean"],
                "status": _determine_status(result, baseline_nll=baseline_nll),
            })
        all_results.append({
            "config_name": config["name"],
            "num_layers": num_layers,
            "scenarios": protection_results,
        })

    payload: dict[str, Any] = {
        "release": "main28",
        "analysis": "targeted_layer_protection",
        "model": model_id,
        "tokens_tested": tokens,
        "configs": all_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote early-layer protection to {out_path}")
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
            capture_output=True, text=True, check=False,
        )
        chip = result.stdout.strip() or "unknown"
    except Exception:
        pass

    ram_gb = 16
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        pass

    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "os": platform.system(),
        "mlx_version": mlx_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate KV quality against a real HuggingFace model"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID to validate against",
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
        help="Number of decode positions to evaluate (causal NLL scoring)",
    )
    parser.add_argument(
        "--configs",
        default="baseline_fp16,mixed_L0-1k8v4_restk6v4_gs64,"
        "k8_v4_gs64,k8_v5_gs64,k8_v3_gs64,"
        "k6_v6_gs64,k8_v4_gs32,k8_v5_gs32,k4_v4_gs64",
        help="Comma-separated config names to test",
    )
    parser.add_argument(
        "--contexts",
        default="",
        help="Comma-separated token counts for long-context validation "
        "(e.g., 512,1024,2048)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main28/real_model_validation.json",
        help="Output JSON path for real-model validation",
    )
    parser.add_argument(
        "--long-context-out",
        default="artifacts/proof/main28/long_context_validation.json",
        help="Output JSON path for long-context validation",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model from HuggingFace",
    )
    parser.add_argument(
        "--per-layer-sensitivity",
        action="store_true",
        help="Run per-layer sensitivity analysis",
    )
    parser.add_argument(
        "--per-layer-out",
        default="artifacts/proof/main28/per_layer_sensitivity.json",
        help="Output path for per-layer sensitivity",
    )
    parser.add_argument(
        "--early-layer-protection",
        action="store_true",
        help="Run early-layer higher-precision test",
    )
    parser.add_argument(
        "--early-layer-out",
        default="artifacts/proof/main28/early_layer_protection.json",
        help="Output path for early-layer protection",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [_parse_config(name) for name in config_names]

    # Real-model validation
    print(
        f"Running real-model validation: model={args.model}, "
        f"tokens={args.tokens}, positions={args.positions}"
    )
    payload = _run_real_model_validation(
        model_id=args.model,
        tokens=args.tokens,
        configs=configs,
        device=device,
        out_path=Path(args.out),
        trust_remote_code=args.trust_remote_code,
        n_decode_positions=args.positions,
    )
    exit_code = 0
    if any(c.get("status") == "fail" for c in payload.get("configs", [])):
        exit_code = 1

    # Long-context validation
    if args.contexts:
        context_tokens = [
            int(c.strip()) for c in args.contexts.split(",") if c.strip()
        ]
        _run_long_context_validation(
            model_id=args.model,
            contexts=context_tokens,
            configs=configs,
            device=device,
            out_path=Path(args.long_context_out),
            trust_remote_code=args.trust_remote_code,
            n_decode_positions=args.positions,
        )

    # Per-layer sensitivity analysis
    if args.per_layer_sensitivity:
        _run_per_layer_sensitivity(
            model_id=args.model,
            tokens=args.tokens,
            configs=configs,
            device=device,
            out_path=Path(args.per_layer_out),
            trust_remote_code=args.trust_remote_code,
        )

    # Targeted layer protection test
    if args.early_layer_protection:
        _run_targeted_layer_protection(
            model_id=args.model,
            tokens=args.tokens,
            configs=configs,
            device=device,
            out_path=Path(args.early_layer_out),
            trust_remote_code=args.trust_remote_code,
            sensitivity_path=Path(args.per_layer_out),
        )

    if exit_code != 0:
        print(
            "FAIL: one or more configs did not meet quality thresholds.",
            file=sys.stderr,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
