#!/usr/bin/env python3
"""Production-grade real-model validation for RFSN v10.

Runs comprehensive validation across diverse tasks using the validation suite.
Supports batch evaluation across multiple prompts and models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(F.cosine_similarity(a_f, b_f, dim=0).item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


def _perplexity_for_target(logits: torch.Tensor, target_id: int) -> float:
    target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits.float(), target)
    return float(torch.exp(loss).item())


def _block_select_indices(
    seq_len: int,
    top_k_ratio: float,
    block_size: int = 64,
    reserved_sink_blocks: int = 1,
    reserved_recent_blocks: int = 2,
) -> torch.Tensor:
    num_blocks = max(1, math.ceil(seq_len / block_size))
    k_active = max(1, math.ceil(num_blocks * top_k_ratio))

    reserved = []
    for idx in range(min(reserved_sink_blocks, num_blocks)):
        reserved.append(idx)
    for offset in range(reserved_recent_blocks):
        idx = num_blocks - 1 - offset
        if idx >= 0 and idx not in reserved:
            reserved.append(idx)

    selected = []
    for idx in reserved:
        if idx not in selected:
            selected.append(idx)

    for idx in range(num_blocks - 1, -1, -1):
        if len(selected) >= k_active:
            break
        if idx not in selected:
            selected.append(idx)

    selected = sorted(set(selected))
    token_indices = []
    for b in selected:
        start = b * block_size
        end = min(seq_len, (b + 1) * block_size)
        token_indices.extend(range(start, end))

    if not token_indices:
        token_indices = [seq_len - 1]

    return torch.tensor(token_indices, dtype=torch.long)


def _compress_decompress_past(
    past_key_values,
    *,
    sparse: bool,
    top_k_ratio: float,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_idx, (k_t, v_t) in enumerate(past_key_values):
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()

        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 64.0) * 64)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = __import__("numpy").pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = __import__("numpy").pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))

        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        mgr = RFSNTurboQuantKVManager(
            k_bits=8,
            v_bits=3,
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=True,
            max_memory_gb=2.0,
        )

        key = f"layer_{layer_idx}"
        mgr.store(key, k_mx, v_mx, token_count=seq)
        rec = mgr.retrieve(key, out_dtype=mx.float32)
        if rec is None:
            raise RuntimeError("Unexpected cache miss during validation")
        rk_mx, rv_mx = rec
        rk = torch.tensor(rk_mx.tolist(), dtype=torch.float32)
        rv = torch.tensor(rv_mx.tolist(), dtype=torch.float32)

        rk = rk[..., :dim]
        rv = rv[..., :dim]

        if sparse:
            idx = _block_select_indices(seq, top_k_ratio=top_k_ratio)
            rk = rk.index_select(2, idx)
            rv = rv.index_select(2, idx)

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


def _evaluate_prompt(
    model,
    tokenizer,
    prompt: str,
    prompt_id: str,
    category: str,
    max_input_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a single prompt across all modes."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][:, : max_input_tokens]
    if input_ids.shape[1] < 2:
        return {
            "prompt_id": prompt_id,
            "category": category,
            "status": "skipped",
            "reason": "prompt_too_short",
        }

    input_ids = input_ids.to(device)
    context_ids = input_ids[:, :-1]
    decode_token = input_ids[:, -1:]

    # Baseline
    with torch.no_grad():
        baseline_ctx = model(input_ids=context_ids, use_cache=True)
        baseline_out = model(
            input_ids=decode_token,
            past_key_values=baseline_ctx.past_key_values,
            use_cache=True,
        )

    baseline_logits = baseline_out.logits[:, -1, :]
    baseline_target = int(decode_token[0, 0].item())
    baseline_ppl = _perplexity_for_target(baseline_logits, baseline_target)

    results = []

    # Compressed KV (dense)
    past_raw = model(input_ids=context_ids, use_cache=True).past_key_values
    past_legacy, cache_cls = _to_legacy_cache(past_raw)
    past_compressed = _compress_decompress_past(
        past_legacy, sparse=False, top_k_ratio=1.0, device=device
    )
    past = _from_legacy_cache(past_compressed, cache_cls)

    with torch.no_grad():
        out = model(input_ids=decode_token, past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]
    ppl = _perplexity_for_target(logits, baseline_target)

    results.append(
        {
            "mode": "compressed_kv_dense",
            "logit_cosine": _cosine(logits, baseline_logits),
            "logit_max_abs_diff": float(torch.max(torch.abs(logits - baseline_logits)).item()),
            "top1_token_match_rate": float(
                torch.argmax(logits, dim=-1).item() == torch.argmax(baseline_logits, dim=-1).item()
            ),
            "top5_overlap": _topk_overlap(logits, baseline_logits, k=5),
            "perplexity_delta": float(ppl - baseline_ppl),
        }
    )

    # Compressed KV (sparse)
    past_compressed_sparse = _compress_decompress_past(
        past_legacy, sparse=True, top_k_ratio=0.5, device=device
    )
    past_sparse = _from_legacy_cache(past_compressed_sparse, cache_cls)

    with torch.no_grad():
        out_sparse = model(input_ids=decode_token, past_key_values=past_sparse, use_cache=True)
    logits_sparse = out_sparse.logits[:, -1, :]
    ppl_sparse = _perplexity_for_target(logits_sparse, baseline_target)

    results.append(
        {
            "mode": "compressed_kv_sparse",
            "logit_cosine": _cosine(logits_sparse, baseline_logits),
            "logit_max_abs_diff": float(
                torch.max(torch.abs(logits_sparse - baseline_logits)).item()
            ),
            "top1_token_match_rate": float(
                torch.argmax(logits_sparse, dim=-1).item()
                == torch.argmax(baseline_logits, dim=-1).item()
            ),
            "top5_overlap": _topk_overlap(logits_sparse, baseline_logits, k=5),
            "perplexity_delta": float(ppl_sparse - baseline_ppl),
        }
    )

    return {
        "prompt_id": prompt_id,
        "category": category,
        "status": "success",
        "tokens_tested": int(input_ids.shape[1]),
        "baseline_perplexity": float(baseline_ppl),
        "mode_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-grade model validation")
    parser.add_argument("--model-path", required=True, help="Path to downloaded model")
    parser.add_argument(
        "--prompt-suite",
        default="prompts/validation_suite.json",
        help="Path to validation suite JSON",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main12/production_validation.json",
        help="Output JSON path",
    )
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to validate (default: all)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    prompt_suite_path = Path(args.prompt_suite)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        return

    if not prompt_suite_path.exists():
        print(f"ERROR: Prompt suite not found: {prompt_suite_path}")
        return

    # Load prompt suite
    with open(prompt_suite_path) as f:
        suite = json.load(f)

    prompts = suite["prompts"]
    if args.categories:
        prompts = [p for p in prompts if p["category"] in args.categories]

    print(f"Loading model from: {model_path}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path.as_posix(),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Run validation
    all_results = []
    category_stats: dict[str, dict] = {}

    for prompt_data in prompts:
        print(f"\nValidating: {prompt_data['id']} ({prompt_data['category']})")
        result = _evaluate_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_data["prompt"],
            prompt_id=prompt_data["id"],
            category=prompt_data["category"],
            max_input_tokens=args.max_input_tokens,
            device=device,
        )
        all_results.append(result)

        if result["status"] == "success":
            category = prompt_data["category"]
            if category not in category_stats:
                category_stats[category] = {
                    "count": 0,
                    "dense_cosine_sum": 0.0,
                    "sparse_cosine_sum": 0.0,
                    "dense_ppl_delta_sum": 0.0,
                    "sparse_ppl_delta_sum": 0.0,
                }
            stats = category_stats[category]
            stats["count"] += 1
            stats["dense_cosine_sum"] += result["mode_results"][0]["logit_cosine"]
            stats["sparse_cosine_sum"] += result["mode_results"][1]["logit_cosine"]
            stats["dense_ppl_delta_sum"] += result["mode_results"][0]["perplexity_delta"]
            stats["sparse_ppl_delta_sum"] += result["mode_results"][1]["perplexity_delta"]

    # Compute category averages
    for category, stats in category_stats.items():
        count = stats["count"]
        stats["avg_dense_cosine"] = stats["dense_cosine_sum"] / count
        stats["avg_sparse_cosine"] = stats["sparse_cosine_sum"] / count
        stats["avg_dense_ppl_delta"] = stats["dense_ppl_delta_sum"] / count
        stats["avg_sparse_ppl_delta"] = stats["sparse_ppl_delta_sum"] / count

    # Overall statistics
    successful = [r for r in all_results if r["status"] == "success"]
    overall_dense_cosine = sum(
        r["mode_results"][0]["logit_cosine"] for r in successful
    ) / len(successful) if successful else 0.0
    overall_sparse_cosine = sum(
        r["mode_results"][1]["logit_cosine"] for r in successful
    ) / len(successful) if successful else 0.0

    payload = {
        "status": "success",
        "model": model_path.as_posix(),
        "device": str(device),
        "prompt_suite": str(prompt_suite_path),
        "prompts_tested": len(all_results),
        "prompts_successful": len(successful),
        "overall_dense_cosine": overall_dense_cosine,
        "overall_sparse_cosine": overall_sparse_cosine,
        "category_statistics": category_stats,
        "results": all_results,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path}")
    print(f"Overall dense cosine: {overall_dense_cosine:.4f}")
    print(f"Overall sparse cosine: {overall_sparse_cosine:.4f}")
    print(f"Success rate: {len(successful)}/{len(all_results)}")


if __name__ == "__main__":
    main()
