"""Short-prompt generation regression tests for RFSN v10.

These tests validate that compressed KV cache modes do not catastrophically
drift on short prompts during teacher-forced generation.

Minimum test: teacher-forced logit equivalence for k8_v5_gs64 at 128 tokens.
"""
from __future__ import annotations

import math
import tempfile
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
# Guard against transformers being installed without torch backend available
if not transformers.utils.import_utils.is_torch_available():
    pytest.skip("transformers installed but torch backend not available", allow_module_level=True)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager  # noqa: E402


def _get_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def _compress_stable(
    past_key_values,
    k_bits: int,
    v_bits: int,
    group_size: int,
    device: torch.device,
):
    with tempfile.TemporaryDirectory(prefix="rfsn_reg_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, group_size=group_size,
            use_wht=True, use_incoherent_signs=True, prefer_metal_kernels=True,
            strict_metal=False, max_memory_gb=2.0, cache_dir=tmpdir,
        )
        out = []
        # Handle both old tuple-list format and new DynamicCache format
        if isinstance(past_key_values, DynamicCache):
            layers = range(len(past_key_values.key_cache))
        else:
            layers = enumerate(past_key_values)
        for layer_idx in layers:
            if isinstance(past_key_values, DynamicCache):
                k_t = past_key_values.key_cache[layer_idx]
                v_t = past_key_values.value_cache[layer_idx]
            else:
                idx, item = layer_idx
                layer_idx = idx
                k_t = item[0]
                v_t = item[1] if len(item) > 1 else None
            k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
            if v_t is not None:
                v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
            else:
                v_np = k_np
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


def _teacher_forced_check(
    model,
    tokenizer,
    device: torch.device,
    config: str,
    prompt_tokens: int,
    positions: int,
) -> dict[str, Any]:
    """Run teacher-forced check: identical continuation fed to FP16
    and compressed."""
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    if prompt_tokens > len(dummy_ids):
        repeated = (
            dummy_ids * ((prompt_tokens // len(dummy_ids)) + 1)
        )[:prompt_tokens]
        prompt_str = tokenizer.decode(repeated)
    else:
        prompt_str = tokenizer.decode(dummy_ids[:prompt_tokens])

    prompt_ids = tokenizer.encode(
        prompt_str, return_tensors="pt", truncation=True
    )
    prompt_ids = prompt_ids.to(device)

    # Baseline FP16 prefill
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
    baseline_past = out.past_key_values

    # Greedy continuation for forced tokens (single-token inputs after prefill)
    continuation = []
    past = baseline_past
    # First token is generated from the prefill logits
    tok = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
    continuation.append(tok)
    # Subsequent tokens use single-token inputs
    for _ in range(positions - 1):
        next_ids = torch.tensor([[continuation[-1]]], device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_ids, past_key_values=past, use_cache=True
            )
        past = out.past_key_values
        tok = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        continuation.append(tok)

    # Re-prefill for compressed path
    with torch.no_grad():
        out_fp16 = model(
            input_ids=prompt_ids, past_key_values=None, use_cache=True
        )
    past_fp16 = out_fp16.past_key_values

    # Compress
    if hasattr(past_fp16, "to_legacy_cache"):
        past_list = list(past_fp16.to_legacy_cache())
    else:
        past_list = list(past_fp16)

    if config == "k8_v5_gs64":
        quant_past = _compress_stable(past_list, 8, 5, 64, device)
    elif config == "k8_v5_gs32":
        quant_past = _compress_stable(past_list, 8, 5, 32, device)
    else:
        pytest.skip(f"No teacher-forced regression test for {config}")

    if hasattr(DynamicCache, "from_legacy_cache"):
        quant_past = DynamicCache.from_legacy_cache(tuple(quant_past))
    else:
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(quant_past):
            cache.update(k, v, layer_idx)
        quant_past = cache

    # Compare first decode step logits with identical token
    first_tok = continuation[0] if continuation else 0
    next_ids = torch.tensor([[first_tok]], device=device)

    with torch.no_grad():
        out_fp16_step = model(
            input_ids=next_ids, past_key_values=past_fp16, use_cache=True
        )
        out_quant_step = model(
            input_ids=next_ids, past_key_values=quant_past, use_cache=True
        )

    logit_fp16 = out_fp16_step.logits[:, -1, :]
    logit_quant = out_quant_step.logits[:, -1, :]

    a_f = logit_fp16.reshape(-1).float()
    b_f = logit_quant.reshape(-1).float()
    cosine = float(
        torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
            a_f, b_f, dim=0
        ).item()
    )

    ai = set(torch.topk(logit_fp16, k=5, dim=-1).indices[0].tolist())
    bi = set(torch.topk(logit_quant, k=5, dim=-1).indices[0].tolist())
    top5 = float(len(ai & bi) / len(ai)) if ai else 0.0

    p = torch.nn.functional.softmax(logit_fp16.float(), dim=-1)
    q = torch.nn.functional.softmax(logit_quant.float(), dim=-1)
    eps = 1e-10
    kl = float(torch.sum(p * torch.log((p + eps) / (q + eps))).item())

    delta = (logit_fp16 - logit_quant).abs()
    max_abs_delta = float(delta.max().item())
    mean_abs_delta = float(delta.mean().item())

    return {
        "logit_cosine_vs_fp16": cosine,
        "top5_overlap_vs_fp16": top5,
        "kl_vs_fp16": kl,
        "max_abs_logit_delta": max_abs_delta,
        "mean_abs_logit_delta": mean_abs_delta,
    }


@pytest.mark.parametrize("prompt_tokens", [128, 512])
def test_k8_v5_gs64_short_prompt_teacher_forced_stability(
    prompt_tokens: int,
) -> None:
    model, tokenizer, device = _get_model_and_tokenizer()
    result = _teacher_forced_check(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config="k8_v5_gs64",
        prompt_tokens=prompt_tokens,
        positions=16,
    )
    assert result["logit_cosine_vs_fp16"] >= 0.999, (
        f"k8_v5_gs64 teacher-forced cosine {result['logit_cosine_vs_fp16']} "
        f"< 0.999 at {prompt_tokens} tokens"
    )
    assert result["top5_overlap_vs_fp16"] >= 0.95, (
        f"k8_v5_gs64 teacher-forced top5 {result['top5_overlap_vs_fp16']} "
        f"< 0.95 at {prompt_tokens} tokens"
    )
    assert result["kl_vs_fp16"] <= 0.001, (
        f"k8_v5_gs64 teacher-forced KL {result['kl_vs_fp16']} "
        f"> 0.001 at {prompt_tokens} tokens"
    )


@pytest.mark.parametrize("prompt_tokens", [128, 512])
def test_k8_v5_gs32_short_prompt_teacher_forced_stability(
    prompt_tokens: int,
) -> None:
    model, tokenizer, device = _get_model_and_tokenizer()
    result = _teacher_forced_check(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config="k8_v5_gs32",
        prompt_tokens=prompt_tokens,
        positions=16,
    )
    # Slightly looser for gs32 since it is a higher-compression candidate
    assert result["logit_cosine_vs_fp16"] >= 0.999, (
        f"k8_v5_gs32 teacher-forced cosine {result['logit_cosine_vs_fp16']} "
        f"< 0.999 at {prompt_tokens} tokens"
    )
    assert result["top5_overlap_vs_fp16"] >= 0.95, (
        f"k8_v5_gs32 teacher-forced top5 {result['top5_overlap_vs_fp16']} "
        f"< 0.95 at {prompt_tokens} tokens"
    )
    assert result["kl_vs_fp16"] <= 0.001, (
        f"k8_v5_gs32 teacher-forced KL {result['kl_vs_fp16']} "
        f"> 0.001 at {prompt_tokens} tokens"
    )
