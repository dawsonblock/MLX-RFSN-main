"""Minimal benchmark-only decode loop.

Supports Qwen2.5 0.5B and 1.5B models with prefill, decode, custom KV cache,
logit capture, and token output. No global monkey-patching.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np


def minimal_decode_loop(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 200,
    temp: float = 0.0,
    custom_cache: Any | None = None,
    capture_logits: bool = False,
) -> dict[str, Any]:
    """Run a minimal prefill + decode loop.

    Parameters
    ----------
    model
        MLX model object.
    tokenizer
        MLX tokenizer object.
    prompt
        Input prompt string.
    max_tokens
        Maximum new tokens to generate.
    temp
        Sampling temperature (0.0 for greedy).
    custom_cache
        Optional custom KV cache object to inject.
    capture_logits
        If True, capture logits per step.

    Returns
    -------
    dict with keys:
        tokens, text, prefill_ms, decode_ms, total_ms, logits (optional)
    """
    try:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step
    except ImportError as exc:
        raise RuntimeError("mlx_lm is required for minimal_decode_loop") from exc

    input_ids = mx.array(tokenizer.encode(prompt))
    sampler = make_sampler(temp=temp)

    tokens: list[int] = []
    logits_list: list[np.ndarray] = []

    t0 = time.perf_counter()

    # Prefill
    if custom_cache is not None:
        # Use custom cache as prompt_cache for the first step
        first_token = None
        for token, logprobs in generate_step(
            prompt=input_ids,
            model=model,
            max_tokens=1,
            sampler=sampler,
            prompt_cache=custom_cache,
        ):
            first_token = int(token.item() if hasattr(token, "item") else token)
            if capture_logits and logprobs is not None:
                logits_list.append(np.array(logprobs))
            break
        prefill_ms = (time.perf_counter() - t0) * 1000
        if first_token is not None and first_token not in tokenizer.eos_token_ids:
            tokens.append(first_token)
    else:
        # Standard prefill without custom cache
        first_token = None
        for token, logprobs in generate_step(
            prompt=input_ids,
            model=model,
            max_tokens=1,
            sampler=sampler,
        ):
            first_token = int(token.item() if hasattr(token, "item") else token)
            if capture_logits and logprobs is not None:
                logits_list.append(np.array(logprobs))
            break
        prefill_ms = (time.perf_counter() - t0) * 1000
        if first_token is not None and first_token not in tokenizer.eos_token_ids:
            tokens.append(first_token)

    # Decode
    t_decode = time.perf_counter()
    for token, logprobs in generate_step(
        prompt=mx.array([tokens[-1]] if tokens else [0]),
        model=model,
        max_tokens=max_tokens - 1,
        sampler=sampler,
        prompt_cache=custom_cache,
    ):
        tok_id = int(token.item() if hasattr(token, "item") else token)
        if tok_id in tokenizer.eos_token_ids:
            break
        tokens.append(tok_id)
        if capture_logits and logprobs is not None:
            logits_list.append(np.array(logprobs))
        if len(tokens) >= max_tokens:
            break
    decode_ms = (time.perf_counter() - t_decode) * 1000
    total_ms = prefill_ms + decode_ms

    generated_text = tokenizer.decode(tokens)

    result: dict[str, Any] = {
        "tokens": tokens,
        "text": generated_text,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
    }
    if capture_logits:
        result["logits"] = logits_list
    return result
