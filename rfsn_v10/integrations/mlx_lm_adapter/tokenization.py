"""Hermetic tokenization helpers for the MLX-LM adapter.

Provides consistent token counting that does not depend on the global
tokenizer state changing between calls.
"""
from __future__ import annotations

from typing import Any


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """Count tokens in a prompt string.

    Uses the tokenizer's encode method with add_special_tokens=False
    to get a consistent count.
    """
    try:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback: rough heuristic (not accurate, but safe)
        return len(prompt) // 4


def count_generation_tokens(tokenizer: Any, generated_ids: list[int]) -> int:
    """Count tokens in generated IDs.

    This is more accurate than re-encoding the text, as it counts
    the actual tokens produced by the generator.
    """
    return len(generated_ids)


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a list of token IDs to text.

    Handles the case where the tokenizer returns bytes or str.
    """
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return ""


def apply_chat_template_safe(tokenizer: Any, messages: list[dict]) -> str:
    """Apply chat template with graceful fallback.

    Returns the raw message if the tokenizer does not support
    chat templates.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        # Fallback: simple concatenation
        return "\n".join(m.get("content", "") for m in messages)

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback if template application fails
        return "\n".join(m.get("content", "") for m in messages)
