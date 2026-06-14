"""Generator wrapper that exposes the RFSNGenerator interface over RfsnMLXReferenceAdapter.

This allows the server to use the new adapter without changing its call sites.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnMLXReferenceAdapter
from rfsn_v10.integrations.mlx_lm_adapter.tokenization import (
    apply_chat_template_safe,
    count_generation_tokens,
    count_prompt_tokens,
)

try:
    import mlx_lm  # noqa: F401
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class AdapterAvailability:
    requested: bool
    dependency_available: bool
    active: bool
    reason: str


class RfsnMLXGenerator:
    """Drop-in replacement for RFSNGenerator using the reference adapter.

    Provides:
      - chat(prompt, ...) -> GenerationResult-like object
      - generate(prompt, ...) -> Iterator[str] (token strings)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        num_layers: int | None = None,
        key_bits: int = 8,
        value_bits: int = 5,
        group_size: int = 64,
        staging_capacity: int = 64,
        dense_residual_window: int = 0,
        packed_reference: bool = False,
        strict: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.adapter: RfsnMLXReferenceAdapter | None = None

        if not MLX_LM_AVAILABLE:
            self.adapter_availability = AdapterAvailability(
                requested=True,
                dependency_available=False,
                active=False,
                reason="MLX-LM is unavailable on this platform",
            )
        else:
            self.adapter = RfsnMLXReferenceAdapter(
                model=model,
                tokenizer=tokenizer,
                num_layers=num_layers,
                key_bits=key_bits,
                value_bits=value_bits,
                group_size=group_size,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
                strict=strict,
                use_direct_packed=packed_reference,
            )
            self.adapter_availability = AdapterAvailability(
                requested=True,
                dependency_available=True,
                active=True,
                reason="reference adapter active",
            )

    def chat(
        self,
        message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a chat response."""
        if self.adapter is None:
            raise RuntimeError(
                f"Adapter unavailable: {self.adapter_availability.reason}"
            )

        t_start = time.monotonic()

        # Build prompt using hermetic template helper
        messages = [{"role": "user", "content": message}]
        prompt = apply_chat_template_safe(self.tokenizer, messages)

        # Count prompt tokens accurately
        prompt_tokens = count_prompt_tokens(self.tokenizer, prompt)

        text = self.adapter.generate(
            prompt,
            max_tokens=max_new_tokens,
            verbose=False,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in text:
                    text = text.split(seq)[0]
                    break

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Create a result object matching GenerationResult interface
        # Count generation tokens by decoding the text (most reliable cross-tokenizer)
        tokens = []
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            pass

        # Decode TPS uses actual generation tokens, not prompt+decode
        decode_tokens = count_generation_tokens(self.tokenizer, tokens)
        tps = decode_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        return _SimpleResult(
            text=text,
            tokens=tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tps,
            prompt_token_count=prompt_tokens,
            decode_token_count=decode_tokens,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Generate text, yielding token strings."""
        if self.adapter is None:
            raise RuntimeError(
                f"Adapter unavailable: {self.adapter_availability.reason}"
            )

        for token, _ in self.adapter.generate_step(
            prompt,
            max_tokens=max_new_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            token_text = self.tokenizer.decode([token.item()])

            # Check stop sequences
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in token_text:
                        token_text = token_text.split(seq)[0]
                        if token_text:
                            yield token_text
                        return

            yield token_text


class _SimpleResult:
    """Lightweight result object matching GenerationResult interface."""

    def __init__(
        self,
        text: str,
        tokens: list[int],
        generation_time_ms: float,
        tokens_per_second: float,
        prompt_token_count: int = 0,
        decode_token_count: int = 0,
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.generation_time_ms = generation_time_ms
        self.tokens_per_second = tokens_per_second
        self.prompt_token_count = prompt_token_count
        self.decode_token_count = decode_token_count
        self.finish_reason = "stop"
        self.stopped_on = None
        self.telemetry = []
