"""RFSN v10 production generation loop.

Provides ``RFSNGenerator`` — a high-level inference interface that wraps
a loaded model and tokenizer with:

- Prefill (dense causal attention for the initial prompt)
- Decode loop (streaming token generation)
- Explicit per-layer quantized KV cache via ``RfsnMLXReferenceAdapter``
- Temperature / top-p / repetition-penalty sampling
- Telemetry / proof counters per generation

The generator is backend-agnostic: it works with ``mlx-lm`` models on
Apple Silicon or ``transformers`` models on any platform.

**No global monkeypatching.**  The MLX path creates one
``RfsnQuantizedKVCache`` per transformer layer and passes them to
``mlx_lm.utils.stream_generate`` via ``prompt_cache``.  This avoids
process-global SDPA mutation and is safe for concurrent serving.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..config import RFSNConfig, load_config

try:
    from ..compat import mx
except ImportError:
    mx = None  # type: ignore[assignment]


try:
    from mlx_lm.utils import generate as _mlx_generate
    from mlx_lm.utils import stream_generate as _mlx_stream_generate
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    _mlx_generate = None  # type: ignore[assignment]
    _mlx_stream_generate = None  # type: ignore[assignment]


try:
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass
class GenerationResult:
    """Result of a single generation request."""

    text: str
    tokens: list[int]
    generation_time_ms: float
    tokens_per_second: float
    telemetry: list[dict] = field(default_factory=list)
    decode_token_count: int = 0
    finish_reason: str = ""


class RFSNGenerator:
    """High-level inference generator with explicit per-layer quantized KV.

    Usage (MLX) ::

        from rfsn_v10.model_loader import load_mlx_model
        from rfsn_v10.runtime.generation import RFSNGenerator

        model, tokenizer = load_mlx_model(
            "mlx-community/Llama-3-8B-Instruct-4bit"
        )
        gen = RFSNGenerator(model=model, tokenizer=tokenizer)
        result = gen.chat("Hello, world!")
        print(result.text)

    Usage (streaming) ::

        for token in gen.generate("Hello", stream=True):
            print(token, end="")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: RFSNConfig | None = None,
        enable_quantized_kv: bool = True,
        key_bits: int = 8,
        value_bits: int = 5,
        group_size: int = 64,
        staging_capacity: int = 64,
        dense_residual_window: int = 0,
        packed_reference: bool = False,
        # Deprecated no-ops (kept for backward compatibility)
        enable_sparse_decode: bool = False,
        audit_mode: bool = False,
        use_compressed_on_miss: bool = False,
        kv_manager: Any | None = None,
    ):
        """
        Args:
            model: Loaded model (``mlx-lm`` or ``transformers``).
            tokenizer: Matching tokenizer.
            config: RFSN runtime configuration.  Loaded from env when ``None``.
            enable_quantized_kv: Whether to use quantized KV-cache.
            key_bits: Quantization bits for keys.
            value_bits: Quantization bits for values.
            group_size: Group size for symmetric quantization.
            staging_capacity: Tokens accumulated before encoding a
                sealed block.
            dense_residual_window: Keep last N tokens in dense FP16
                (0 disables).
            packed_reference: If True, use the direct packed-attention wrapper
                instead of dense-reconstruction fallback.  This bypasses the
                model's native attention and runs blockwise quantized attention
                directly.  Only effective when ``enable_quantized_kv=True``.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or load_config()
        self.enable_quantized_kv = enable_quantized_kv
        self.packed_reference = packed_reference

        # Direct packed Metal currently requires K8/V8 GS64.
        if packed_reference and (
            key_bits != 8 or value_bits != 8 or group_size != 64
        ):
            raise ValueError(
                "Direct packed generation currently requires K8/V8 GS64; "
                f"got K{key_bits}/V{value_bits} GS{group_size}"
            )

        self._adapter = None
        if MLX_LM_AVAILABLE and enable_quantized_kv:
            from ..integrations.mlx_lm_adapter.adapter import RfsnMLXReferenceAdapter
            self._adapter = RfsnMLXReferenceAdapter(
                model=model,
                tokenizer=tokenizer,
                key_bits=key_bits,
                value_bits=value_bits,
                group_size=group_size,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
                strict=self.config.runtime.strict_packed_mode,
                use_direct_packed=packed_reference,
            )

        self._telemetry_log: list[dict] = []
        self._last_counters: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response to a single user message.

        Args:
            message: User message text.
            system_prompt: Optional system prompt prepended to the message.
            **gen_kwargs: Overrides for :class:`GenerationConfig` fields.

        Returns:
            :class:`GenerationResult` with full text and metadata.
        """
        prompt = self._build_chat_prompt(message, system_prompt)
        return self._generate_sync(prompt, **gen_kwargs)

    def generate(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> Iterator[str]:
        """Generate text from a raw prompt, yielding tokens as strings.

        Args:
            prompt: Raw prompt string.
            **gen_kwargs: Overrides for :class:`GenerationConfig` fields.

        Yields:
            Decoded token strings (one per yield).
        """
        cfg = self._make_gen_config(**gen_kwargs)
        if MLX_LM_AVAILABLE and hasattr(self.model, "__call__"):
            yield from self._stream_mlx(prompt, cfg)
        else:
            raise RuntimeError(
                "Streaming generation requires mlx-lm.  "
                "Install with: pip install mlx-lm"
            )

    async def generate_async(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming variant of :meth:`generate`.

        Yields:
            Decoded token strings.
        """
        for token in self.generate(prompt, **gen_kwargs):
            yield token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_prompt(
        self,
        message: str,
        system_prompt: str | None = None,
    ) -> str:
        """Build a chat prompt using the tokenizer's chat template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: simple concatenation
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(message)
        return "\n".join(parts)

    def _make_gen_config(self, **overrides: Any) -> GenerationConfig:
        """Build a :class:`GenerationConfig` from defaults + overrides."""
        defaults = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "stream": True,
        }
        defaults.update(overrides)
        return GenerationConfig(**defaults)

    def _generate_sync(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Run synchronous generation and return the full result."""
        cfg = self._make_gen_config(stream=False, **kwargs)
        t_start = time.monotonic()
        tokens: list[int] = []
        telemetry: list[dict] = []

        if MLX_LM_AVAILABLE:
            text, tokens = self._generate_mlx_collect(prompt, cfg)
            telemetry = self.get_telemetry()
        elif TRANSFORMERS_AVAILABLE:
            text = self._generate_torch(prompt, cfg)
        else:
            raise RuntimeError(
                "No generation backend available.  "
                "Install mlx-lm (Apple Silicon) or transformers."
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        tps = len(tokens) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        return GenerationResult(
            text=text,
            tokens=tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tps,
            telemetry=telemetry,
        )

    def _generate_mlx_collect(
        self, prompt: str, cfg: GenerationConfig
    ) -> tuple[str, list[int]]:
        """Generate via ``mlx_lm`` stream and collect tokens."""
        text = ""
        tokens: list[int] = []
        for response in self._mlx_gen_iter(prompt, cfg):
            text += response.text
            tokens.append(response.token)
        return text, tokens

    def _generate_mlx(self, prompt: str, cfg: GenerationConfig) -> str:
        """Generate via ``mlx_lm`` (non-streaming)."""
        text, _tokens = self._generate_mlx_collect(prompt, cfg)
        return text

    def _generate_torch(self, prompt: str, cfg: GenerationConfig) -> str:
        """Generate via ``transformers`` pipeline."""
        assert TRANSFORMERS_AVAILABLE
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "to"):
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=cfg.temperature > 0,
        )
        generated = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _stream_mlx(self, prompt: str, cfg: GenerationConfig) -> Iterator[str]:
        """Stream generation via ``mlx_lm``, yielding individual tokens."""
        for response in self._mlx_gen_iter(prompt, cfg):
            yield response.text

    def _mlx_gen_iter(self, prompt: str, cfg: GenerationConfig):
        """Yield ``GenerationResponse`` from ``mlx_lm``."""
        assert MLX_LM_AVAILABLE and _mlx_stream_generate is not None

        gen_kwargs = dict(
            max_tokens=cfg.max_new_tokens,
            temp=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        )

        if self._adapter is not None and self.enable_quantized_kv:
            if self.packed_reference:
                # Direct packed-attention path — intercept attention modules.
                from rfsn_v10.integrations.mlx_lm_model_support import (
                    attention_wrapper,
                )
                _aw = attention_wrapper
                RfsnDirectPackedKVCache = _aw.RfsnDirectPackedKVCache
                packed_attention_context = _aw.packed_attention_context

                # Create session for direct packed path
                session = self._adapter._new_session()
                caches = [
                    RfsnDirectPackedKVCache(
                        layer_id=i,
                        key_codec=self._adapter.key_codec,
                        value_codec=self._adapter.value_codec,
                        staging_capacity=self._adapter.staging_capacity,
                        dense_residual_window=(
                            self._adapter.dense_residual_window
                        ),
                        strict=self.config.runtime.strict_packed_mode,
                        session=session,
                    )
                    for i in range(self._adapter.num_layers)
                ]

                # P0 #3: Use lifecycle management to ensure model is unwrapped
                strict_mode = (
                    self.config.runtime.strict_packed_mode
                    if self.config else True
                )

                # Import backend stats collection for execution tracking
                try:
                    from ..integrations.mlx_lm_model_support import (
                        attention_wrapper as _aw2,
                    )
                    collect_backend_stats = _aw2.collect_backend_stats
                except ImportError:
                    collect_backend_stats = None  # type: ignore

                with packed_attention_context(
                    self.model, caches, strict=strict_mode
                ):
                    gen_iter = _mlx_stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        prompt_cache=caches,
                        **gen_kwargs,
                    )
                    try:
                        yield from gen_iter
                    finally:
                        if caches is not None:
                            # Collect backend stats BEFORE context exits
                            backend_stats = []
                            if collect_backend_stats:
                                backend_stats = collect_backend_stats(
                                    self.model
                                )

                            # Capture memory report BEFORE session destruction
                            memory_report_dict = {}
                            if session:
                                try:
                                    mr = session.memory_report()
                                    memory_report_dict = mr.to_dict()
                                except Exception:
                                    pass  # Best-effort

                            # Flatten generator counter output
                            runtime_dict = (
                                session.runtime_counters.to_dict()
                                if session else {}
                            )
                            self._last_counters = {
                                "direct_packed_tokens": sum(
                                    c.layer_cache.total_token_count()
                                    for c in caches
                                ) // self._adapter.num_layers,
                                **runtime_dict,
                                "backend_stats": backend_stats,
                                "execution_backend": (
                                    self._derive_execution_backend(
                                        backend_stats
                                    )
                                ),
                                "memory_report": memory_report_dict,
                                "_last_memory_report": (
                                    memory_report_dict
                                ),
                            }
                            # Cleanup session
                            if session:
                                session.destroy()
                return

            # Explicit per-layer cache path — dense reconstruction fallback.
            from ..integrations.mlx_lm_adapter.adapter import (
                RfsnQuantizedKVCache,
            )

            session = self._adapter._new_session()
            cache_list = [
                RfsnQuantizedKVCache(
                    layer_cache=session.get_layer_cache(i),
                    session=session,
                )
                for i in range(self._adapter.num_layers)
            ]
            gen_iter = _mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                prompt_cache=cache_list,
                **gen_kwargs,
            )
            try:
                yield from gen_iter
            finally:
                # P0 Fix: Capture memory report before session destruction
                memory_report_dict = {}
                try:
                    memory_report = session.memory_report()
                    memory_report_dict = memory_report.to_dict()
                except Exception:
                    pass  # Memory report is best-effort

                self._last_counters = {
                    **session.counters(),
                    "memory_report": memory_report_dict,
                    "_last_memory_report": memory_report_dict,
                    "execution_backend": (
                        "DENSE_RECONSTRUCTED"
                    ),  # dense fallback path
                }
                session.destroy()
        else:
            # Plain dense path
            gen_iter = _mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                **gen_kwargs,
            )
            yield from gen_iter

    # ------------------------------------------------------------------
    # Telemetry / proof counters
    # ------------------------------------------------------------------

    def get_telemetry(self) -> list[dict]:
        """Return accumulated proof counters from the last generation."""
        counters = getattr(self, "_last_counters", {})
        if counters:
            return [dict(counters)]
        return []

    def clear_telemetry(self) -> None:
        """Clear telemetry / counters."""
        self._last_counters = {}

    def _derive_execution_backend(self, backend_stats: list[dict]) -> str:
        """Derive single authoritative execution backend from layer stats.

        P0 Fix: Returns one of:
        - PACKED_MLX_REFERENCE: All layers used packed reference
        - METAL_DENSE_RECONSTRUCTED: Any layer used Metal dense reconstruction
        - DENSE_FALLBACK: Any layer fell back to dense
        - MIXED_INVALID: Inconsistent backends across layers
        - UNKNOWN: No backend stats available
        """
        if not backend_stats:
            return "UNKNOWN"

        backends = [
            s.get("executed_backend", "unknown") for s in backend_stats
        ]

        # Check for any Metal dense reconstruction (violation of invariant)
        metal_dense = [
            b for b in backends if "metal_dense_reconstruction" in b
        ]
        if metal_dense:
            return "METAL_DENSE_RECONSTRUCTED"

        # Check for fallback
        fallbacks = [
            b for b in backends if b == "dense" or "fallback" in b
        ]
        if fallbacks:
            return "DENSE_FALLBACK"

        # Check for true packed Metal (MLX inline or standalone)
        metal_packed = [
            b for b in backends if "true_packed_metal" in b
        ]
        if metal_packed and len(metal_packed) == len(backends):
            return "TRUE_PACKED_METAL"

        # Check for packed reference
        packed_ref = [
            b for b in backends if b in ("packed_reference", "packed")
        ]
        if packed_ref and len(packed_ref) == len(backends):
            return "PACKED_MLX_REFERENCE"

        # Mixed or unknown
        if all(b == "unknown" for b in backends):
            return "UNKNOWN"

        return "MIXED_INVALID"
