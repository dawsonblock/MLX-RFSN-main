"""RFSN v10 production generation loop.

Provides ``RFSNGenerator`` — a high-level inference interface that wraps
a loaded model and tokenizer with:

- Prefill (dense causal attention for the initial prompt)
- Decode loop (streaming token generation)
- RFSNRuntime integration hooks for KV-cache + sparse attention
- Temperature / top-p / repetition-penalty sampling
- Telemetry collection per decode step

The generator is backend-agnostic: it works with ``mlx-lm`` models on
Apple Silicon or ``transformers`` models on any platform.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from ..config import RFSNConfig, load_config
from ..kv_manager import RFSNTurboQuantKVManager


# Thread-local storage for RFSNRuntime SDPA patching context.
_rfsn_thread_local = threading.local()


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


def _rfsn_sdpa_wrapper(
    original_sdpa,
    queries,
    keys,
    values,
    cache,
    scale,
    mask,
    sinks=None,
):
    """Intercept SDPA for decode steps and route through RFSNRuntime when active."""
    runtime = getattr(_rfsn_thread_local, "runtime", None)
    layer_id = getattr(_rfsn_thread_local, "layer_id", "unknown")
    # Only intercept single-token decode steps with an active runtime.
    if (
        runtime is not None
        and cache is not None
        and queries is not None
        and queries.ndim == 4
        and queries.shape[2] == 1
        and queries.shape[1] == keys.shape[1]  # same head count (no GQA)
    ):
        try:
            output, _info = runtime.execute_decode_step(
                skill_pattern="decode",
                layer_id=layer_id,
                batch_id="batch_0",
                queries=queries,
                keys=keys,
                values=values,
            )
            return output
        except Exception:
            # Telemetry / audit failures should not crash generation.
            pass
    # Fallback to original SDPA.
    if sinks is not None:
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)
    return original_sdpa(queries, keys, values, cache, scale, mask)


class _RFSNSDPAPatcher:
    """Context manager that patches mlx_lm SDPA for RFSNRuntime decode steps."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._original: Any = None

    def __enter__(self):
        try:
            import mlx_lm.models.base as base_module

            self._original = base_module.scaled_dot_product_attention
            original = self._original

            def _patched(queries, keys, values, cache, scale, mask, sinks=None):
                return _rfsn_sdpa_wrapper(
                    original, queries, keys, values, cache, scale, mask, sinks
                )

            base_module.scaled_dot_product_attention = _patched
            _rfsn_thread_local.runtime = self.runtime
        except Exception:
            # If patching fails, silently degrade to upstream path.
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._original is not None:
                import mlx_lm.models.base as base_module

                base_module.scaled_dot_product_attention = self._original
        except Exception:
            pass
        _rfsn_thread_local.runtime = None
        return False


def _wrap_layers_for_rfsn(model: Any) -> None:
    """Wrap model attention layers to set layer_id before each forward."""
    inner = model
    if hasattr(model, "model"):
        inner = model.model
    if not hasattr(inner, "layers"):
        return
    for idx, layer in enumerate(inner.layers):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if hasattr(attn, "_rfsn_original_call"):
            continue
        original = attn.__call__
        attn._rfsn_original_call = original

        def _make_wrapper(orig, lid):
            def wrapper(x, mask=None, cache=None):
                old = getattr(_rfsn_thread_local, "layer_id", None)
                _rfsn_thread_local.layer_id = lid
                try:
                    return orig(x, mask, cache)
                finally:
                    _rfsn_thread_local.layer_id = old

            return wrapper

        attn.__call__ = _make_wrapper(original, f"layer_{idx}")


def _unwrap_layers_for_rfsn(model: Any) -> None:
    """Restore original attention layer call methods."""
    inner = model
    if hasattr(model, "model"):
        inner = model.model
    if not hasattr(inner, "layers"):
        return
    for layer in inner.layers:
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if hasattr(attn, "_rfsn_original_call"):
            attn.__call__ = attn._rfsn_original_call
            delattr(attn, "_rfsn_original_call")


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


class RFSNGenerator:
    """High-level inference generator with RFSN runtime integration.

    Usage (MLX) ::

        from rfsn_v10.model_loader import load_mlx_model
        from rfsn_v10.runtime.generation import RFSNGenerator

        model, tokenizer = load_mlx_model("mlx-community/Llama-3-8B-Instruct-4bit")
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
        kv_manager: RFSNTurboQuantKVManager | None = None,
        enable_sparse_decode: bool = False,
        enable_quantized_kv: bool = True,
        audit_mode: bool = False,
    ):
        """
        Args:
            model: Loaded model (``mlx-lm`` or ``transformers``).
            tokenizer: Matching tokenizer.
            config: RFSN runtime configuration.  Loaded from env when ``None``.
            kv_manager: Optional KV-cache manager.  Created automatically when
                ``None`` and ``enable_quantized_kv`` is ``True``.
            enable_sparse_decode: Whether to enable RFSN sparse decode.
            enable_quantized_kv: Whether to use quantized KV-cache.
            audit_mode: Enable per-step quality auditing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or load_config()
        self.enable_sparse_decode = enable_sparse_decode
        self.enable_quantized_kv = enable_quantized_kv
        self.audit_mode = audit_mode

        self._kv_manager = kv_manager
        if kv_manager is None and enable_quantized_kv:
            self._kv_manager = RFSNTurboQuantKVManager(
                k_bits=8,
                v_bits=5,
                group_size=32,
            )

        self._runtime = None
        if self._kv_manager is not None:
            from .engine import RFSNRuntime
            self._runtime = RFSNRuntime(
                kv_manager=self._kv_manager,
                model_id=getattr(
                    tokenizer, "name_or_path", "unknown"
                ),
                enable_sparse_decode=enable_sparse_decode,
                audit_mode=audit_mode,
            )

        self._telemetry_log: list[dict] = []

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
        """Build a chat prompt using the tokenizer's chat template if present."""
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
        """Yield ``GenerationResponse`` from ``mlx_lm``, optionally via
        RFSNRuntime."""
        assert MLX_LM_AVAILABLE and _mlx_stream_generate is not None
        gen_iter = _mlx_stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temp=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        )
        if self._runtime is not None and self.enable_sparse_decode:
            with _RFSNSDPAPatcher(self._runtime):
                _wrap_layers_for_rfsn(self.model)
                try:
                    yield from gen_iter
                finally:
                    _unwrap_layers_for_rfsn(self.model)
        else:
            yield from gen_iter

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_telemetry(self) -> list[dict]:
        """Return accumulated telemetry from the runtime (if any)."""
        if self._runtime is not None:
            return [ev.__dict__ for ev in self._runtime.get_telemetry()]
        return []

    def clear_telemetry(self) -> None:
        """Clear telemetry log."""
        if self._runtime is not None:
            self._runtime.clear_telemetry()
