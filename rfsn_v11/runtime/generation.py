"""RFSN v11 production generation loop.

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

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from ..errors import BackendError, ModelNotLoadedError, RFSNError  # noqa: F401

# ---------------------------------------------------------------------------
# Config import — v11 config module always exists; guard defensively anyway.
# ---------------------------------------------------------------------------
try:
    from ..config import RFSNConfig, load_config
except ImportError:
    RFSNConfig = None  # type: ignore[assignment,misc]

    def load_config(path=None):  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# Optional heavy dependencies — silently degrade when absent.
# ---------------------------------------------------------------------------

# RFSNTurboQuantKVManager lives in kv_manager, which does not exist in v11 yet.
try:
    from ..kv_manager import RFSNTurboQuantKVManager
except ImportError:
    RFSNTurboQuantKVManager = None  # type: ignore[assignment,misc]
    logging.getLogger(__name__).warning(
        "rfsn_v11: RFSNTurboQuantKVManager not importable — "
        "quantized KV-cache disabled.  Install the kv_manager module to enable."
    )

# RFSNRuntime lives in .engine, which does not exist in v11 yet.
try:
    from .engine import RFSNRuntime
except ImportError:
    RFSNRuntime = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Thread-local storage for RFSNRuntime SDPA patching context.
# ---------------------------------------------------------------------------
_rfsn_thread_local = threading.local()

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional platform / framework dependencies.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SDPA intercept helpers
# ---------------------------------------------------------------------------

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


class _LayerIdWrapper:
    """Wrapper that injects layer_id into thread-local before each forward."""

    __slots__ = ("_original", "_layer_id")

    def __init__(self, original: Any, layer_id: str) -> None:
        self._original = original
        self._layer_id = layer_id

    def __call__(self, x: Any, mask: Any | None = None, cache: Any | None = None) -> Any:
        old = getattr(_rfsn_thread_local, "layer_id", None)
        _rfsn_thread_local.layer_id = self._layer_id
        try:
            return self._original(x, mask, cache)
        finally:
            _rfsn_thread_local.layer_id = old

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self._original, name, value)


_original_attns: dict[int, Any] = {}


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
        key = id(layer)
        if key in _original_attns:
            continue
        _original_attns[key] = attn
        layer.self_attn = _LayerIdWrapper(attn, f"layer_{idx}")


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
        key = id(layer)
        if key in _original_attns:
            layer.self_attn = _original_attns.pop(key)


# ---------------------------------------------------------------------------
# Public data-classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class RFSNGenerator:
    """High-level inference generator with RFSN runtime integration.

    Usage (MLX) ::

        from rfsn_v11.model_loader import load_mlx_model
        from rfsn_v11.runtime.generation import RFSNGenerator

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
        config: "RFSNConfig | None" = None,
        kv_manager: Any | None = None,
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
                ``None`` and ``enable_quantized_kv`` is ``True`` and
                ``RFSNTurboQuantKVManager`` is importable.
            enable_sparse_decode: Whether to enable RFSN sparse decode.
            enable_quantized_kv: Whether to use quantized KV-cache.
            audit_mode: Enable per-step quality auditing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config if config is not None else load_config()
        self.enable_sparse_decode = enable_sparse_decode
        self.enable_quantized_kv = enable_quantized_kv
        self.audit_mode = audit_mode

        # ------------------------------------------------------------------
        # KV-cache manager — only instantiate when the class is importable.
        # ------------------------------------------------------------------
        self._kv_manager = kv_manager
        if kv_manager is None and enable_quantized_kv:
            if RFSNTurboQuantKVManager is not None:
                self._kv_manager = RFSNTurboQuantKVManager(
                    k_bits=8,
                    v_bits=5,
                    group_size=64,
                )
            else:
                _logger.warning(
                    "rfsn_v11: enable_quantized_kv=True but "
                    "RFSNTurboQuantKVManager is not importable — "
                    "_kv_manager left as None."
                )

        # ------------------------------------------------------------------
        # Runtime — only instantiate when the class is importable.
        # ------------------------------------------------------------------
        self._runtime = None
        if self._kv_manager is not None:
            if RFSNRuntime is not None:
                self._runtime = RFSNRuntime(
                    kv_manager=self._kv_manager,
                    model_id=getattr(
                        tokenizer, "name_or_path", "unknown"
                    ),
                    enable_sparse_decode=enable_sparse_decode,
                    audit_mode=audit_mode,
                )
            else:
                _logger.warning(
                    "rfsn_v11: RFSNRuntime is not importable — "
                    "runtime integration disabled.  _runtime left as None."
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

        Raises:
            :class:`~rfsn_v11.errors.ModelNotLoadedError`: If neither mlx-lm nor
                transformers is available.
            :class:`~rfsn_v11.errors.BackendError`: If the active backend raises
                an unrecoverable error during generation.
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

        Raises:
            :class:`~rfsn_v11.errors.BackendError`: If mlx-lm is not available
                or the backend encounters an unrecoverable error.
        """
        cfg = self._make_gen_config(**gen_kwargs)
        if MLX_LM_AVAILABLE and hasattr(self.model, "__call__"):
            yield from self._stream_mlx(prompt, cfg)
        else:
            raise BackendError(
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

        Raises:
            :class:`~rfsn_v11.errors.BackendError`: Propagated from
                :meth:`generate` if the backend is unavailable.
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
        """Run synchronous generation and return the full result.

        Raises:
            :class:`~rfsn_v11.errors.ModelNotLoadedError`: If no backend is
                available at all.
            :class:`~rfsn_v11.errors.BackendError`: If the active backend raises
                an unrecoverable error during generation.
        """
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
            raise ModelNotLoadedError(
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
        """Generate via ``transformers`` pipeline.

        Raises:
            :class:`~rfsn_v11.errors.BackendError`: If the transformers model
                raises an unrecoverable error.
        """
        assert TRANSFORMERS_AVAILABLE
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "to"):
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.temperature > 0,
            )
        except Exception as exc:
            raise BackendError(
                f"transformers model.generate() failed: {exc}"
            ) from exc

        generated = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _stream_mlx(self, prompt: str, cfg: GenerationConfig) -> Iterator[str]:
        """Stream generation via ``mlx_lm``, yielding individual tokens."""
        for response in self._mlx_gen_iter(prompt, cfg):
            yield response.text

    def _mlx_gen_iter(self, prompt: str, cfg: GenerationConfig):
        """Yield ``GenerationResponse`` from ``mlx_lm``, optionally via
        RFSNRuntime.

        Raises:
            :class:`~rfsn_v11.errors.BackendError`: If ``mlx_lm`` stream
                generation raises an unrecoverable error.
        """
        assert MLX_LM_AVAILABLE and _mlx_stream_generate is not None
        try:
            gen_iter = _mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=cfg.max_new_tokens,
                temp=cfg.temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )
        except Exception as exc:
            raise BackendError(
                f"mlx_lm stream_generate() initialisation failed: {exc}"
            ) from exc

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
