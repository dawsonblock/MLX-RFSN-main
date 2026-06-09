"""Model and tokenizer loading for RFSN v10 inference.

Provides a unified interface for loading HuggingFace transformers models
and MLX-native models.  The loader prefers ``mlx-lm`` on Apple Silicon and
falls back to ``transformers`` (PyTorch) when MLX is unavailable.
"""
from __future__ import annotations

from typing import Any


def _require_mlx_lm() -> Any:
    """Import and return the ``mlx_lm`` module or raise a clear error."""
    try:
        import mlx_lm
        return mlx_lm
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is required for model loading.  "
            "Install with: pip install mlx-lm"
        ) from exc


def _require_transformers() -> Any:
    """Import and return the ``transformers`` module or raise a clear error."""
    try:
        import transformers
        return transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenizer loading. "
            "Install with: pip install transformers"
        ) from exc


def load_tokenizer(model_id: str, **kwargs: Any) -> Any:
    """Load a tokenizer from a HuggingFace model ID or local path.

    Args:
        model_id: HuggingFace model ID (e.g. ``"meta-llama/Llama-3-8B-Instruct"``)
            or a local path to a model directory.
        **kwargs: Passed to ``AutoTokenizer.from_pretrained``.

    Returns:
        A transformers tokenizer instance.
    """
    transformers = _require_transformers()
    return transformers.AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_mlx_model(
    model_id: str,
    quant_config: str | None = None,
    lazy: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load an MLX model and tokenizer via ``mlx-lm``.

    Args:
        model_id: HuggingFace model ID or local path.
        quant_config: Optional quantization config string
            (e.g. ``"k8_v5_gs32"``).  Not used by ``mlx-lm`` directly;
            passed through for downstream RFSN runtime configuration.
        lazy: Whether to load model weights lazily (default ``True``).
        **kwargs: Passed to ``mlx_lm.load``.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    mlx_lm = _require_mlx_lm()
    model, tokenizer = mlx_lm.load(model_id, lazy=lazy, **kwargs)
    if quant_config:
        # Attach the RFSN quant config to the model for downstream use.
        model._rfsn_quant_config = quant_config
    return model, tokenizer


def load_model_auto(
    model_id: str,
    backend: str | None = None,
    quant_config: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a model and tokenizer using the best available backend.

    This is the recommended entry-point for inference code.

    Args:
        model_id: HuggingFace model ID or local path.
        backend: ``"mlx"`` or ``"torch"``.  When ``None`` the backend is
            chosen automatically: ``"mlx"`` on Apple Silicon when ``mlx-lm``
            is installed, otherwise ``"torch"``.
        quant_config: Optional RFSN quantization config string.
        **kwargs: Passed to the underlying loader.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    if backend is None:
        try:
            import mlx  # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "torch"

    if backend == "mlx":
        return load_mlx_model(model_id, quant_config=quant_config, **kwargs)

    if backend == "torch":
        transformers = _require_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            **kwargs,
        )
        if quant_config:
            model._rfsn_quant_config = quant_config
        return model, tokenizer

    raise ValueError(f"Unknown backend {backend!r}.  Use 'mlx' or 'torch'.")


def get_model_info(model: Any) -> dict[str, Any]:
    """Return a dict of model metadata for telemetry / logging.

    Args:
        model: A loaded model instance (MLX or PyTorch).

    Returns:
        Dict with keys like ``architecture``, ``num_layers``, ``hidden_size``,
        etc.  Values may be ``None`` if the backend does not expose them.
    """
    info: dict[str, Any] = {
        "architecture": None,
        "num_layers": None,
        "hidden_size": None,
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "vocab_size": None,
        "quant_config": getattr(model, "_rfsn_quant_config", None),
    }

    # mlx-lm models are dict-like (weights)
    if hasattr(model, "config"):
        cfg = model.config
        info["architecture"] = getattr(cfg, "architectures", [None])[0]
        info["num_layers"] = getattr(cfg, "num_hidden_layers", None)
        info["hidden_size"] = getattr(cfg, "hidden_size", None)
        info["num_attention_heads"] = getattr(cfg, "num_attention_heads", None)
        info["num_key_value_heads"] = getattr(
            cfg, "num_key_value_heads", info["num_attention_heads"]
        )
        info["vocab_size"] = getattr(cfg, "vocab_size", None)
    elif hasattr(model, "_model") and hasattr(model._model, "config"):
        # mlx-lm wrapper
        cfg = model._model.config
        info["architecture"] = getattr(cfg, "model_type", None)
        info["num_layers"] = getattr(cfg, "num_hidden_layers", None)
        info["hidden_size"] = getattr(cfg, "hidden_size", None)
        info["num_attention_heads"] = getattr(cfg, "num_attention_heads", None)
        info["num_key_value_heads"] = getattr(
            cfg, "num_key_value_heads", info["num_attention_heads"]
        )
        info["vocab_size"] = getattr(cfg, "vocab_size", None)

    return info
