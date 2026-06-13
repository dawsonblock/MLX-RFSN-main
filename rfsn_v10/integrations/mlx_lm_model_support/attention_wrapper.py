"""Attention wrapper that replaces MLX-LM standard attention with packed blockwise attention.

Usage::

    from rfsn_v10.integrations.mlx_lm_model_support import (
        RfsnDirectPackedKVCache,
        wrap_model_attention,
    )

    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(arch.num_layers)
    ]
    wrap_model_attention(model, caches)
    # ... run generation with caches as prompt_cache ...
    unwrap_model_attention(model)

The wrapper:
1. Calls original Q/K/V projections.
2. Applies RoPE at the original offset.
3. Appends K/V to the per-layer QuantizedLayerCache.
4. Invokes packed reference attention (no full dense reconstruction).
5. Calls the original output projection.
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
from rfsn_v10.cache.mlx_packed_attention_reference import attend
from rfsn_v10.compat import mx


class RfsnDirectPackedKVCache:
    """Cache adapter for the direct packed attention path.

    Wraps a ``QuantizedLayerCache`` and implements the minimal MLX-LM cache
    interface so that the generation loop can pass it to attention layers.
    Unlike the dense-reconstruction reference, this cache does **not**
    return full dense K/V history from ``update_and_fetch``.
    """

    def __init__(
        self,
        layer_id: int,
        key_codec: CartesianCodec,
        value_codec: CartesianCodec,
        staging_capacity: int = 64,
        dense_residual_window: int = 0,
    ) -> None:
        self.layer_id = layer_id
        self.layer_cache = QuantizedLayerCache(
            key_codec=key_codec,
            value_codec=value_codec,
            staging_capacity=staging_capacity,
            dense_residual_window=dense_residual_window,
        )
        self.offset: int = 0

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        """Append new K/V tokens and return the new tokens only (not full history).

        For the direct packed path, the attention wrapper ignores the
        returned dense history and instead calls ``attend()`` on the
        ``QuantizedLayerCache`` directly.
        """
        self.layer_cache.append(keys, values)
        self.offset = self.layer_cache.total_token_count()
        return keys, values

    @property
    def state(self) -> tuple[Any, ...]:
        """Lightweight state for ``mx.eval``."""
        return ()

    @state.setter
    def state(self, v: Any) -> None:
        if v:
            raise NotImplementedError(
                "RfsnDirectPackedKVCache does not support state injection"
            )

    def is_trimmable(self) -> bool:
        return False

    def trim(self, n: int) -> int:
        if n > 0:
            raise NotImplementedError(
                "trim() is not supported in the direct packed path. "
                "Use reset() and re-prefill."
            )
        return 0

    def reset(self) -> None:
        self.layer_cache.reset()
        self.offset = 0

    def destroy(self) -> None:
        self.layer_cache.destroy()


# ------------------------------------------------------------------
# Attention wrapper
# ------------------------------------------------------------------

_original_callers: dict[int, Any] = {}


def _make_packed_attention_call(
    original_call: Any,
    layer_cache: QuantizedLayerCache,
    key_codec: CartesianCodec,
    value_codec: CartesianCodec,
    scale: float,
) -> Any:
    """Build a replacement ``__call__`` for an MLX-LM Attention module."""

    def _packed_attn_call(
        self: Any,
        x: Any,  # (B, L, D)
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> Any:
        B, L, D = x.shape

        # Original projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to BHTD
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE at original offset
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            # Append to our quantized cache (cache is RfsnDirectPackedKVCache)
            cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Direct packed attention over the full quantized cache
        output, _ = attend(
            queries,
            layer_cache,
            scale=scale,
            mask=mask,
            query_start_pos=layer_cache.total_token_count() - L,
            causal=True,
        )

        # Reshape back and output projection
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    return _packed_attn_call


def wrap_model_attention(
    model: Any,
    caches: list[RfsnDirectPackedKVCache],
) -> None:
    """Replace every attention module in *model* with the packed attention path.

    Parameters
    ----------
    model
        An MLX-LM model (e.g. Qwen2Model).
    caches
        One ``RfsnDirectPackedKVCache`` per layer, in layer order.
    """
    layers = getattr(model, "layers", [])
    if len(layers) != len(caches):
        raise ValueError(
            f"Model has {len(layers)} layers but {len(caches)} caches provided"
        )

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError(f"Layer {i} has no self_attn attribute")

        cache_wrapper = caches[i]
        key_codec = cache_wrapper.layer_cache.key_codec
        value_codec = cache_wrapper.layer_cache.value_codec
        scale = key_codec.group_size ** -0.5  # fallback; real scale from args

        # Try to get actual scale from the attention module
        scale = getattr(attn, "scale", scale)

        original = attn.__call__
        _original_callers[id(attn)] = original

        wrapped = _make_packed_attention_call(
            original,
            cache_wrapper.layer_cache,
            key_codec,
            value_codec,
            scale,
        )
        # Bind the wrapper to the module instance
        import types
        attn.__call__ = types.MethodType(wrapped, attn)


def unwrap_model_attention(model: Any) -> None:
    """Restore the original attention modules."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        key = id(attn)
        if key in _original_callers:
            original = _original_callers.pop(key)
            attn.__call__ = original
