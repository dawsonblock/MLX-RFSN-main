"""Attention wrapper that replaces MLX-LM standard attention with packed blockwise attention.

Usage::

    from rfsn_v10.integrations.mlx_lm_model_support import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
    )

    caches = [
        RfsnDirectPackedKVCache(layer_id=i, key_codec=k_codec, value_codec=v_codec)
        for i in range(arch.num_layers)
    ]
    install_packed_attention(model, caches)
    # ... run generation with caches as prompt_cache ...
    # Wrappers stay installed; per-request caches select the backend.

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
            layer_id=layer_id,
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
        """Eval-able state for ``mx.eval`` during chunked prefill.

        Returns the current staging tensors so that ``mx.eval`` forces
        computation without materialising dense history.
        """
        stage_k, stage_v, stage_n = self.layer_cache.get_staging()
        if stage_n > 0 and stage_k is not None and stage_v is not None:
            return (stage_k, stage_v)
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

class _PackedAttentionWrapper:
    """Wrapper that intercepts attention calls and routes through packed reference.

    Python resolves ``__call__`` on the *class*, not the instance.  Therefore
    monkeypatching ``attn.__call__`` does **not** affect ``attn(...)``.  This
    wrapper class solves the problem by installing a wrapper *instance* whose
    class defines ``__call__``.

    All attribute accesses (``q_proj``, ``k_proj``, ``rope``, etc.) are
    transparently delegated to the original attention module.
    """

    __slots__ = ("_original", "_scale")

    def __init__(self, original: Any, scale: float) -> None:
        self._original = original
        self._scale = scale

    def __call__(
        self,
        x: Any,  # (B, L, D)
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> Any:
        # If no cache is provided, or the cache is not our packed cache,
        # fall back to the original attention.  This handles:
        #   - The very first prefill call when MLX-LM has not yet
        #     initialised the per-layer cache objects.
        #   - Any caller that passes a non-packed cache.
        if not isinstance(cache, RfsnDirectPackedKVCache):
            return self._original(x, mask=mask, cache=cache)

        B, L, D = x.shape
        attn = self._original

        # Original projections
        queries = attn.q_proj(x)
        keys = attn.k_proj(x)
        values = attn.v_proj(x)

        # Reshape to BHTD
        queries = queries.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE at original offset
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        # Append to our quantized cache (cache is RfsnDirectPackedKVCache)
        cache.update_and_fetch(keys, values)

        # Direct packed attention over the full quantized cache
        layer_cache = cache.layer_cache
        output, _ = attend(
            queries,
            layer_cache,
            scale=self._scale,
            mask=mask,
            query_start_pos=layer_cache.total_token_count() - L,
            causal=True,
        )

        # Reshape back and output projection
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self._original, name, value)

    def __repr__(self) -> str:
        return f"<_PackedAttentionWrapper wrapping {self._original!r}>"


def install_packed_attention(
    model: Any,
    caches: list[RfsnDirectPackedKVCache],
) -> None:
    """Install packed attention wrappers permanently on *model*.

    Each layer's ``self_attn`` is replaced with a ``_PackedAttentionWrapper``
    that delegates to the original module.  Wrappers are installed once at
    model-load time and stay in place; per-request ``prompt_cache`` objects
    select the packed path when provided.

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

        # Don't double-wrap
        if isinstance(attn, _PackedAttentionWrapper):
            continue

        # Attention scale — always present on MLX-LM attention modules.
        scale = getattr(attn, "scale", 1.0)

        wrapper = _PackedAttentionWrapper(attn, scale)
        layer.self_attn = wrapper


def uninstall_packed_attention(model: Any) -> None:
    """Remove packed attention wrappers and restore original modules."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, _PackedAttentionWrapper):
            layer.self_attn = attn._original


def is_model_wrapped(model: Any) -> bool:
    """Return ``True`` if any layer has a packed attention wrapper."""
    for layer in getattr(model, "layers", []):
        if isinstance(getattr(layer, "self_attn", None), _PackedAttentionWrapper):
            return True
    return False


# Backward compatibility aliases (deprecated)
wrap_model_attention = install_packed_attention
unwrap_model_attention = uninstall_packed_attention
