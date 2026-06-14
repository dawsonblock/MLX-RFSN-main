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
from rfsn_v10.compat import nn


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

        Returns every live tensor (sealed packed codes and scales, staging
        K/V, and dense residual K/V) so that ``mx.eval`` forces computation
        without materialising dense history.
        """
        tensors: list[Any] = []

        # Sealed key blocks
        for block in self.layer_cache.iter_key_blocks():
            if block.packed_codes is not None:
                tensors.append(block.packed_codes)
            if block.scales is not None:
                tensors.append(block.scales)

        # Sealed value blocks
        for block in self.layer_cache.iter_value_blocks():
            if block.packed_codes is not None:
                tensors.append(block.packed_codes)
            if block.scales is not None:
                tensors.append(block.scales)

        # Staging buffers
        stage_k, stage_v, stage_n = self.layer_cache.get_staging()
        if stage_n > 0:
            if stage_k is not None:
                tensors.append(stage_k)
            if stage_v is not None:
                tensors.append(stage_v)

        # Dense residual window
        dense_k, dense_v = self.layer_cache.get_dense_residual()
        if dense_k is not None:
            tensors.append(dense_k)
        if dense_v is not None:
            tensors.append(dense_v)

        return tuple(tensors)

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

class _PackedAttentionWrapper(nn.Module):
    """Wrapper that intercepts attention calls and routes through packed reference.

    Subclasses ``mlx.nn.Module`` so that the original attention subtree
    remains visible in MLX's parameter and module tree.  All dict entries
    and instance attributes from the original module are copied into the
    wrapper, preserving exact parameter paths.
    """

    def __init__(
        self, original: Any, scale: float, strict: bool = False
    ) -> None:
        super().__init__()
        # Copy all dict entries (parameters, submodules, arrays) so that
        # parameter paths like ``model.layers[0].self_attn.q_proj.weight``
        # remain valid after wrapping.
        for k, v in original.items():
            self[k] = v

        # Copy instance attributes that MLX stores outside the dict
        # (ints, floats, callables, etc.).  Skip MLX internal flags.
        for k, v in original.__dict__.items():
            if k not in ("_no_grad", "_training"):
                object.__setattr__(self, k, v)

        # Keep a private reference to the original for uninstall.
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_fallback_count", 0)
        object.__setattr__(self, "_executed_backend", "unknown")

    def __call__(
        self,
        x: Any,  # (B, L, D)
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> Any:
        # If no cache is provided, or the cache is not our packed cache,
        # handle according to strictness.
        if not isinstance(cache, RfsnDirectPackedKVCache):
            if self._strict:
                raise RuntimeError(
                    "Strict packed mode: received non-packed cache or no cache; "
                    "dense fallback is disabled."
                )
            object.__setattr__(self, "_fallback_count", self._fallback_count + 1)
            object.__setattr__(self, "_executed_backend", "dense")
            return self._original(x, mask=mask, cache=cache)

        B, L, D = x.shape

        # Original projections (copied into self, so direct access works)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to BHTD
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE at original offset
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        # Append to our quantized cache (cache is RfsnDirectPackedKVCache)
        cache.update_and_fetch(keys, values)

        object.__setattr__(self, "_executed_backend", "packed")

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
        return self.o_proj(output)

    def __getattr__(self, name: str) -> Any:
        # Delegate to the wrapped original module
        return getattr(self._original, name)

    def __repr__(self) -> str:
        return f"<_PackedAttentionWrapper wrapping {self._original!r}>"


def install_packed_attention(
    model: Any,
    caches: list[RfsnDirectPackedKVCache],
    strict: bool = False,
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
    strict
        If ``True``, the wrapper raises instead of silently falling back
        to dense attention when the cache is missing or of the wrong type.
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

        wrapper = _PackedAttentionWrapper(attn, scale, strict=strict)
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
