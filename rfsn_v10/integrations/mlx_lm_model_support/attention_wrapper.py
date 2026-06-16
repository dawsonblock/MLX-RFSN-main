"""Attention wrapper replacing MLX-LM attention with packed blockwise.

Usage::

    from rfsn_v10.integrations.mlx_lm_model_support import (
        RfsnDirectPackedKVCache,
        install_packed_attention,
    )

    caches = [
        RfsnDirectPackedKVCache(
            layer_id=i, key_codec=k_codec, value_codec=v_codec
        )
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

from contextlib import contextmanager
from typing import Any

import mlx.core as mx

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache
from rfsn_v10.cache.mlx_packed_attention_reference import attend
from rfsn_v10.compat import nn

# Dense-reconstruction Metal kernel (kept as fallback)
from rfsn_v10.kernels.metal.packed_attention_metal import (
    attend_metal,
    metal_available,
)

# Canonical true-packed kernel for PackedBlockV4 (K8/V8 only).
# This is gated by an explicit opt-in environment variable and a
# self-test; it is NOT automatically enabled merely because MLX
# imports.  See ``rfsn_v10.kernels.metal.packed_v4_attention``.
from rfsn_v10.kernels.metal.packed_v4_attention import (
    HAS_TRUE_PACKED_KERNEL,
    PackedV4AttentionKernel,
)

HAS_METAL_KERNEL = metal_available()


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
        strict: bool = False,
        session: Any = None,
    ) -> None:
        self.layer_id = layer_id
        self.strict = strict
        self.session = session

        # P0 #4: Make session own the actual layer cache
        # If session is provided, use its layer cache; otherwise create one
        if session is not None:
            self.layer_cache = session.get_layer_cache(layer_id)
        else:
            self.layer_cache = QuantizedLayerCache(
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
                layer_id=layer_id,
                session=session,
            )
        self.offset: int = 0

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        """Append new K/V tokens and return them (not full history).

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

def _dense_attention_with_stats(
    queries: Any,
    keys: Any,
    values: Any,
    scale: float,
    query_start_pos: int,
    kv_start_pos: int,
    mask: Any | None = None,
    causal: bool = True,
) -> tuple[Any, Any, Any]:
    """Dense attention with online-softmax statistics for region merging.

    Returns
    -------
    output : (B, Hq, Lq, D)
    running_max : (B, Hq, Lq)
    running_sum : (B, Hq, Lq)
    """
    B, Hq, Lq, D = queries.shape
    _, Hkv, Tk, _ = keys.shape

    # GQA: repeat KV heads to match query heads
    if Hq != Hkv:
        repeats = Hq // Hkv
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale  # (B, Hq, Lq, Tk)

    if mask is not None and not isinstance(mask, str):
        scores = scores + mask
    elif causal or (isinstance(mask, str) and mask.lower() == "causal"):
        q_positions = mx.arange(query_start_pos, query_start_pos + Lq)[:, None]
        kv_positions = mx.arange(kv_start_pos, kv_start_pos + Tk)[None, :]
        causal_mask = (q_positions >= kv_positions).astype(mx.float32)
        causal_mask = mx.broadcast_to(
            causal_mask[None, None, :, :], (B, Hq, Lq, Tk)
        )
        scores = mx.where(
            causal_mask, scores, mx.array(-mx.inf, dtype=scores.dtype)
        )

    running_max = mx.max(scores, axis=-1, keepdims=True)  # (B, Hq, Lq, 1)
    exp_scores = mx.exp(scores - running_max)
    exp_scores = mx.where(mx.isfinite(exp_scores), exp_scores, 0.0)
    running_sum = mx.sum(exp_scores, axis=-1, keepdims=True)  # (B, Hq, Lq, 1)
    acc = exp_scores @ values  # (B, Hq, Lq, D)  unnormalized

    output = mx.where(
        running_sum > 0,
        acc / running_sum,
        mx.zeros_like(acc),
    )
    return output, running_max.squeeze(-1), running_sum.squeeze(-1)


def _merge_attention_regions(
    regions: list[tuple[Any, Any, Any]],
) -> Any:
    """Merge multiple attention regions using online-softmax statistics.

    Each region is a tuple ``(output, running_max, running_sum)``.
    Returns the merged attention output in the same domain.
    """
    if not regions:
        raise ValueError("No regions to merge")
    if len(regions) == 1:
        return regions[0][0]

    global_max = regions[0][1]
    for _, max_i, _ in regions[1:]:
        global_max = mx.maximum(global_max, max_i)

    acc_total = None
    sum_total = 0
    for output_i, max_i, sum_i in regions:
        scale = mx.exp(max_i - global_max)
        weighted_acc = output_i * sum_i[..., None] * scale[..., None]
        if acc_total is None:
            acc_total = weighted_acc
        else:
            acc_total = acc_total + weighted_acc
        sum_total = sum_total + sum_i * scale

    safe_sum = mx.where(sum_total == 0, mx.ones_like(sum_total), sum_total)
    return mx.where(
        sum_total[..., None] > 0,
        acc_total / safe_sum[..., None],
        mx.zeros_like(acc_total),
    )


class _PackedAttentionWrapper(nn.Module):
    """Wrapper intercepting attention calls via packed reference.

    Subclasses ``mlx.nn.Module`` so that the original attention subtree
    remains visible in MLX's parameter and module tree.  All dict entries
    and instance attributes from the original module are copied into the
    wrapper, preserving exact parameter paths.
    """

    def __init__(
        self,
        original: Any,
        scale: float,
        strict: bool = False,
        layer_id: int | None = None,
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
        object.__setattr__(self, "_attempted_backends", [])
        object.__setattr__(self, "_layer_id", layer_id)
        # P1: Execution contract storage for auditability
        object.__setattr__(self, "_last_execution_contract", None)
        object.__setattr__(self, "_cached_kernel", None)

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
                    "Strict packed mode: received non-packed "
                    "cache or no cache; dense fallback is disabled."
                )
            object.__setattr__(
                self, "_fallback_count", self._fallback_count + 1
            )
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
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        # RoPE at original offset
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        # Append to our quantized cache (cache is RfsnDirectPackedKVCache)
        cache.update_and_fetch(keys, values)

        object.__setattr__(self, "_executed_backend", "packed")

        # Direct packed attention over the full quantized cache
        layer_cache = cache.layer_cache
        query_start_pos = layer_cache.total_token_count() - L

        # ------------------------------------------------------------------
        # P0.7: Reject unsupported masks before dispatch.
        # The true-packed path only supports causal masking or no mask.
        # mlx_lm passes causal mask arrays during prefill (T>1); we accept
        # MLX arrays because our kernel implements causal logic internally.
        # Non-causal masks (additive, sliding-window, padding) are not yet
        # supported and will be ignored — this is a known limitation tracked
        # in the P3 repair plan.
        # ------------------------------------------------------------------
        _mask_is_mlx_array = isinstance(mask, mx.array)
        _mask_unsupported = False
        if mask is not None:
            if isinstance(mask, str):
                _mask_unsupported = mask.lower() != "causal"
            elif not _mask_is_mlx_array:
                _mask_unsupported = True

        if _mask_unsupported and self._strict:
            raise RuntimeError(
                f"Strict packed mode: unsupported mask {type(mask).__name__}; "
                "only causal=True or mask=None is supported."
            )

        # ------------------------------------------------------------------
        # P0: Strict mode must fail immediately if the packed kernel is
        # unavailable, rather than falling through to a cryptic None output.
        # ------------------------------------------------------------------
        if self._strict and not HAS_TRUE_PACKED_KERNEL:
            raise RuntimeError(
                "Strict packed mode: HAS_TRUE_PACKED_KERNEL is False. "
                "Set RFSN_ENABLE_TRUE_PACKED=1 and ensure the Metal "
                "self-test passes on Apple Silicon."
            )

        # ------------------------------------------------------------------
        # Dispatch
        # ------------------------------------------------------------------
        output = None
        contract = None
        _attempted: list[str] = []
        _packed_kernel_dispatched = False
        _staging_dispatched = False
        _dense_residual_dispatched = False

        if HAS_TRUE_PACKED_KERNEL and not _mask_unsupported:
            _attempted.append("true_packed_v4")
            try:
                _has_codec = hasattr(layer_cache, "key_codec")
                if self._cached_kernel is None:
                    import os as _os
                    _kv_tile = int(_os.environ.get("RFSN_KV_TILE_SIZE", "0"))
                    object.__setattr__(
                        self,
                        "_cached_kernel",
                        PackedV4AttentionKernel(
                            bits=layer_cache.key_codec.bits if _has_codec else 8,
                            group_size=(
                                layer_cache.key_codec.group_size
                                if _has_codec
                                else 64
                            ),
                            sign_seed=(
                                layer_cache.key_codec.sign_seed
                                if _has_codec
                                else 42
                            ),
                            kv_tile_size=_kv_tile,
                        ),
                    )
                kernel = self._cached_kernel
                assert kernel is not None

                # Gather regions
                key_blocks = list(layer_cache.iter_key_blocks())
                value_blocks = list(layer_cache.iter_value_blocks())
                stage_k, stage_v, stage_n = layer_cache.get_staging()
                dense_k, dense_v = layer_cache.get_dense_residual()

                regions: list[tuple[Any, Any, Any]] = []

                # ---- Packed region ----
                if key_blocks:
                    packed_out, packed_max, packed_sum, packed_contract = (
                        kernel(
                            queries=queries,
                            key_blocks=key_blocks,
                            value_blocks=value_blocks,
                            scale=self._scale,
                            causal=True,
                            query_start_pos=query_start_pos,
                            strict=self._strict,
                        )
                    )
                    regions.append((packed_out, packed_max, packed_sum))
                    contract = packed_contract
                    _packed_kernel_dispatched = True

                # ---- Staging region ----
                if stage_n > 0 and stage_k is not None and stage_v is not None:
                    stage_offset = layer_cache.encoded_token_count
                    stage_out, stage_max, stage_sum = _dense_attention_with_stats(
                        queries,
                        stage_k,
                        stage_v,
                        scale=self._scale,
                        query_start_pos=query_start_pos,
                        kv_start_pos=stage_offset,
                        mask=None,
                        causal=True,
                    )
                    regions.append((stage_out, stage_max, stage_sum))
                    _staging_dispatched = True

                # ---- Dense residual region ----
                if dense_k is not None and dense_v is not None:
                    dense_tokens = int(dense_k.shape[2])
                    dense_offset = layer_cache.total_token_count() - dense_tokens
                    dense_out, dense_max, dense_sum = _dense_attention_with_stats(
                        queries,
                        dense_k,
                        dense_v,
                        scale=self._scale,
                        query_start_pos=query_start_pos,
                        kv_start_pos=dense_offset,
                        mask=None,
                        causal=True,
                    )
                    regions.append((dense_out, dense_max, dense_sum))
                    _dense_residual_dispatched = True

                if regions:
                    output = _merge_attention_regions(regions)
                else:
                    # Empty cache — all zeros
                    output = mx.zeros(
                        (B, self.n_heads, L, D), dtype=queries.dtype
                    )

                # P0: truthful backend attribution based on what actually ran
                if _packed_kernel_dispatched:
                    if _staging_dispatched and _dense_residual_dispatched:
                        backend_label = "packed_metal_plus_staging_residual"
                    elif _staging_dispatched:
                        backend_label = "packed_metal_plus_staging"
                    elif _dense_residual_dispatched:
                        backend_label = "packed_metal_plus_residual"
                    else:
                        backend_label = "packed_metal_only"
                elif _staging_dispatched:
                    backend_label = "dense_staging_only"
                elif _dense_residual_dispatched:
                    backend_label = "dense_residual_only"
                else:
                    backend_label = "empty_cache"

                object.__setattr__(self, "_executed_backend", backend_label)
                if contract is not None:
                    object.__setattr__(
                        self, "_last_execution_contract", contract
                    )

                # P0: record packed attention only when the kernel actually ran
                if _packed_kernel_dispatched:
                    _sess = getattr(layer_cache, "session", None)
                    if _sess is not None:
                        _sess.runtime_counters.record_packed_attention()
                        # P1: Record per-block reads and bytes from contract
                        if contract is not None:
                            _sess.runtime_counters.record_block_read(
                                contract.packed_blocks_read
                            )
                            _sess.runtime_counters.record_packed_read(
                                contract.packed_bytes_read
                            )
            except Exception as exc:
                _sess = getattr(layer_cache, "session", None)
                if _sess is not None:
                    _sess.runtime_counters.record_attempted_backend(
                        "true_packed_v4"
                    )
                # P1.1: In strict mode, forbid any fallback backend.
                if self._strict:
                    raise RuntimeError(
                        "Strict packed mode: True packed V4 kernel failed: "
                        f"{exc}"
                    ) from exc
                output = None

        # ------------------------------------------------------------------
        # Fallback paths (only when strict=False and true-packed failed)
        # ------------------------------------------------------------------
        if output is None and not self._strict:
            _attempted.append("metal_dense")
            if HAS_METAL_KERNEL:
                try:
                    output, _ = attend_metal(
                        queries,
                        layer_cache,
                        scale=self._scale,
                        mask=mask,
                        query_start_pos=query_start_pos,
                        causal=True,
                        strict=self._strict,
                    )
                    object.__setattr__(
                        self,
                        "_executed_backend",
                        "metal_dense_reconstruction_violates_invariant",
                    )
                except Exception as exc:
                    _sess = getattr(layer_cache, "session", None)
                    if _sess is not None:
                        _sess.runtime_counters.record_attempted_backend(
                            "metal_dense"
                        )
                    if self._strict:
                        raise RuntimeError(
                            "Strict packed mode: Metal kernel failed and "
                            f"fallback is disabled: {exc}"
                        ) from exc
                    output = None

        if output is None and not self._strict:
            _attempted.append("packed_reference")
            output, _ = attend(
                queries,
                layer_cache,
                scale=self._scale,
                mask=mask,
                query_start_pos=query_start_pos,
                causal=True,
            )
            object.__setattr__(
                self, "_executed_backend", "packed_reference"
            )

        # Record fallback if we ended up on reference after trying higher-priority paths.
        self._attempted_backends.extend(_attempted)
        if (
            self._executed_backend == "packed_reference"
            and len(_attempted) > 1
        ):
            object.__setattr__(
                self, "_fallback_count", self._fallback_count + 1
            )
            _sess = getattr(layer_cache, "session", None)
            if _sess is not None:
                _sess.runtime_counters.record_fallback()

        # P0: record_packed_attention() is now called inside the packed-kernel
        # branch only when the kernel actually dispatches.  Do NOT infer packed
        # execution from wrapper exit.

        # Reshape back and output projection
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def __getattr__(self, name: str) -> Any:
        # Delegate to the wrapped original module
        return getattr(self._original, name)

    def __repr__(self) -> str:
        return f"<_PackedAttentionWrapper wrapping {self._original!r}>"

    def get_backend_stats(self) -> dict[str, Any]:
        """Return backend execution statistics."""
        stats = {
            "layer_id": self._layer_id,
            "executed_backend": self._executed_backend,
            "attempted_backends": self._attempted_backends.copy(),
            "fallback_count": self._fallback_count,
            "strict": self._strict,
        }

        # P0: Include execution contract with truthful measured counters
        if self._last_execution_contract is not None:
            contract = self._last_execution_contract
            stats["execution_contract"] = {
                "backend": contract.backend,
                "kernel_hash": contract.kernel_hash,
                "num_key_blocks": contract.num_key_blocks,
                "num_value_blocks": contract.num_value_blocks,
                "total_kv_tokens": contract.total_kv_tokens,
                "dense_kv_materialized_bytes": contract.dense_kv_materialized_bytes,
                "packed_history_copy_bytes": contract.packed_history_copy_bytes,
                "query_transform_bytes": contract.query_transform_bytes,
                "scratch_bytes": contract.scratch_bytes,
                "output_bytes": contract.output_bytes,
                "decoded_dense_tokens": contract.decoded_dense_tokens,
                "packed_blocks_read": contract.packed_blocks_read,
                "packed_bytes_read": contract.packed_bytes_read,
                "materialized_bytes": contract.materialized_bytes,
                "decoded_tokens": contract.decoded_tokens,
                "execution_ms": contract.execution_ms,
            }
            # Validate and report invariant status
            passed, violations = contract.validate_invariant()
            stats["invariant_passed"] = passed
            if violations:
                stats["invariant_violations"] = violations

        return stats


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

        wrapper = _PackedAttentionWrapper(
            attn, scale, strict=strict, layer_id=i
        )
        layer.self_attn = wrapper


def uninstall_packed_attention(model: Any) -> None:
    """Remove packed attention wrappers and restore original modules."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, _PackedAttentionWrapper):
            layer.self_attn = attn._original


@contextmanager
def packed_attention_context(
    model: Any,
    caches: list[RfsnDirectPackedKVCache],
    strict: bool = False,
):
    """Context manager that installs packed attention wrappers.

    Ensures wrappers are always uninstalled even if generation raises,
    preventing cross-run contamination.

    Parameters
    ----------
    model
        An MLX-LM model (e.g. Qwen2Model).
    caches
        One ``RfsnDirectPackedKVCache`` per layer, in layer order.
    strict
        If ``True``, the wrapper raises instead of silently falling back
        to dense attention when the cache is missing or of the wrong type.

    Example
    -------
    ::

        with packed_attention_context(model, caches, strict=True):
            logits = model(prompt_ids, cache=caches)
        # Wrappers are automatically uninstalled here
    """
    install_packed_attention(model, caches, strict=strict)
    try:
        yield
    finally:
        uninstall_packed_attention(model)


def collect_backend_stats(model: Any) -> list[dict[str, Any]]:
    """Collect backend execution stats from all wrapped attention layers."""
    stats = []
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, _PackedAttentionWrapper):
            stats.append(attn.get_backend_stats())
    return stats


def is_model_wrapped(model: Any) -> bool:
    """Return ``True`` if any layer has a packed attention wrapper."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, _PackedAttentionWrapper):
            return True
    return False


# Backward compatibility aliases (deprecated)
wrap_model_attention = install_packed_attention
unwrap_model_attention = uninstall_packed_attention
