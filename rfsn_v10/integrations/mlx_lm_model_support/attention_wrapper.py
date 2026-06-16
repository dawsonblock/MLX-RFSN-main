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
from rfsn_v10.cache.paged_arena import validate_direct_packed_format
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
        max_context_tokens: int = 16384,
    ) -> None:
        self.layer_id = layer_id
        self.strict = strict
        self.session = session

        # Direct packed Metal currently requires K8/V8 GS64.
        validate_direct_packed_format(
            key_codec, value_codec, label="RfsnDirectPackedKVCache"
        )

        # P0 #4: Make session own the actual layer cache
        # If session is provided, use its layer cache; otherwise create a
        # standalone layer cache with persistent paging forced on.
        if session is not None:
            self.layer_cache = session.get_layer_cache(layer_id)
        else:
            import math

            max_pages = math.ceil(max_context_tokens / staging_capacity)
            self.layer_cache = QuantizedLayerCache(
                key_codec=key_codec,
                value_codec=value_codec,
                staging_capacity=staging_capacity,
                dense_residual_window=dense_residual_window,
                layer_id=layer_id,
                session=session,
                use_paged_arena=True,
                max_pages=max_pages,
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

        Returns every live tensor (paged arena arrays or sealed blocks,
        staging K/V, and dense residual K/V) so that ``mx.eval`` forces
        computation without materialising dense history.
        """
        tensors: list[Any] = []

        # Paged arena arrays (production path)
        paged = self.layer_cache.get_paged_kv_view()
        if paged is not None:
            tensors.extend([
                paged.k_codes,
                paged.k_scales,
                paged.v_codes,
                paged.v_scales,
                paged.page_table,
                paged.page_starts,
                paged.page_counts,
            ])
        else:
            # Fallback path: sealed key/value blocks
            for block in self.layer_cache.iter_key_blocks():
                if block.packed_codes is not None:
                    tensors.append(block.packed_codes)
                if block.scales is not None:
                    tensors.append(block.scales)
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
        # When global_max is -inf (all regions empty / fully masked),
        # exp(max_i - global_max) produces NaN.  Guard so that a
        # zero-sum region contributes nothing rather than poisoning
        # the total with NaN.
        raw_scale = mx.exp(max_i - global_max)
        scale = mx.where(mx.isfinite(raw_scale), raw_scale, mx.ones_like(raw_scale))
        weighted_acc = output_i * sum_i[..., None] * scale[..., None]
        if acc_total is None:
            acc_total = weighted_acc
        else:
            acc_total = acc_total + weighted_acc
        sum_total = sum_total + sum_i * scale

    safe_sum = mx.where(
        mx.logical_or(sum_total == 0, mx.isnan(sum_total)),
        mx.ones_like(sum_total),
        sum_total,
    )
    return mx.where(
        sum_total[..., None] > 0,
        acc_total / safe_sum[..., None],
        mx.zeros_like(acc_total),
    )


def _is_expected_causal_mask(
    mask: mx.array,
    queries: mx.array,
    query_start_pos: int,
) -> bool:
    """Return True if ``mask`` is the expected additive causal mask.

    MLX-LM creates an additive mask with shape ``(Lq, total_kv_len)`` where
    positions that a query token is not allowed to attend to are set to a
    large negative value (``-1e9``, which may underflow to ``-inf`` in
    float16) and allowed positions are ``0`` (or ``-0``).  The packed kernel
    implements this logic internally, so we verify the supplied mask and
    then ignore it.
    """
    B, Hq, Lq, D = queries.shape
    total_kv_len = query_start_pos + Lq

    if Lq == 0:
        return True

    if mask.shape[-2:] != (Lq, total_kv_len):
        return False

    # Work with the trailing (Lq, total_kv_len) slice.  Leading dims are
    # broadcast, so we just need one representative slice.
    mask_2d = mask.reshape(-1, Lq, total_kv_len)[0] if mask.ndim > 2 else mask

    # Build the oracle: allowed iff kv_pos <= q_pos.
    q_positions = mx.arange(query_start_pos, query_start_pos + Lq)[:, None]
    kv_positions = mx.arange(total_kv_len)[None, :]
    allowed = q_positions >= kv_positions

    # Allowed positions must be numerically zero (either 0 or -0).
    allowed_values = mx.where(allowed, mask_2d, mx.zeros_like(mask_2d))
    if not bool(mx.all(allowed_values == 0).item()):
        return False

    # Masked positions must be strongly negative (<= -1e8 or -inf).
    masked_values = mx.where(~allowed, mask_2d, mx.array(-1e9, dtype=mask_2d.dtype))
    if not bool(mx.all(masked_values <= -1e8).item()):
        return False

    return True


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
        # P4.6: Accumulate all contracts for prefill/decode aggregation
        object.__setattr__(self, "_execution_contracts", [])
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
        # P0.8: Verify causal masks; reject all others.
        # The true-packed path implements causal logic internally, so an
        # externally-provided mask is only safe if it is exactly the causal
        # additive mask that the model would have applied.  Arbitrary MLX
        # masks (padding, sliding-window, additive, etc.) must not be ignored.
        # ------------------------------------------------------------------
        if mask is not None:
            if not _is_expected_causal_mask(mask, queries, query_start_pos):
                raise RuntimeError(
                    "Packed attention currently supports only the expected "
                    "causal additive mask.  Custom masks are not supported."
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

        if HAS_TRUE_PACKED_KERNEL:
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
                paged_kv = layer_cache.get_paged_kv_view()
                stage_k, stage_v, stage_n = layer_cache.get_staging()
                dense_k, dense_v = layer_cache.get_dense_residual()

                regions: list[tuple[Any, Any, Any]] = []

                # ---- Packed region ----
                # P0.7: In strict mode the direct-packed path requires a
                # persistent paged arena.  Rebuilding a temporary arena from
                # blocks on every attention call would reintroduce the O(history)
                # copy that paged storage is meant to eliminate.
                if paged_kv is None and self._strict:
                    raise RuntimeError(
                        "Direct packed attention requires persistent paged storage"
                    )

                if paged_kv is not None:
                    # Heuristic: Lq > 1 is prefill, Lq == 1 is decode
                    _is_prefill = queries.shape[2] > 1
                    packed_out, packed_max, packed_sum, packed_contract = (
                        kernel(
                            queries=queries,
                            paged_kv=paged_kv,
                            scale=self._scale,
                            causal=True,
                            query_start_pos=query_start_pos,
                            strict=self._strict,
                            layer_id=layer_cache.layer_id,
                            is_prefill=_is_prefill,
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
                    # Empty cache — all zeros (use head_dim, not hidden_size)
                    head_dim = D // self.n_heads
                    output = mx.zeros(
                        (B, self.n_heads, L, head_dim), dtype=queries.dtype
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
                    self._execution_contracts.append(contract)

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
                "prefill_ms": contract.prefill_ms,
                "decode_ms": contract.decode_ms,
            }
            # Validate and report invariant status
            passed, violations = contract.validate_invariant()
            stats["invariant_passed"] = passed
            if violations:
                stats["invariant_violations"] = violations

        # P4.6: Aggregate prefill/decode across all accumulated contracts
        total_prefill_ms = 0.0
        total_decode_ms = 0.0
        total_execution_ms = 0.0
        for c in self._execution_contracts:
            total_prefill_ms += c.prefill_ms
            total_decode_ms += c.decode_ms
            total_execution_ms += c.execution_ms
        stats["aggregated_prefill_ms"] = total_prefill_ms
        stats["aggregated_decode_ms"] = total_decode_ms
        stats["aggregated_execution_ms"] = total_execution_ms
        stats["num_calls"] = len(self._execution_contracts)

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
