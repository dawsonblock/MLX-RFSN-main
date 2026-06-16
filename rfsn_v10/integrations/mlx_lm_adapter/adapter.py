"""Explicit MLX-LM adapter — no monkeypatching.

The adapter creates ``RfsnQuantizedKVCache`` objects that implement the
MLX-LM cache interface.  These caches are passed directly to
``mlx_lm.utils.generate`` or ``mlx_lm.utils.generate_step``.

No attention function is replaced.  The model's standard attention runs
with reconstructed dense K/V (fallback path).  When Metal kernels arrive,
attention will read packed blocks directly without reconstructing dense.

Proof counters (tracked per session):
  new_tokens_received       — K/V tokens passed by the model
  new_tokens_encoded        — tokens successfully appended to quantized cache
  packed_blocks_read        — sealed blocks decoded for attention
  sealed_blocks_created     — immutable blocks created
  fallback_attention_calls  — times dense reconstruction was needed
  dense_shadow_bytes        — total bytes in temporary dense reconstructions
  requantized_tokens        — tokens re-quantized (should always be 0)
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.session import GenerationCacheSession

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]


class RfsnDenseReconstructionReferenceCache:
    """Reference cache adapter that reconstructs dense K/V on every call.

    This is the **reference-only** fallback path.  It decompresses the
    full historical K/V cache into dense FP16 on every attention step so
    that the model's unmodified attention can run.  It must not be promoted
    as a speed or memory improvement.
    """

    def __init__(
        self,
        layer_cache: Any,  # QuantizedLayerCache
        session: GenerationCacheSession,
        strict: bool = False,
    ) -> None:
        self.layer_cache = layer_cache
        self.session = session
        self.strict = strict
        self.offset = 0
        self._shape_meta: tuple[int, int, int] | None = None  # (B, Hkv, D)

    # ------------------------------------------------------------------
    # MLX-LM cache interface
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        """Append new K/V and return full dense cache.

        Parameters
        ----------
        keys, values
            Shape ``(B, n_kv_heads, new_tokens, head_dim)``.

        Returns
        -------
        full_keys, full_values
            Shape ``(B, n_kv_heads, total_tokens, head_dim)``.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        B, Hkv, new_T, D = keys.shape
        if self._shape_meta is None:
            self._shape_meta = (B, Hkv, D)

        # Append to quantized cache and track both legacy counter names.
        # new_tokens_received: how many K/V tokens the model passed in.
        # new_tokens_encoded: how many were encoded into the quantized cache.
        # record_token_appended is NOT called here; layer_cache.append() handles it.
        self.layer_cache.append(keys, values)
        self.session.increment("new_tokens_received", new_T)
        self.session.increment("new_tokens_encoded", new_T)

        # Track sealed blocks
        stats = self.layer_cache.stats()
        if stats.staged_tokens == 0 and stats.sealed_blocks > 0:
            self.session.increment("sealed_blocks_created", stats.sealed_blocks)

        # Fallback: reconstruct dense cache
        if self.strict:
            raise RuntimeError("Strict mode enabled: fallback to dense reconstruction is prohibited")
        dense_k, dense_v = self._reconstruct_dense()
        self.offset = dense_k.shape[2]

        # Proof: dense shadow bytes (temporary reconstruction)
        dense_bytes = int(dense_k.size) * 2 + int(dense_v.size) * 2  # FP16
        self.session.increment("dense_shadow_bytes", dense_bytes)
        self.session.increment("fallback_attention_calls", 1)

        return dense_k, dense_v

    @property
    def state(self) -> tuple[Any, ...]:
        """Lightweight state for ``mx.eval`` in MLX-LM generation."""
        # Return empty tuple — our quantized data is already on-device.
        # We do not materialise dense arrays just for mx.eval.
        return ()

    @state.setter
    def state(self, v: Any) -> None:
        # No-op — our cache does not support state injection.
        if self.strict and v:
            raise NotImplementedError(
                "RfsnDenseReconstructionReferenceCache does not support "
                "state injection in strict mode"
            )

    def is_trimmable(self) -> bool:
        # Partial trim of sealed blocks drops whole blocks rather than
        # re-encoding a partial block.  Return False until this is fixed.
        return False

    def trim(self, n: int) -> int:
        """Trim the last n tokens from the cache.

        Raises:
            NotImplementedError: Partial trim is not yet supported.
                The caller must use reset() and re-prefill.
        """
        if n > 0:
            raise NotImplementedError(
                "RfsnQuantizedKVCache.trim() is not supported. "
                "Use reset() and re-prefill."
            )
        return 0

    # ------------------------------------------------------------------
    # Dense reconstruction (fallback path — temporary, not retained)
    # ------------------------------------------------------------------

    def _reconstruct_dense(self) -> tuple[Any, Any]:
        """Reconstruct dense K/V from all quantized blocks."""
        if self._shape_meta is None:
            raise RuntimeError("Cache has no shape metadata; call update_and_fetch first")
        B, Hkv, D = self._shape_meta

        key_parts: list[Any] = []
        value_parts: list[Any] = []

        # Sealed blocks
        for kb in self.layer_cache.iter_key_blocks():
            k_flat = self.layer_cache.key_codec.decode(kb)
            block_T = kb.token_count
            k_reshaped = k_flat.reshape(B, Hkv, block_T, D)
            key_parts.append(k_reshaped)
            self.session.increment("packed_blocks_read", 1)

        for vb in self.layer_cache.iter_value_blocks():
            v_flat = self.layer_cache.value_codec.decode(vb)
            block_T = vb.token_count
            v_reshaped = v_flat.reshape(B, Hkv, block_T, D)
            value_parts.append(v_reshaped)

        # Staging — already full-shaped (B, Hkv, staged_T, D)
        stage_k, stage_v, _stage_n = self.layer_cache.get_staging()
        if stage_k is not None:
            key_parts.append(stage_k)
            value_parts.append(stage_v)

        # Dense residual — already full-shaped (B, Hkv, dense_T, D)
        dense_k, dense_v = self.layer_cache.get_dense_residual()
        if dense_k is not None:
            key_parts.append(dense_k)
            value_parts.append(dense_v)

        if not key_parts:
            # Empty cache — return empty arrays
            empty_k = mx.zeros((B, Hkv, 0, D), dtype=mx.float16)
            empty_v = mx.zeros((B, Hkv, 0, D), dtype=mx.float16)
            return empty_k, empty_v

        full_k = mx.concatenate(key_parts, axis=2)
        full_v = mx.concatenate(value_parts, axis=2)
        return full_k, full_v

    def blockwise_attention(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        scale: float,
        mask: Any | None = None,
    ) -> Any:
        """Compute attention directly on quantized blocks without dense reconstruction.

        Returns shape ``(B, Hq, Lq, D)``.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")
        return self.layer_cache.blockwise_attention(queries, scale, mask)


class RfsnMLXReferenceAdapter:
    """Reference adapter that runs MLX-LM models with dense-reconstruction fallback.

    This is a **reference-only** path: every decode step reconstructs the
    full dense K/V history.  It is useful for correctness validation but
    must never be promoted as a production speed or memory win.

    Usage::

        adapter = RfsnMLXReferenceAdapter(model, tokenizer, num_layers=24)
        text = adapter.generate("Hello", max_tokens=32)
        print(adapter.counters())
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
        strict: bool = False,
        use_direct_packed: bool = False,
        max_context_tokens: int = 16384,
    ) -> None:
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        self.model = model
        self.tokenizer = tokenizer
        self.strict = strict
        self.use_direct_packed = use_direct_packed
        self.max_context_tokens = max_context_tokens

        if num_layers is None:
            num_layers = len(getattr(model, "layers", []))
        self.num_layers = num_layers

        # Codecs
        self.key_codec = CartesianCodec(bits=key_bits, group_size=group_size)
        self.value_codec = CartesianCodec(bits=value_bits, group_size=group_size)

        # Direct packed Metal currently requires K8/V8 GS64.
        if self.use_direct_packed:
            from rfsn_v10.cache.paged_arena import validate_direct_packed_format

            validate_direct_packed_format(
                self.key_codec,
                self.value_codec,
                label="RfsnMLXReferenceAdapter",
            )

        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window

        # Session (created per generation, not persisted)
        self._session: GenerationCacheSession | None = None
        self._cache_list: list[Any] = []
        self._last_counters: dict[str, int] = {}
        self._last_memory_report: dict[str, Any] = {}

        # Strict mode: require packed cache and reject fallback
        if self.strict and not self.use_direct_packed:
            raise ValueError("Strict mode requires use_direct_packed=True")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        verbose: bool = False,
        **generate_kwargs: Any,
    ) -> str:
        """Generate text using the standard MLX-LM path with our caches.

        Creates a fresh ``GenerationCacheSession`` for each call.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        try:
            from mlx_lm import generate
        except ImportError:
            from mlx_lm.utils import generate

        if self.use_direct_packed:
            from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
                RfsnDirectPackedKVCache,
                install_packed_attention,
                is_model_wrapped,
            )
            # Create session for direct packed path
            self._session = self._new_session()
            caches = [
                RfsnDirectPackedKVCache(
                    layer_id=i,
                    key_codec=self.key_codec,
                    value_codec=self.value_codec,
                    staging_capacity=self.staging_capacity,
                    dense_residual_window=self.dense_residual_window,
                    strict=self.strict,
                    session=self._session,
                )
                for i in range(self.num_layers)
            ]
            if not is_model_wrapped(self.model):
                install_packed_attention(self.model, caches, strict=self.strict)
            text = generate(
                self.model,
                self.tokenizer,
                prompt,
                verbose=verbose,
                prompt_cache=caches,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
            self._last_counters = {
                "direct_packed_tokens": sum(
                    c.layer_cache.total_token_count() for c in caches
                )
                // self.num_layers,
            }
            # Capture runtime counters from session
            if self._session:
                self._last_counters["runtime_counters"] = self._session.runtime_counters
            # Capture execution backend from wrappers
            try:
                from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
                    collect_backend_stats,
                )
                backend_stats = collect_backend_stats(self.model)
                if backend_stats:
                    backends = [s.get("executed_backend", "unknown") for s in backend_stats]
                    # Use the most specific backend (not "packed" which is set before Metal)
                    metal_backends = [b for b in backends if b.startswith("metal_")]
                    if metal_backends:
                        self._last_counters["execution_backend"] = metal_backends[0]
                    elif "packed_reference" in backends:
                        self._last_counters["execution_backend"] = "packed_reference"
                    elif "dense" in backends:
                        self._last_counters["execution_backend"] = "dense"
                    else:
                        self._last_counters["execution_backend"] = backends[0]
            except Exception:
                self._last_counters["execution_backend"] = "unknown"
            # Capture memory report before destroying session
            if self._session:
                self._last_memory_report = self._session.memory_report().to_dict()
                self._last_counters = self._session.counters()
            # Destroy session
            if self._session:
                self._session.destroy()
            return text

        # Dense reconstruction path
        session = self._new_session()
        try:
            # Build cache list for this generation
            self._cache_list = [
                RfsnQuantizedKVCache(
                    layer_cache=session.get_layer_cache(i),
                    session=session,
                    strict=self.strict,
                )
                for i in range(self.num_layers)
            ]

            # Pass our caches to MLX-LM via prompt_cache
            text = generate(
                self.model,
                self.tokenizer,
                prompt,
                verbose=verbose,
                prompt_cache=self._cache_list,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
            return text
        finally:
            # Capture report and counters before destroy
            self._last_memory_report = session.memory_report().to_dict()
            self._last_counters = session.counters()
            session.destroy()

    def generate_step(
        self,
        prompt: str,
        max_tokens: int = 32,
        **generate_kwargs: Any,
    ):
        """Yield tokens one at a time using ``generate_step``."""
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        try:
            from mlx_lm import generate_step
        except ImportError:
            from mlx_lm.utils import generate_step

        if self.use_direct_packed:
            from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
                RfsnDirectPackedKVCache,
                install_packed_attention,
                is_model_wrapped,
            )
            # Create session for direct packed path
            self._session = self._new_session()
            caches = [
                RfsnDirectPackedKVCache(
                    layer_id=i,
                    key_codec=self.key_codec,
                    value_codec=self.value_codec,
                    staging_capacity=self.staging_capacity,
                    dense_residual_window=self.dense_residual_window,
                    strict=self.strict,
                    session=self._session,
                )
                for i in range(self.num_layers)
            ]
            if not is_model_wrapped(self.model):
                install_packed_attention(self.model, caches, strict=self.strict)
            prompt_ids = (
                prompt if isinstance(prompt, mx.array)
                else mx.array(self.tokenizer.encode(prompt))
            )
            try:
                yield from generate_step(
                    prompt_ids,
                    self.model,
                    max_tokens=max_tokens,
                    prompt_cache=caches,
                    **generate_kwargs,
                )
            finally:
                self._last_counters = {
                    "direct_packed_tokens": sum(
                        c.layer_cache.total_token_count() for c in caches
                    )
                    // self.num_layers,
                }
                # Capture runtime counters from session
                if self._session:
                    self._last_counters["runtime_counters"] = self._session.runtime_counters
                # Capture execution backend from wrappers
                try:
                    from rfsn_v10.integrations.mlx_lm_model_support.attention_wrapper import (
                        collect_backend_stats,
                    )
                    backend_stats = collect_backend_stats(self.model)
                    if backend_stats:
                        backends = [s.get("executed_backend", "unknown") for s in backend_stats]
                        metal_backends = [b for b in backends if b.startswith("metal_")]
                        if metal_backends:
                            self._last_counters["execution_backend"] = metal_backends[0]
                        elif "packed_reference" in backends:
                            self._last_counters["execution_backend"] = "packed_reference"
                        elif "dense" in backends:
                            self._last_counters["execution_backend"] = "dense"
                        else:
                            self._last_counters["execution_backend"] = backends[0]
                except Exception:
                    self._last_counters["execution_backend"] = "unknown"
                # Capture memory report before destroying session
                if self._session:
                    self._last_memory_report = self._session.memory_report().to_dict()
                    self._last_counters = self._session.counters()
                # Destroy session
                if self._session:
                    self._session.destroy()
            return

        session = self._new_session()
        try:
            self._cache_list = [
                RfsnQuantizedKVCache(
                    layer_cache=session.get_layer_cache(i),
                    session=session,
                    strict=self.strict,
                )
                for i in range(self.num_layers)
            ]

            prompt_ids = (
                prompt if isinstance(prompt, mx.array)
                else mx.array(self.tokenizer.encode(prompt))
            )

            yield from generate_step(
                prompt_ids,
                self.model,
                max_tokens=max_tokens,
                prompt_cache=self._cache_list,
                **generate_kwargs,
            )
        finally:
            self._last_memory_report = session.memory_report().to_dict()
            self._last_counters = session.counters()
            session.destroy()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _new_session(self) -> GenerationCacheSession:
        """Create a new isolated session for this generation."""
        import math

        max_pages = math.ceil(
            self.max_context_tokens / self.staging_capacity
        )

        self._session = GenerationCacheSession(
            model_id=getattr(self.model, "model_type", "unknown"),
            num_layers=self.num_layers,
            key_codec=self.key_codec,
            value_codec=self.value_codec,
            staging_capacity=self.staging_capacity,
            dense_residual_window=self.dense_residual_window,
            use_paged_arena=self.use_direct_packed,
            max_pages=max_pages,
        )
        # Fix #1: Track strict mode in session runtime counters
        self._session.runtime_counters.requested_strict_mode = self.strict
        self._session.runtime_counters.effective_strict_mode = self.strict
        return self._session

    # ------------------------------------------------------------------
    # Proof counters
    # ------------------------------------------------------------------

    def counters(self) -> dict[str, int]:
        """Return proof counters from the last generation session."""
        if hasattr(self, "_last_counters"):
            return self._last_counters
        if self._session is not None:
            return self._session.counters()
        return {}

    def total_memory_bytes(self) -> int:
        """Total memory across all layer caches in the current session."""
        if self._session is None:
            return 0
        return self._session.total_memory_bytes()

    def dense_shadow_bytes(self) -> int:
        """Total dense shadow bytes (temporary reconstructions)."""
        return self.counters().get("dense_shadow_bytes", 0)

    def fallback_calls(self) -> int:
        """Number of fallback attention calls."""
        return self.counters().get("fallback_attention_calls", 0)

    def memory_report(self) -> dict[str, Any]:
        """Return detailed memory report from the last generation."""
        if hasattr(self, "_last_memory_report"):
            return self._last_memory_report
        from rfsn_v10.cache.memory import MemoryReport
        return MemoryReport().to_dict()


# Backward compatibility aliases (deprecated)
RfsnMLXModelAdapter = RfsnMLXReferenceAdapter
RfsnQuantizedKVCache = RfsnDenseReconstructionReferenceCache
