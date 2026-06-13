"""MLX model integration adapter for polar_fused.

Provides end-to-end generation with Polar quantization.

Usage modes:
1. Shadow mode: model runs standard attention, Polar path exercised in parallel
2. Replacement mode: eligible layer attentions are replaced with Polar attention

Replacement is done via instance-level method wrapping (not global monkey-patching).
"""
from __future__ import annotations

import time
from typing import Any

from rfsn_v11.polar_fused.attention_backend import PolarFusedAttentionBackend
from rfsn_v11.polar_fused.config import PolarFusedConfig
from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache
from rfsn_v11.polar_fused.quantize import PolarQuantizer

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]


class PolarAttentionWrapper:
    """Real attention module wrapper that replaces a layer's attention with Polar.

    Uses IncrementalPolarCache to avoid re-quantizing the full cache on every
    decode step.  Only newly appended tokens are quantized.

    This replaces ``layer.self_attn`` with the wrapper instance, which is
    functionally correct because Python resolves ``__call__`` on the class.
    All attribute accesses are transparently delegated to the original module.
    """

    __slots__ = (
        "original",
        "layer_id",
        "head_dim",
        "key_q",
        "value_q",
        "scale",
        "_inc_cache",
        "_prev_token_count",
    )

    def __init__(
        self,
        original_attn: Any,
        layer_id: int,
        key_quantizer: PolarQuantizer,
        value_quantizer: PolarQuantizer,
        head_dim: int,
    ) -> None:
        self.original = original_attn
        self.layer_id = layer_id
        self.head_dim = head_dim
        self.key_q = key_quantizer
        self.value_q = value_quantizer
        self.scale = getattr(original_attn, "scale", None)

        # Incremental quantized cache (persistent across steps)
        self._inc_cache = IncrementalPolarCache(
            key_quantizer=key_quantizer,
            value_quantizer=value_quantizer,
            batch_size=1,
            num_kv_heads=getattr(original_attn, "n_kv_heads", 1),
            head_dim=head_dim,
            chunk_size=64,
        )

        # Track how many tokens we've processed so we only append new ones
        self._prev_token_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self.original, name, value)

    def __call__(self, x: Any, mask: Any | None = None, cache: Any | None = None) -> Any:
        """Polar attention forward with incremental quantized cache."""
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        B, L, D = x.shape

        # Projections (same as original)
        queries = self.original.q_proj(x)
        keys = self.original.k_proj(x)
        values = self.original.v_proj(x)

        # Reshape to (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.original.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.original.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.original.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE + cache update (same as original)
        if cache is not None:
            queries = self.original.rope(queries, offset=cache.offset)
            keys = self.original.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.original.rope(queries)
            keys = self.original.rope(keys)

        # ---- Incremental Polar attention ----
        # keys, values are full cached tensors: (B, Hkv, T, head_dim)
        # Only append the NEW tokens to the incremental cache
        total_tokens = keys.shape[2]
        new_tokens = total_tokens - self._prev_token_count

        if new_tokens > 0:
            # Extract only the new tokens
            new_keys = keys[:, :, -new_tokens:, :]
            new_values = values[:, :, -new_tokens:, :]
            self._inc_cache.append(new_keys, new_values)
            self._prev_token_count = total_tokens

        # Attend from the incremental cache (kernel path — no dequantize)
        output = self._inc_cache.attend_kernel(queries, mask, scale=self.scale)
        # ---- end replacement ----

        # Output projection (same as original)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.original.o_proj(output)

class PolarModelRunner:
    """Runs an MLX model with Polar attention replacing standard attention.

    Instance-level replacement (not global monkey-patching).
    Only eligible middle layers (not boundary layers) use Polar.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: PolarFusedConfig | None = None,
    ) -> None:
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config or PolarFusedConfig.polar_safe()

        # Determine actual head_dim from first layer (must match all layers)
        actual_head_dim = self.cfg.head_dim
        for layer in getattr(model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
                n_heads = getattr(attn, "n_heads", 0)
                if n_heads > 0:
                    q_out = attn.q_proj.weight.shape[0]
                    if q_out % n_heads == 0:
                        actual_head_dim = q_out // n_heads
                        break

        # Setup quantizers (one pair for all layers)
        self._key_q = PolarQuantizer(
            bits=self.cfg.key_bits,
            head_dim=actual_head_dim,
            rotation_seed=self.cfg.key_rotation_seed,
        )
        self._value_q = PolarQuantizer(
            bits=self.cfg.value_bits,
            head_dim=actual_head_dim,
            rotation_seed=self.cfg.value_rotation_seed,
        )
        self._actual_head_dim = actual_head_dim

        # Determine which layers to replace
        backend = PolarFusedAttentionBackend(model, self.cfg)
        self._layer_modes = backend._layer_modes
        self._wrappers: list[PolarAttentionWrapper] = []

    def install(self) -> list[int]:
        """Replace eligible layer attentions with Polar wrappers.

        Returns list of layer IDs that were replaced.
        """
        replaced: list[int] = []
        for layer_id, mode in self._layer_modes.items():
            if mode != "polar":
                continue
            if layer_id >= len(self.model.layers):
                continue
            layer = self.model.layers[layer_id]
            if not hasattr(layer, "self_attn"):
                continue

            wrapper = PolarAttentionWrapper(
                layer.self_attn,
                layer_id,
                self._key_q,
                self._value_q,
                self._actual_head_dim,
            )
            # Replace the module instance with the wrapper.
            # Python resolves __call__ on the class, so this works correctly.
            layer.self_attn = wrapper  # type: ignore[assignment]
            self._wrappers.append((layer_id, wrapper))
            replaced.append(layer_id)

        return replaced

    def uninstall(self) -> None:
        """Restore all original attention methods."""
        for layer_id, wrapper in self._wrappers:
            if layer_id < len(self.model.layers):
                layer = self.model.layers[layer_id]
                if layer.self_attn is wrapper:
                    layer.self_attn = wrapper.original
        self._wrappers.clear()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        verbose: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Generate text with Polar attention.

        Installs Polar wrappers, runs generation, then uninstalls.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        from mlx_lm.models import cache as mlx_cache

        replaced = self.install()

        try:
            prompt_ids = self.tokenizer.encode(prompt)
            cache_list = [mlx_cache.KVCache() for _ in range(len(self.model.layers))]

            # Prefill
            t0 = time.monotonic()
            y = mx.array(prompt_ids)
            while y.size > 512:
                self.model(y[:512][None], cache=cache_list)
                y = y[512:]
            self.model(y[None], cache=cache_list)
            mx.eval([c.state for c in cache_list])
            prefill_time = time.monotonic() - t0

            # Decode
            gen_ids = list(prompt_ids)
            decode_times: list[float] = []

            for _ in range(max_tokens):
                t0 = time.monotonic()
                token_logits = self.model(mx.array([gen_ids[-1]])[None], cache=cache_list)
                mx.eval(token_logits)
                decode_times.append(time.monotonic() - t0)

                token_id = int(mx.argmax(token_logits[0, -1, :]).item())
                gen_ids.append(token_id)

            text = self.tokenizer.decode(gen_ids)

            # Gather cache metadata from wrappers
            cache_meta = {}
            for layer_id, wrapper in self._wrappers:
                cache_meta[layer_id] = wrapper._inc_cache.metadata()

            metrics = {
                "prefill_time_ms": prefill_time * 1000.0,
                "mean_decode_time_ms": (sum(decode_times) / len(decode_times)) * 1000.0,
                "tokens_generated": max_tokens,
                "polar_layers": replaced,
                "boundary_layers": [
                    lid for lid, mode in self._layer_modes.items() if mode == "fp16"
                ],
                "cache_metadata": cache_meta,
            }

            if verbose:
                print(f"Polar layers: {replaced}")
                print(f"Boundary layers (FP16): {metrics['boundary_layers']}")
                print(f"Generated {max_tokens} tokens in {metrics['mean_decode_time_ms']:.1f} ms/token")
                total_cache_mem = sum(m["memory_bytes"] for m in cache_meta.values())
                print(f"Total Polar cache memory: {total_cache_mem / 1024**2:.2f} MB")

            return text, metrics
        finally:
            self.uninstall()


# Backwards compat: keep shadow-mode adapter under a different name
class PolarModelAdapter(PolarModelRunner):
    """Deprecated alias — PolarModelRunner now performs actual replacement."""
    pass
