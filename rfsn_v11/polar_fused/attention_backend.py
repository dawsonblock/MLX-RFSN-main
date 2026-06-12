"""Complete PolarFusedAttentionBackend with explicit routing.

This is the main entry point for polar_fused attention. It:
1. Inspects the model to find eligible layers
2. Applies boundary-layer protection
3. Routes each layer to either Polar or standard MLX attention
4. Collects telemetry

The Polar path uses NaivePolarAttention (dequantize-full-cache reference)
which is the correctness oracle.  Metal kernels can be swapped in later
via the kernel registry without changing this backend.
"""
from __future__ import annotations

import time
from typing import Any

from .adapters.boundary_layers import BoundaryLayerPolicy
from .adapters.model_inspection import ModelInspector
from .attention import NaivePolarAttention
from .config import PolarFusedConfig
from .contracts import AttentionOutputResult, QuantizedVectors
from .fallback import StandardMLXAttention
from .lazy_convert import LazyPolarCache
from .quantize import PolarQuantizer
from .rotations import get_default_rotation_registry
from .telemetry import PolarTelemetry, PolarTelemetryCollector

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


class PolarFusedAttentionBackend:
    """Explicit backend for Polar fused attention.

    No global monkey-patching.  The caller selects this backend explicitly
    through an attention strategy or cache protocol.

    Usage for model integration::

        backend = PolarFusedAttentionBackend(model, config)
        # During prefill: store K/V in lazy caches
        backend.prefill_store(layer_id, keys, values)
        # During decode: compute attention from cached K/V
        output = backend.attend(layer_id, queries, mask=mask, scale=scale)
    """

    def __init__(
        self,
        model: Any,
        config: PolarFusedConfig | None = None,
    ) -> None:
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        self.cfg = config or PolarFusedConfig.polar_safe()
        self.model = model

        # Inspect model
        self._inspector = ModelInspector(self.cfg)
        self._layer_classes = self._inspector.inspect(model)
        self._eligible_layers = self._inspector.get_polar_eligible_layers(model)

        # Apply boundary protection
        self._boundary = BoundaryLayerPolicy(self.cfg)
        self._layer_modes = self._boundary.apply(
            self._eligible_layers,
            total_layers=len(self._layer_classes),
        )

        # Setup quantizers
        self._key_q = PolarQuantizer(
            bits=self.cfg.key_bits,
            head_dim=self.cfg.head_dim,
            rotation_seed=self.cfg.key_rotation_seed,
        )
        self._value_q = PolarQuantizer(
            bits=self.cfg.value_bits,
            head_dim=self.cfg.head_dim,
            rotation_seed=self.cfg.value_rotation_seed,
        )

        # NaivePolarAttention instance (correctness oracle)
        self._naive_attn = NaivePolarAttention(
            key_quantizer=self._key_q,
            value_quantizer=self._value_q,
        )

        # Fallback backend
        self._fallback = StandardMLXAttention()

        # Telemetry
        self._telemetry = PolarTelemetryCollector()

        # Per-layer lazy caches (initialized lazily)
        self._lazy_caches: dict[int, LazyPolarCache] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supports(self, layer_id: int) -> bool:
        """Check if this backend can handle the given layer."""
        return self._layer_modes.get(layer_id) == "polar"

    def prefill_store(
        self,
        layer_id: int,
        keys: Any,    # (B, Hkv, Lkv, D)
        values: Any,  # (B, Hkv, Lkv, D)
    ) -> None:
        """Store prefill K/V into the lazy cache for this layer.

        This should be called once per layer after prefill completes.
        The lazy cache will either keep FP16 (below threshold) or
        bulk-convert to Polar packed (at/above threshold).
        """
        if not HAS_MLX:
            return
        cache = self._get_or_create_lazy_cache(layer_id)
        # Append all prefill tokens at once
        cache.append(keys, values)

    def decode_store(
        self,
        layer_id: int,
        key: Any,    # (B, Hkv, 1, D)
        value: Any,  # (B, Hkv, 1, D)
    ) -> None:
        """Store a single decode-step K/V token into the lazy cache."""
        if not HAS_MLX:
            return
        cache = self._get_or_create_lazy_cache(layer_id)
        cache.append(key, value)

    def attend(
        self,
        layer_id: int,
        queries: Any,       # (B, Hq, Lq, D)
        mask: Any | None = None,
        scale: float | None = None,
    ) -> AttentionOutputResult:
        """Compute attention for a layer using cached K/V.

        This is the decode-path entry point.  The K/V must have been
        previously stored via ``prefill_store`` or ``decode_store``.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        t0 = time.monotonic()

        if self.supports(layer_id):
            try:
                result = self._polar_attend(layer_id, queries, mask, scale)
                backend = "polar_fused"
            except Exception:
                if self.cfg.allow_fallback:
                    # Retrieve FP16 from cache if available
                    cache = self._lazy_caches.get(layer_id)
                    if cache is not None:
                        cached = cache.get_cache_for_attention()
                        if cached["mode"] == "fp16":
                            result = self._fallback.attend(
                                queries, cached["keys"], cached["values"], mask, scale
                            )
                            backend = "fallback_fp16"
                        else:
                            result = self._fallback.attend(queries, None, None, mask, scale)
                            backend = "fallback_no_cache"
                    else:
                        result = self._fallback.attend(queries, None, None, mask, scale)
                        backend = "fallback_no_cache"
                else:
                    raise
        else:
            # Boundary/fallback layer — caller should provide standard cache
            # If no cache available, we cannot compute attention
            result = self._fallback.attend(queries, None, None, mask, scale)
            backend = "mlx_standard"

        latency_ms = (time.monotonic() - t0) * 1000.0

        self._telemetry.record(PolarTelemetry(
            backend=backend,
            latency_ms=latency_ms,
            tokens=queries.shape[2],
            heads=queries.shape[1],
            head_dim=queries.shape[3],
        ))

        return result

    def get_telemetry(self) -> dict[str, Any]:
        """Return telemetry summary."""
        return self._telemetry.summary()

    def get_layer_summary(self) -> dict[str, Any]:
        """Return model inspection + boundary layer summary."""
        return {
            "inspection": self._inspector.summary(self.model),
            "boundary": self._boundary.summary(self._layer_modes),
            "layer_modes": self._layer_modes,
        }

    def get_cache_metadata(self) -> dict[int, dict[str, Any]]:
        """Return metadata for all per-layer caches."""
        return {
            layer_id: cache.metadata()
            for layer_id, cache in self._lazy_caches.items()
        }

    # ------------------------------------------------------------------
    # Polar attention path
    # ------------------------------------------------------------------

    def _polar_attend(
        self,
        layer_id: int,
        queries: Any,
        mask: Any | None,
        scale: float | None,
    ) -> AttentionOutputResult:
        """Polar attention using NaivePolarAttention with cached quantized K/V."""
        cache = self._lazy_caches.get(layer_id)
        if cache is None:
            raise RuntimeError(f"No cache for layer {layer_id}; call prefill_store first")

        cached = cache.get_cache_for_attention()

        if cached["mode"] == "fp16":
            # Below lazy threshold — use standard attention
            return self._fallback.attend(
                queries, cached["keys"], cached["values"], mask, scale
            )

        # Polar packed mode — retrieve quantized vectors
        polar_cache = cached["polar_cache"]
        s = polar_cache.get_valid_slice()

        # Unpack indices and build QuantizedVectors for keys and values
        from .packing import unpack_indices

        key_indices = unpack_indices(
            s.key_indices, self.cfg.key_bits, self.cfg.head_dim
        )
        value_indices = unpack_indices(
            s.value_indices, self.cfg.value_bits, self.cfg.head_dim
        )

        key_qv = QuantizedVectors(
            indices=key_indices,
            norms=s.key_norms,
            original_dim=self.cfg.head_dim,
            bits=self.cfg.key_bits,
            rotation_id=self._key_q._rotation_id,
            codebook_id=self._key_q._codebook_id,
        )
        value_qv = QuantizedVectors(
            indices=value_indices,
            norms=s.value_norms,
            original_dim=self.cfg.head_dim,
            bits=self.cfg.value_bits,
            rotation_id=self._value_q._rotation_id,
            codebook_id=self._value_q._codebook_id,
        )

        # Use NaivePolarAttention (correctness oracle)
        return self._naive_attn.attend(queries, key_qv, value_qv, mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_lazy_cache(self, layer_id: int) -> LazyPolarCache:
        if layer_id not in self._lazy_caches:
            # Determine actual head_dim from model inspection
            head_dim = self.cfg.head_dim
            n_kv_heads = 1
            if layer_id < len(self._layer_classes):
                n_kv_heads = self._layer_classes[layer_id].n_kv_heads or 1

            self._lazy_caches[layer_id] = LazyPolarCache(
                config=self.cfg,
                batch_size=1,
                num_kv_heads=n_kv_heads,
                head_dim=head_dim,
                key_quantizer=self._key_q,
                value_quantizer=self._value_q,
            )
        return self._lazy_caches[layer_id]
