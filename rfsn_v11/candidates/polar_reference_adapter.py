"""Candidate: PolarQuant reference adapter — real KV-cache compression.

Wires external/mlx-turboquant's TurboQuantKVCache (which uses the PolarQuant
codebook-based compressor internally) into the generation loop via
mlx_lm.utils.generate_step(prompt_cache=...).

PolarQuant algorithm (from external/mlx-turboquant/mlx_turboquant/polar_quant.py)
----------------------------------------------------------------------------------
1. Store the L2 norm of each KV vector.
2. Normalize vector to unit sphere.
3. Apply a fixed random orthogonal rotation R (so coordinates are
   Beta-distributed — data-oblivious Lloyd-Max quantization applies).
4. Quantize each rotated coordinate using a precomputed codebook
   (stored in external/mlx-turboquant/mlx_turboquant/data/codebooks.npz).
5. On fetch: look up centroids, inverse-rotate, rescale by norm.

This produces *genuinely lower* quantization error than affine (minmax)
quantization on un-rotated vectors because the rotation maps the
cosine-sphere distribution to Beta-distributed marginals that match
the Lloyd-Max codebook exactly.

SDPA patch note
---------------
mlx_turboquant.integration._turboquant_sdpa passes a stale `sinks` kwarg
that current mlx_lm.models.base.scaled_dot_product_attention no longer
accepts.  We install a minimal fixed patch here instead of calling
integration.patch_sdpa(), so TurboQuantKVCache (which only needs the
standard SDPA path when use_qjl=False) runs cleanly.

Source:
  external/mlx-turboquant/mlx_turboquant/cache.py
  external/mlx-turboquant/mlx_turboquant/polar_quant.py
  external/mlx-turboquant/mlx_turboquant/integration.py

Status: reference candidate — not promotion eligible without full logit gate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE

_EXT_POLAR = str(
    Path(__file__).parent.parent.parent / "external" / "mlx-turboquant"
)


def _ensure_ext_on_path() -> None:
    if _EXT_POLAR not in sys.path:
        sys.path.insert(0, _EXT_POLAR)


# -----------------------------------------------------------------------
# Vectorized PolarQuant.quantize — replaces the slow Python loop
# -----------------------------------------------------------------------
# polar_quant.PolarQuant.quantize() uses a sequential Python loop over
# codebook boundaries:
#
#   for i in range(n_levels - 1):
#       indices = indices + (rotated > inner_bounds[i]).astype(uint8)
#
# This is O(n_levels) kernel launches per token step.  We replace it with
# a single vectorized comparison:
#
#   (rotated[..., None] > inner_bounds).sum(axis=-1).astype(uint8)
#
# Same result, single MLX kernel.  Applied as a module-level patch after
# the first import of mlx_turboquant.polar_quant.

_polar_quant_patched = False


def _patch_polar_quant_vectorize() -> None:
    """Replace PolarQuant.quantize with a vectorized (single-kernel) version."""
    global _polar_quant_patched
    if _polar_quant_patched:
        return
    _ensure_ext_on_path()
    import mlx.core as mx
    from mlx_turboquant.polar_quant import PolarQuant

    def _vectorized_quantize(self, vectors: mx.array):  # type: ignore[override]
        norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        unit = vectors / mx.maximum(norms, 1e-8)
        rotated = unit @ self.rotation_t
        # Single broadcast comparison: (..., dim, 1) > (n_levels-1,) -> sum
        inner_bounds = self.boundaries[1:-1]  # (n_levels-1,)
        indices = (rotated[..., None] > inner_bounds).sum(axis=-1).astype(mx.uint8)
        return indices, norms

    PolarQuant.quantize = _vectorized_quantize  # type: ignore[method-assign]
    _polar_quant_patched = True


# -----------------------------------------------------------------------
# Minimal SDPA patch for TurboQuantKVCache (no-QJL path)
# -----------------------------------------------------------------------

_polar_original_sdpa: Any = None
_polar_patched_fn: Any = None
_polar_patched = False


def _apply_polar_patch() -> None:
    """Patch mlx_lm SDPA to accept TurboQuantKVCache (standard path)."""
    global _polar_original_sdpa, _polar_patched_fn, _polar_patched
    if _polar_patched:
        return
    import mlx_lm.models.base as base

    _polar_original_sdpa = base.scaled_dot_product_attention

    def _patched(queries, keys, values, cache, scale, mask):
        # TurboQuantKVCache: update_and_fetch returns dequantized arrays
        # so keys/values here are plain mx.array — standard SDPA applies.
        return _polar_original_sdpa(queries, keys, values, cache, scale, mask)

    _polar_patched_fn = _patched
    base.scaled_dot_product_attention = _patched
    # Also patch any already-imported model modules
    import sys as _sys
    for _name, _mod in list(_sys.modules.items()):
        if _name.startswith("mlx_lm.models.") and _mod is not None:
            if (
                hasattr(_mod, "scaled_dot_product_attention")
                and _mod.scaled_dot_product_attention
                is _polar_original_sdpa
            ):
                _mod.scaled_dot_product_attention = _patched
    _polar_patched = True


def _revert_polar_patch() -> None:
    global _polar_original_sdpa, _polar_patched_fn, _polar_patched
    if not _polar_patched or _polar_original_sdpa is None:
        return
    import sys as _sys

    import mlx_lm.models.base as base

    orig = _polar_original_sdpa
    patched = _polar_patched_fn
    # Safety: only revert if our patch is still in place.
    # If another candidate patched after us, leave their patch alone.
    if base.scaled_dot_product_attention is patched:
        base.scaled_dot_product_attention = orig
    for _name, _mod in list(_sys.modules.items()):
        if _name.startswith("mlx_lm.models.") and _mod is not None:
            if hasattr(_mod, "scaled_dot_product_attention"):
                if getattr(_mod, "scaled_dot_product_attention") is patched:
                    _mod.scaled_dot_product_attention = orig
    _polar_patched = False
    _polar_original_sdpa = None
    _polar_patched_fn = None


class PolarReferenceAdapter(KVCompressionCandidate):
    """PolarQuant reference adapter — real codebook-based KV compression.

    Uses TurboQuantKVCache from external/mlx-turboquant, which internally
    applies PolarQuant (rotation + Lloyd-Max codebook quantization) to
    each key and value vector.
    """

    name = "polar_reference_offline"
    candidate_status = CandidateStatus.REFERENCE_ONLY

    def __init__(self, bits: int = 4, dim: int = 128, seed: int = 42) -> None:
        self.bits = bits
        self.dim = dim
        self.seed = seed
        self.name = f"polar_reference_offline_b{bits}_d{dim}"

    def is_available(self) -> bool:
        try:
            _ensure_ext_on_path()
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            from mlx_turboquant.cache import TurboQuantKVCache  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _detect_head_dim(model: Any) -> int:
        args = getattr(model, "args", None)
        if args is not None:
            hd = getattr(args, "head_dim", None)
            if hd is not None:
                return int(hd)
            hidden = getattr(args, "hidden_size", None)
            n_heads = getattr(args, "num_attention_heads", None)
            if hidden and n_heads:
                return hidden // n_heads
        return 128  # sensible fallback

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> Any:
        """Generate with Polar cache and return log-probability array."""
        if not self.is_available():
            return None
        try:
            _ensure_ext_on_path()
            _patch_polar_quant_vectorize()
            import mlx.core as mx
            import numpy as np
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step
            from mlx_turboquant.cache import TurboQuantKVCache

            head_dim = self._detect_head_dim(model)
            n_layers = len(model.layers)
            caches = [
                TurboQuantKVCache(
                    bits=self.bits,
                    head_dim=head_dim,
                    key_seed=self.seed + i,
                    value_seed=self.seed + i + 1000,
                )
                for i in range(n_layers)
            ]

            _apply_polar_patch()
            try:
                input_ids = mx.array(tokenizer.encode(prompt))
                sampler = make_sampler(temp=temp)
                logprob_list: list[Any] = []
                for _token, log_probs in generate_step(
                    prompt=input_ids,
                    model=model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=caches,
                ):
                    lp_np = np.array(log_probs.astype(mx.float32))
                    logprob_list.append(lp_np)
                    if len(logprob_list) >= max_tokens:
                        break
                if not logprob_list:
                    return None
                return np.stack(logprob_list, axis=0)
            finally:
                _revert_polar_patch()
        except Exception:
            return None

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                error="mlx-turboquant or mlx/mlx_lm not available",
            )
        try:
            _ensure_ext_on_path()
            # Apply vectorized quantize patch before importing the cache
            # (TurboQuantKVCache imports PolarQuant at instantiation time)
            _patch_polar_quant_vectorize()
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step
            from mlx_turboquant.cache import TurboQuantKVCache

            head_dim = self._detect_head_dim(model)
            n_layers = len(model.layers)

            _apply_polar_patch()
            try:
                caches = [
                    TurboQuantKVCache(
                        bits=self.bits,
                        head_dim=head_dim,
                        key_seed=self.seed + i,
                        value_seed=self.seed + i + 1000,
                    )
                    for i in range(n_layers)
                ]

                input_ids = mx.array(tokenizer.encode(prompt))
                sampler = make_sampler(temp=temp)

                tokens: list[int] = []
                t0 = time.perf_counter()
                for token, _logprobs in generate_step(
                    prompt=input_ids,
                    model=model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=caches,
                ):
                    tok_id = (
                        token.item() if hasattr(token, "item") else int(token)
                    )
                    if tok_id in tokenizer.eos_token_ids:
                        break
                    tokens.append(tok_id)

                total_ms = (time.perf_counter() - t0) * 1000

                gen_tokens = max(len(tokens), 1)
                tps = gen_tokens / (total_ms / 1000)
                generated_text = tokenizer.decode(tokens)

                # Real compression ratio from cache memory
                compressed_bytes = sum(c.nbytes for c in caches)
                fp16_bytes = sum(
                    c._key_indices.shape[2] * head_dim * 2 * 2
                    if c._key_indices is not None
                    else 0
                    for c in caches
                )
                if fp16_bytes > 0 and compressed_bytes > 0:
                    size_ratio = round(compressed_bytes / fp16_bytes, 4)
                    compression_factor = round(fp16_bytes / compressed_bytes, 3)
                    actual_kv_memory_mb = compressed_bytes / (1024 * 1024)
                else:
                    size_ratio = None
                    compression_factor = None
                    actual_kv_memory_mb = None

            finally:
                _revert_polar_patch()

            return CandidateResult(
                name=self.name,
                model_id=getattr(
                    getattr(model, "args", model), "model_type", "unknown"
                ),
                prompt=prompt,
                total_ms=total_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                generated_text=generated_text,
                actual_kv_memory_mb=actual_kv_memory_mb,
                size_ratio=size_ratio,
                compression_factor=compression_factor,
                gate_status=GATE_STATUS_PENDING_LOGIT_GATE,
                candidate_status=self.candidate_status,
                cache_backend_used="polar_reference_dequant_on_fetch",
                cache_events=["prefetch_dequant", "decode_dequant", "attention_fp16"],
                cache_bytes_written=int(compressed_bytes),
                cache_bytes_read=int(compressed_bytes),
                notes=(
                    "Polar reference dequantizes on fetch. It is a reference candidate "
                    "and is not promotion eligible unless full generation, logit, and "
                    "working-set metrics pass."
                ),
            )
        except Exception as exc:
            try:
                _revert_polar_patch()
            except Exception:
                pass
            return CandidateResult(
                name=self.name,
                model_id="unknown",
                prompt=prompt,
                gate_status="ERROR",
                candidate_status=CandidateStatus.FAILED,
                error=str(exc),
            )
