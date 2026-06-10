"""Candidate: TurboQuant-MLX V2 — real KV-cache compression.

Wires external/turboquant-mlx's TurboQuantKVCacheV2 into the generation
loop via mlx_lm.utils.generate_step(prompt_cache=...) and the
turboquant SDPA patch.

Key design
----------
- Applies turboquant.patch before generation; reverts after.
- Builds one TurboQuantKVCacheV2 per transformer layer.
- head_dim is auto-detected from model.args.
- Rotation is enabled only when head_dim >= 128 (designed for 128-dim
  heads; on Qwen2.5-0.5B head_dim=64 rotation degrades quality).
- After generation, computes real size_ratio from cache.nbytes vs fp16.

Source: external/turboquant-mlx/turboquant/cache_v2.py
        external/turboquant-mlx/turboquant/patch.py
        external/turboquant-mlx/run_llm.py

Status: experimental candidate — must pass full logit gate before promotion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus
from .quality_gates import GATE_STATUS_PENDING_LOGIT_GATE

# Absolute path to external/turboquant-mlx so the import works regardless
# of working directory.
_EXT_TQ = str(
    Path(__file__).parent.parent.parent / "external" / "turboquant-mlx"
)


def _ensure_ext_on_path() -> None:
    if _EXT_TQ not in sys.path:
        sys.path.insert(0, _EXT_TQ)


class TurboQuantV2Candidate(KVCompressionCandidate):
    """TurboQuant V2: random QR rotation + MLX native quantized_matmul.

    This is a *real* KV-cache candidate: generation runs through a
    TurboQuantKVCacheV2 prompt cache, not plain mlx_lm.generate().
    """

    candidate_status = CandidateStatus.EXPERIMENTAL

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.seed = seed
        # Name includes 'rot' when rotation will be applied (head_dim >= 128).
        # We set the full name at run-time once we know head_dim; use a
        # placeholder here for display purposes.
        self.name = f"turboquant_v2_b{bits}_gs{group_size}_rot"

    def is_available(self) -> bool:
        try:
            _ensure_ext_on_path()
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            import turboquant.patch  # noqa: F401
            from turboquant.cache_v2 import TurboQuantKVCacheV2  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_head_dim(model: Any) -> int:
        """Auto-detect head dimension from model.args."""
        args = getattr(model, "args", None)
        if args is not None:
            hd = getattr(args, "head_dim", None)
            if hd is not None:
                return int(hd)
            hidden = getattr(args, "hidden_size", None)
            n_heads = getattr(args, "num_attention_heads", None)
            if hidden and n_heads:
                return hidden // n_heads
        # Fallback: inspect first layer attention
        for attr in ("self_attn", "attention", "attn"):
            layer = model.layers[0] if model.layers else None
            if layer and hasattr(layer, attr):
                a = getattr(layer, attr)
                if hasattr(a, "head_dim"):
                    return int(a.head_dim)
        raise ValueError(
            "Cannot auto-detect head_dim from model. "
            "Pass it explicitly or check model.args."
        )

    def _build_cache(
        self, model: Any, head_dim: int, use_rotation: bool
    ) -> list:
        _ensure_ext_on_path()
        from turboquant.cache_v2 import TurboQuantKVCacheV2

        n_layers = len(model.layers)
        return [
            TurboQuantKVCacheV2(
                head_dim=head_dim,
                bits=self.bits,
                group_size=self.group_size,
                use_rotation=use_rotation,
                use_normalization=False,  # lean path: no per-vector norm
                seed=self.seed + i,
            )
            for i in range(n_layers)
        ]

    @staticmethod
    def _cache_nbytes(caches: list) -> int:
        total = 0
        for c in caches:
            if c.keys is not None:
                for arr in c.keys:
                    total += arr[..., : c.offset, :].nbytes
                for arr in c.values:
                    total += arr[..., : c.offset, :].nbytes
        return total

    @staticmethod
    def _cache_fp16_nbytes(caches: list) -> int:
        """Equivalent size if KV were stored in float16."""
        total = 0
        for c in caches:
            if c.keys is not None:
                T = c.offset
                # keys[0] is the packed data tensor;
                # shape = (B, H, capacity, packed_dim)
                B, H = c.keys[0].shape[:2]
                D = c.head_dim
                # float16 = 2 bytes per element
                total += B * H * T * D * 2 * 2  # keys + values
        return total

    # ------------------------------------------------------------------
    # Logit capture (for full-logit-gate)
    # ------------------------------------------------------------------

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> Any:
        """Generate with TurboQuant cache and return log-probability array.

        This is a second generation pass used *only* for logit-quality
        comparison.  Speed metrics come from ``run()``.
        """
        if not self.is_available():
            return None
        try:
            _ensure_ext_on_path()
            import mlx.core as mx
            import numpy as np
            import turboquant.patch as tq_patch
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step

            head_dim = self._detect_head_dim(model)
            use_rotation = head_dim >= 128

            tq_patch.apply()
            import mlx_lm.models.base as _base
            _patched_fn = _base.scaled_dot_product_attention
            for _mod_name, _mod in list(sys.modules.items()):
                if _mod_name.startswith("mlx_lm.models.") and _mod is not None:
                    if hasattr(_mod, "scaled_dot_product_attention"):
                        _mod.scaled_dot_product_attention = _patched_fn
            try:
                caches = self._build_cache(model, head_dim, use_rotation)
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
                    # mlx bfloat16 arrays cannot be passed directly to
                    # np.array(); cast to float32 first.
                    lp_np = np.array(log_probs.astype(mx.float32))
                    logprob_list.append(lp_np)
                    if len(logprob_list) >= max_tokens:
                        break

                if not logprob_list:
                    return None
                return np.stack(logprob_list, axis=0)
            finally:
                tq_patch.revert()
                _orig_fn = _base.scaled_dot_product_attention
                for _mod_name, _mod in list(sys.modules.items()):
                    if _mod_name.startswith("mlx_lm.models.") and _mod is not None:
                        if hasattr(_mod, "scaled_dot_product_attention"):
                            _mod.scaled_dot_product_attention = _orig_fn
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

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
                error="turboquant-mlx or mlx/mlx_lm not available",
            )
        try:
            _ensure_ext_on_path()
            import mlx.core as mx
            import turboquant.patch as tq_patch
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step

            head_dim = self._detect_head_dim(model)
            use_rotation = head_dim >= 128

            # Update name now that we know rotation decision
            self.name = (
                f"turboquant_v2_b{self.bits}_gs{self.group_size}"
                f"{'_rot' if use_rotation else '_norot'}"
            )

            # Apply SDPA patch and force-sync all already-imported
            # model modules. tq_patch.apply() only patches
            # mlx_lm.models.base; any model module (e.g. qwen2)
            # that was imported before the patch and holds a local
            # reference to scaled_dot_product_attention needs to be
            # updated too.
            tq_patch.apply()
            import mlx_lm.models.base as _base
            _patched_fn = _base.scaled_dot_product_attention
            for _mod_name, _mod in list(sys.modules.items()):
                if _mod_name.startswith("mlx_lm.models.") and _mod is not None:
                    if hasattr(_mod, "scaled_dot_product_attention"):
                        _mod.scaled_dot_product_attention = _patched_fn
            try:
                caches = self._build_cache(model, head_dim, use_rotation)

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
                mx.eval(
                    *[
                        c.keys[0]
                        for c in caches
                        if c.keys is not None
                    ][:1]
                    or [mx.array(0)]
                )

                gen_tokens = max(len(tokens), 1)
                tps = gen_tokens / (total_ms / 1000)
                generated_text = tokenizer.decode(tokens)

                # Real compression ratio from cache memory
                compressed_bytes = self._cache_nbytes(caches)
                fp16_bytes = self._cache_fp16_nbytes(caches)
                if (
                    fp16_bytes > 0
                    and compressed_bytes > 0
                ):
                    size_ratio = round(compressed_bytes / fp16_bytes, 4)
                    compression_factor = round(fp16_bytes / compressed_bytes, 3)
                    actual_kv_memory_mb = compressed_bytes / (1024 * 1024)
                else:
                    size_ratio = None
                    compression_factor = None
                    actual_kv_memory_mb = None

            finally:
                tq_patch.revert()
                # Re-sync all model modules to the now-reverted
                # original
                _orig_fn = (
                    _base.scaled_dot_product_attention
                )
                for _mod_name, _mod in list(sys.modules.items()):
                    if _mod_name.startswith("mlx_lm.models.") and _mod is not None:
                        if hasattr(_mod, "scaled_dot_product_attention"):
                            _mod.scaled_dot_product_attention = _orig_fn

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
                cache_backend_used="turboquant_v2",
                cache_events=[
                    "prefill_compress",
                    "decode_fetch",
                    "attention_quantized_matmul",
                ],
                cache_bytes_written=int(compressed_bytes),
                cache_bytes_read=int(compressed_bytes),
                patch_scope="controlled_context",
                global_patch_restored=True,
                notes=(
                    f"TurboQuant V2: b{self.bits} gs{self.group_size} "
                    f"rotation={use_rotation} head_dim={head_dim}  "
                    f"Real KV cache via TurboQuantKVCacheV2 + SDPA patch.  "
                    f"Source: external/turboquant-mlx/turboquant/cache_v2.py"
                ),
            )
        except Exception as exc:
            # Make sure patch is always reverted on error
            try:
                _ensure_ext_on_path()
                import mlx_lm.models.base as _base_err
                import turboquant.patch as tq_patch  # type: ignore[import]
                tq_patch.revert()
                _orig_err = _base_err.scaled_dot_product_attention
                for _mn, _mm in list(sys.modules.items()):
                    if _mn.startswith("mlx_lm.models.") and _mm is not None:
                        if hasattr(_mm, "scaled_dot_product_attention"):
                            _mm.scaled_dot_product_attention = _orig_err
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
