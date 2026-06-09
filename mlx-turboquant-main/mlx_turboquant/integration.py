"""
Integration with mlx-lm generate pipeline.

Patches scaled_dot_product_attention to apply QJL correction when
a TurboQuantKVCache is detected.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .cache import TurboQuantKVCache


_original_sdpa = None


def _turboquant_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """Patched SDPA that applies QJL correction for TurboQuant caches."""
    if not hasattr(cache, 'turbo_bits') or not hasattr(cache, '_current_signs'):
        # Not a TurboQuant cache, use original
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    compressor = cache.compressor
    if not compressor.use_qjl:
        # No QJL correction, standard path
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    B, n_q_heads, L_q, D = queries.shape
    n_kv_heads = keys.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    queries_scaled = queries * scale
    qjl = compressor.qjl

    # Base scores from dequantized keys
    if n_repeats > 1:
        queries_r = mx.reshape(queries_scaled, (B, n_kv_heads, n_repeats, L_q, D))
        keys_t = mx.swapaxes(keys, -2, -1)
        keys_t = mx.expand_dims(keys_t, axis=2)
        base_scores = queries_r @ keys_t
    else:
        queries_r = queries_scaled
        base_scores = queries_r @ mx.swapaxes(keys, -2, -1)

    # QJL correction
    signs = cache._current_signs
    residual_norms = cache._current_residual_norms
    sign_values = mx.where(signs, 1.0, -1.0).astype(queries.dtype)

    if n_repeats > 1:
        proj_q = queries_r @ qjl.projection.T.astype(queries.dtype)
        sign_t = mx.swapaxes(sign_values, -2, -1)
        sign_t = mx.expand_dims(sign_t, axis=2)
        correction = proj_q @ sign_t
        rn_t = mx.swapaxes(residual_norms, -2, -1).astype(queries.dtype)
        rn_t = mx.expand_dims(rn_t, axis=2)
    else:
        proj_q = queries_r @ qjl.projection.T.astype(queries.dtype)
        correction = proj_q @ mx.swapaxes(sign_values, -2, -1)
        rn_t = mx.swapaxes(residual_norms, -2, -1).astype(queries.dtype)

    correction = correction * rn_t * qjl.correction_scale
    scores = base_scores + correction

    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L_kv = keys.shape[2]
            q_off = L_kv - L_q
            q_indices = mx.arange(q_off, q_off + L_q)
            k_indices = mx.arange(L_kv)
            mask = q_indices[:, None] >= k_indices[None]
        if hasattr(mask, 'dtype'):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores += mask

    weights = mx.softmax(scores, axis=-1, precise=True)

    # Value computation
    if n_repeats > 1:
        values_e = mx.expand_dims(values, axis=2)
        out = weights @ values_e
        out = mx.reshape(out, (B, n_q_heads, L_q, D))
    else:
        out = weights @ values

    return out


_patched_modules = []


def patch_sdpa(model=None):
    """Monkey-patch scaled_dot_product_attention for TurboQuant.

    If model is provided, patches only the specific model's module.
    Otherwise patches base and all already-imported model modules.
    """
    global _original_sdpa
    import sys
    import mlx_lm.models.base as base_module

    if _original_sdpa is None:
        _original_sdpa = base_module.scaled_dot_product_attention

    base_module.scaled_dot_product_attention = _turboquant_sdpa

    # Patch already-imported model modules
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, 'scaled_dot_product_attention'):
                if mod.scaled_dot_product_attention is not _turboquant_sdpa:
                    _patched_modules.append((mod, mod.scaled_dot_product_attention))
                    mod.scaled_dot_product_attention = _turboquant_sdpa


def unpatch_sdpa():
    """Restore original scaled_dot_product_attention in all patched modules."""
    global _original_sdpa
    if _original_sdpa is not None:
        import mlx_lm.models.base as base_module
        base_module.scaled_dot_product_attention = _original_sdpa
        for mod, orig_fn in _patched_modules:
            mod.scaled_dot_product_attention = orig_fn
        _patched_modules.clear()
        _original_sdpa = None


def make_turboquant_cache(
    model: nn.Module,
    bits: int = 3,
    head_dim: int | None = None,
) -> list:
    """Create TurboQuantKVCache instances for each layer and patch SDPA.

    Args:
        model: mlx-lm model
        bits: TurboQuant bits per coordinate (2-4)
        head_dim: Head dimension (auto-detected if None)

    Returns:
        List of TurboQuantKVCache, one per layer
    """
    num_layers = len(model.layers)

    if head_dim is None:
        args = model.args
        head_dim = getattr(args, 'head_dim', None)
        if head_dim is None and hasattr(args, 'hidden_size') and hasattr(args, 'num_attention_heads'):
            head_dim = args.hidden_size // args.num_attention_heads
        if head_dim is None:
            raise ValueError("Could not auto-detect head_dim.")

    # Patch SDPA for QJL correction
    patch_sdpa()

    return [
        TurboQuantKVCache(bits=bits, head_dim=head_dim)
        for _ in range(num_layers)
    ]
