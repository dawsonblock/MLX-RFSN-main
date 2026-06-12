"""Tests for IncrementalPolarCache.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_append_and_attend() -> None:
    """Basic append + attend with incremental cache."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
    )

    # Append 10 tokens
    keys = mx.random.normal(shape=(1, 2, 10, 64))
    values = mx.random.normal(shape=(1, 2, 10, 64))
    cache.append(keys, values)

    assert cache.token_count == 10

    # Attend with 1 query
    queries = mx.random.normal(shape=(1, 14, 1, 64))  # 14 query heads, GQA 7x
    output = cache.attend_naive(queries)
    assert output.shape == (1, 14, 1, 64)
    assert mx.all(mx.isfinite(output)).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_multi_append() -> None:
    """Multiple appends should accumulate tokens."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
    )

    # Append 5 tokens
    cache.append(mx.random.normal((1, 2, 5, 64)), mx.random.normal((1, 2, 5, 64)))
    assert cache.token_count == 5

    # Append 3 more
    cache.append(mx.random.normal((1, 2, 3, 64)), mx.random.normal((1, 2, 3, 64)))
    assert cache.token_count == 8

    # Attend
    queries = mx.random.normal(shape=(1, 14, 1, 64))
    output = cache.attend_naive(queries)
    assert output.shape == (1, 14, 1, 64)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_matches_full_quantize() -> None:
    """Incremental cache output should match full-cache quantize output."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.attention import NaivePolarAttention
    from rfsn_v11.polar_fused.contracts import QuantizedVectors
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    # Full keys/values
    B, Hkv, T, D = 1, 2, 10, 64
    keys = mx.random.normal(shape=(B, Hkv, T, D))
    values = mx.random.normal(shape=(B, Hkv, T, D))
    queries = mx.random.normal(shape=(1, 14, 1, D))

    # Full quantize path
    key_qv_raw = key_q.quantize(keys.reshape(-1, D))
    value_qv_raw = value_q.quantize(values.reshape(-1, D))
    key_qv = QuantizedVectors(
        indices=key_qv_raw.indices.reshape(B, Hkv, T, D),
        norms=key_qv_raw.norms.reshape(B, Hkv, T),
        original_dim=D, bits=cfg.key_bits,
        rotation_id=key_q._rotation_id, codebook_id=key_q._codebook_id,
    )
    value_qv = QuantizedVectors(
        indices=value_qv_raw.indices.reshape(B, Hkv, T, D),
        norms=value_qv_raw.norms.reshape(B, Hkv, T),
        original_dim=D, bits=cfg.value_bits,
        rotation_id=value_q._rotation_id, codebook_id=value_q._codebook_id,
    )
    attn = NaivePolarAttention(key_q, value_q)
    full_output = attn.attend(queries, key_qv, value_qv).output

    # Incremental path
    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=Hkv,
        head_dim=D,
    )
    cache.append(keys, values)
    inc_output = cache.attend_naive(queries)

    # Should match (same quantized data)
    diff = mx.max(mx.abs(full_output - inc_output)).item()
    assert diff < 1e-4, f"Incremental diverged from full: max_diff={diff}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_memory_grows_correctly() -> None:
    """Memory should grow proportionally with token count."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
    )

    mem0 = cache.memory_bytes()
    assert mem0 == 0

    cache.append(mx.random.normal((1, 2, 10, 64)), mx.random.normal((1, 2, 10, 64)))
    mem10 = cache.memory_bytes()

    cache.append(mx.random.normal((1, 2, 10, 64)), mx.random.normal((1, 2, 10, 64)))
    mem20 = cache.memory_bytes()

    # Memory should roughly double
    assert mem20 > mem10
    assert mem20 == mem10 * 2


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_incremental_kernel_path() -> None:
    """Kernel path should produce output without error."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
    )

    cache.append(mx.random.normal((1, 2, 10, 64)), mx.random.normal((1, 2, 10, 64)))
    queries = mx.random.normal(shape=(1, 14, 1, 64))

    output = cache.attend_kernel(queries)
    assert output.shape == (1, 14, 1, 64)
    assert mx.all(mx.isfinite(output)).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_kernel_matches_naive() -> None:
    """Kernel path output should match naive path (quality gate)."""
    from rfsn_v11.polar_fused.config import PolarFusedConfig
    from rfsn_v11.polar_fused.quantize import PolarQuantizer
    from rfsn_v11.polar_fused.incremental_cache import IncrementalPolarCache

    cfg = PolarFusedConfig.polar_safe()
    key_q = PolarQuantizer(bits=cfg.key_bits, head_dim=64, rotation_seed=cfg.key_rotation_seed)
    value_q = PolarQuantizer(bits=cfg.value_bits, head_dim=64, rotation_seed=cfg.value_rotation_seed)

    cache = IncrementalPolarCache(
        key_quantizer=key_q,
        value_quantizer=value_q,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
    )

    # Use fixed random seed for determinism
    mx.random.seed(42)
    keys = mx.random.normal((1, 2, 10, 64))
    values = mx.random.normal((1, 2, 10, 64))
    queries = mx.random.normal((1, 14, 1, 64))

    cache.append(keys, values)

    naive_out = cache.attend_naive(queries)
    kernel_out = cache.attend_kernel(queries)

    assert naive_out.shape == kernel_out.shape
    diff = mx.max(mx.abs(naive_out - kernel_out)).item()
    assert diff < 1e-3, f"Kernel diverged from naive: max_diff={diff}"

    # Cosine similarity
    cos = _cosine_similarity(naive_out, kernel_out)
    assert cos.item() > 0.99, f"Kernel cosine similarity too low: {cos.item()}"


def _cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a * b)
    na = mx.sqrt(mx.sum(a * a))
    nb = mx.sqrt(mx.sum(b * b))
    return dot / (na * nb)
