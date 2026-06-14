"""Verify RFSN v11 cache injection contract without global monkey-patching."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_cache_injector_exists():
    from rfsn_v11.generation.cache_injection import RFSNV11CacheInjector
    injector = RFSNV11CacheInjector()
    assert injector is not None
    assert injector.key_bits == 8
    assert injector.value_bits == 5


@pytest.mark.unit
def test_minimal_decode_loop_importable():
    from rfsn_v11.generation.minimal_decode import minimal_decode_loop
    assert callable(minimal_decode_loop)


@pytest.mark.mlx
def test_no_global_monkey_patch():
    """Verify that importing the generation module does not patch mlx_lm globally."""
    base_before = pytest.importorskip("mlx_lm.models.base")
    original_sdpa = base_before.scaled_dot_product_attention

    # Import should not change anything globally
    from rfsn_v11.generation import cache_injection, minimal_decode  # noqa: F401

    base_after = pytest.importorskip("mlx_lm.models.base")
    assert base_after.scaled_dot_product_attention is original_sdpa
