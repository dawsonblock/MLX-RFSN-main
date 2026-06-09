"""Tests for pack_indices / unpack_indices vectorized roundtrip.

Phase 5 — 5-3: Assert no Python loop in forward path.
The implementation uses mx.sum(shifted, axis=-1) instead of:
    packed = shifted[..., 0]
    for i in range(1, vals_per_int):
        packed = packed | shifted[..., i]
"""
import ast
import textwrap
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v11.quant.packing import pack_indices, unpack_indices, pack_signs, unpack_signs


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_pack_unpack_roundtrip_exact(bits, dim):
    """pack_indices → unpack_indices must be exactly lossless for bits=2,3,4."""
    rng = np.random.RandomState(bits * 100 + dim)
    for batch in [1, 4, 8]:
        idx_np = rng.randint(0, 2**bits, size=(batch, dim)).astype(np.uint8)
        idx = mx.array(idx_np)
        packed = pack_indices(idx, bits)
        unpacked = unpack_indices(packed, bits, dim)
        mx.eval(unpacked)
        result = np.array(unpacked)
        assert result.shape == (batch, dim), f"shape mismatch: {result.shape} vs {(batch, dim)}"
        assert np.array_equal(result, idx_np), (
            f"Roundtrip failed for bits={bits} dim={dim} batch={batch}: "
            f"max_err={np.max(np.abs(result.astype(int) - idx_np.astype(int)))}"
        )


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_pack_signs_roundtrip(dim):
    """pack_signs → unpack_signs roundtrip exact."""
    rng = np.random.RandomState(dim)
    signs_np = rng.randint(0, 2, size=(4, dim)).astype(bool)
    signs = mx.array(signs_np)
    packed = pack_signs(signs)
    unpacked = unpack_signs(packed, dim)
    mx.eval(unpacked)
    result = np.array(unpacked)
    assert np.array_equal(result, signs_np)


def test_no_python_loop_in_pack_indices():
    """Assert pack_indices source does NOT contain a for loop.

    The Python loop was the original bug:
        for i in range(1, vals_per_int):
            packed = packed | shifted[..., i]
    The fix uses mx.sum vectorized reduction.
    """
    import inspect
    from rfsn_v11.quant import packing
    src = inspect.getsource(packing.pack_indices)
    tree = ast.parse(textwrap.dedent(src))

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Get the loop variable name for a clear error message
            target_name = ast.unparse(node.target) if hasattr(ast, 'unparse') else "?"
            # Check if this is the problematic packing loop
            loop_src = ast.unparse(node) if hasattr(ast, 'unparse') else ""
            pytest.fail(
                f"pack_indices contains a Python for-loop (target={target_name!r}) "
                f"which causes sequential MLX graph nodes. "
                f"Expected vectorized mx.sum reduction.\nLoop:\n{loop_src[:200]}"
            )


def test_pack_indices_1d_input():
    """1-D input (no batch dim) works correctly."""
    rng = np.random.RandomState(42)
    idx_np = rng.randint(0, 16, size=(128,)).astype(np.uint8)
    idx = mx.array(idx_np)
    packed = pack_indices(idx, bits=4)
    unpacked = unpack_indices(packed, bits=4, dim=128)
    mx.eval(unpacked)
    assert np.array_equal(np.array(unpacked), idx_np)


def test_pack_indices_boundary_values():
    """Test with max values (2^bits - 1) to catch off-by-one in masking."""
    for bits in [2, 3, 4]:
        max_val = 2**bits - 1
        idx_np = np.full((1, 64), max_val, dtype=np.uint8)
        idx = mx.array(idx_np)
        packed = pack_indices(idx, bits)
        unpacked = unpack_indices(packed, bits, 64)
        mx.eval(unpacked)
        assert np.array_equal(np.array(unpacked), idx_np), f"Boundary failed at bits={bits}"

    idx_np = np.zeros((1, 64), dtype=np.uint8)
    for bits in [2, 3, 4]:
        packed = pack_indices(mx.array(idx_np), bits)
        unpacked = unpack_indices(packed, bits, 64)
        mx.eval(unpacked)
        assert np.array_equal(np.array(unpacked), idx_np), f"Zero boundary failed at bits={bits}"
