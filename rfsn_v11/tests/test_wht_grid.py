"""Tests for WHT64 Metal kernel correctness.

Phase 5 — 5-4:
  - WHT Metal kernel correctness vs _wht_block_recursive to 1e-5 for n=128, 256, 1024.
  - MLX grid dispatch note: grid=(n, 1, 1) with threadgroup=(64, 1, 1) dispatches
    n/64 threadgroups of 64 threads each. This is CORRECT for MLX's API where
    grid specifies TOTAL threads (not threadgroup count).
  - Self-inverse property: WHT(WHT(x)) == x.
"""
import math
import pytest
import numpy as np

mx = pytest.importorskip("mlx.core")

from rfsn_v11.quant.key_quant import KeyQuant, wht64_metal, maybe_supports_metal_kernels  # noqa: E402


@pytest.mark.parametrize("n", [128, 256, 512, 1024])
def test_wht_python_self_inverse(n):
    """WHT(WHT(x)) ≈ x for Python fallback (up to floating-point)."""
    kq = KeyQuant(bits=8, use_wht=True, prefer_metal=False)
    rng = np.random.RandomState(42)
    x = mx.array(rng.randn(n).astype(np.float32))
    x_wht = kq._apply_wht_pretransform(x)
    x_restored = kq._apply_wht_pretransform(x_wht)
    mx.eval(x_restored)
    err = float(mx.max(mx.abs(x - x_restored)).item())
    assert err < 1e-4, f"WHT self-inverse error too large: {err}"


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_wht_python_vs_reference(n):
    """Python WHT fallback must produce the standard normalized WHT output.

    Reference: Python recursive WHT / sqrt(64) applied block-by-block.
    """
    kq = KeyQuant(bits=8, use_wht=True, prefer_metal=False)
    rng = np.random.RandomState(n)
    x_np = rng.randn(n).astype(np.float32)
    x = mx.array(x_np)

    # Reference: numpy WHT applied block-by-block
    def np_wht64_block(block: np.ndarray) -> np.ndarray:
        """Recursive WHT on 64 values, normalized by 1/sqrt(64)."""
        n_block = len(block)
        if n_block == 1:
            return block
        h = n_block // 2
        a, b = block[:h], block[h:]
        y0 = np_wht64_block(a + b)
        y1 = np_wht64_block(a - b)
        return np.concatenate([y0, y1])

    expected = np.zeros_like(x_np)
    for i in range(n // 64):
        block = x_np[i * 64 : (i + 1) * 64]
        expected[i * 64 : (i + 1) * 64] = np_wht64_block(block) / math.sqrt(64)

    result = np.array(kq._apply_wht_pretransform(x.reshape(n // 64, 64)).reshape(-1))
    mx.eval()
    err = np.max(np.abs(result - expected))
    assert err < 1e-5, f"WHT output mismatch: max_err={err}"


@pytest.mark.skipif(
    not maybe_supports_metal_kernels(),
    reason="Metal kernel API not available",
)
@pytest.mark.parametrize("n", [128, 256, 1024])
def test_wht_metal_matches_python(n):
    """Metal WHT kernel must match Python fallback to 1e-5."""
    kq_metal = KeyQuant(bits=8, use_wht=True, prefer_metal=True)
    kq_py = KeyQuant(bits=8, use_wht=True, prefer_metal=False)

    rng = np.random.RandomState(n + 1)
    x = mx.array(rng.randn(n).astype(np.float32))

    # Metal path
    out_metal = wht64_metal(x.reshape(n // 64, 64))
    # Python path
    out_py = kq_py._apply_wht_pretransform(x.reshape(n // 64, 64))
    mx.eval(out_metal, out_py)

    err = float(mx.max(mx.abs(out_metal - out_py)).item())
    assert err < 1e-5, (
        f"Metal WHT vs Python mismatch: n={n}, max_err={err}. "
        "This may indicate the grid dispatch bug has been reintroduced."
    )


def test_wht_grid_dispatch_assertion():
    """The WHT kernel must reject inputs whose size is not a multiple of 64."""
    with pytest.raises((ValueError, AssertionError)):
        wht64_metal(mx.array(np.ones(100, dtype=np.float32)))


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_key_compress_decompress_identity(n):
    """KeyQuant compress → decompress should approximately reconstruct (cosine ≥ 0.99)."""
    kq = KeyQuant(bits=8, group_size=64, use_wht=True, use_incoherent_signs=True)
    rng = np.random.RandomState(0)
    x_np = rng.randn(n).astype(np.float32)
    x = mx.array(x_np)

    codes, scales = kq.compress(x)
    x_rec = kq.decompress(codes, scales, x.shape)
    mx.eval(x_rec)

    x_rec_np = np.array(x_rec)
    cosine = np.dot(x_np, x_rec_np) / (
        np.linalg.norm(x_np) * np.linalg.norm(x_rec_np) + 1e-8
    )
    assert cosine >= 0.99, f"KeyQuant cosine too low: {cosine:.4f} for n={n}"
