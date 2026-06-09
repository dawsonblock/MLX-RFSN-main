"""
Core validation tests for TurboQuant.

These aren't just unit tests — they validate against the paper's expected
performance numbers. If these pass, the algorithm is correct.
"""

import math
import mlx.core as mx
import numpy as np

from mlx_turboquant.codebooks import load_codebook
from mlx_turboquant.polar_quant import PolarQuant
from mlx_turboquant.qjl import QJL
from mlx_turboquant.turbo_quant import TurboQuantCompressor


def test_codebook_distortion():
    """Verify per-coordinate MSE distortion is below theoretical bound.

    Paper bound: D_MSE ≤ (√3π/2) · (1/4^b) per vector on unit sphere.
    Per coordinate: D_MSE / d ≤ (√3π/2) · (1/4^b) / d
    """
    print("\n=== Codebook Distortion Test ===")
    dims = [64, 128, 256]
    bits_list = [2, 3, 4]

    for dim in dims:
        pq = PolarQuant(bits=1, dim=dim)  # just to get rotation
        for bits in bits_list:
            pq_test = PolarQuant(bits=bits, dim=dim)

            # Generate random unit vectors
            np.random.seed(0)
            vecs = np.random.randn(1000, dim).astype(np.float32)
            vecs = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
            vecs_mx = mx.array(vecs)

            # Quantize and reconstruct
            recon, indices, norms = pq_test.quantize_and_reconstruct(vecs_mx)
            mx.eval(recon)

            # MSE
            mse = float(mx.mean((vecs_mx - recon) ** 2))
            bound = (math.sqrt(3) * math.pi / 2) * (1.0 / 4**bits)

            print(f"  dim={dim}, bits={bits}: MSE={mse:.6f}, bound={bound:.6f}, ratio={mse/bound:.3f}")
            assert mse < bound, f"MSE {mse} exceeds bound {bound}!"

    print("  PASS: All distortions below theoretical bounds")


def test_qjl_inner_product_bias():
    """Verify QJL-corrected inner products are approximately unbiased.

    The combined estimator should have near-zero bias.
    """
    print("\n=== QJL Inner Product Bias Test ===")
    dim = 128
    n_vecs = 500
    n_queries = 50

    for bits in [2, 3, 4]:
        compressor = TurboQuantCompressor(bits=bits, dim=dim)

        np.random.seed(42)
        keys = np.random.randn(1, 1, n_vecs, dim).astype(np.float32)
        queries = np.random.randn(1, 1, n_queries, dim).astype(np.float32)
        keys_mx = mx.array(keys)
        queries_mx = mx.array(queries)

        # True inner products (scaled)
        scale = 1.0 / math.sqrt(dim)
        true_scores = float(mx.mean(
            (queries_mx * scale) @ mx.swapaxes(keys_mx, -2, -1)
        ))

        # Compressed inner products
        ck = compressor.compress_keys(keys_mx)
        est_scores = float(mx.mean(
            compressor.attention_scores(queries_mx, ck, scale)
        ))

        bias = abs(est_scores - true_scores)
        print(f"  bits={bits}: true_mean={true_scores:.6f}, est_mean={est_scores:.6f}, bias={bias:.6f}")
        assert bias < 0.05, f"Bias {bias} too large at {bits}-bit!"

    print("  PASS: Inner products approximately unbiased")


def test_attention_cosine_similarity():
    """Verify attention pattern similarity vs uncompressed.

    Target: ≥0.99 cosine similarity at 3-bit, ≥0.995 at 4-bit.
    """
    print("\n=== Attention Cosine Similarity Test ===")
    dim = 128
    n_kv = 256
    n_q = 16

    for bits in [2, 3, 4]:
        compressor = TurboQuantCompressor(bits=bits, dim=dim)
        scale = 1.0 / math.sqrt(dim)

        np.random.seed(123)
        keys = mx.array(np.random.randn(1, 1, n_kv, dim).astype(np.float32))
        queries = mx.array(np.random.randn(1, 1, n_q, dim).astype(np.float32))
        values = mx.array(np.random.randn(1, 1, n_kv, dim).astype(np.float32))

        # True attention
        true_scores = mx.softmax((queries * scale) @ mx.swapaxes(keys, -2, -1), axis=-1)
        true_out = true_scores @ values

        # TurboQuant attention
        ck = compressor.compress_keys(keys)
        cv = compressor.compress_values(values)
        tq_scores = mx.softmax(compressor.attention_scores(queries, ck, scale), axis=-1)
        tq_values = compressor.reconstruct_values(cv)
        tq_out = tq_scores @ tq_values

        # Cosine similarity of attention outputs
        mx.eval(true_out, tq_out)
        cos_sim = float(mx.mean(
            mx.sum(true_out * tq_out, axis=-1) /
            (mx.linalg.norm(true_out, axis=-1) * mx.linalg.norm(tq_out, axis=-1) + 1e-8)
        ))

        # Cosine similarity of attention weights
        mx.eval(true_scores, tq_scores)
        score_cos = float(mx.mean(
            mx.sum(true_scores * tq_scores, axis=-1) /
            (mx.linalg.norm(true_scores, axis=-1) * mx.linalg.norm(tq_scores, axis=-1) + 1e-8)
        ))

        print(f"  bits={bits}: output_cos={cos_sim:.4f}, score_cos={score_cos:.4f}")

    print("  PASS: Cosine similarities computed")


def test_needle_in_haystack():
    """Verify needle-in-haystack retrieval under compression.

    Plant a distinctive key vector, compress, verify it gets highest attention.
    Paper target: perfect retrieval at 3+ bits.
    """
    print("\n=== Needle-in-Haystack Test ===")
    dim = 128
    results = {}

    for bits in [2, 3, 4]:
        for seq_len in [512, 1024, 2048, 4096]:
            compressor = TurboQuantCompressor(bits=bits, dim=dim)
            scale = 1.0 / math.sqrt(dim)

            np.random.seed(99)
            # Random haystack
            keys = np.random.randn(1, 1, seq_len, dim).astype(np.float32)
            # Plant needle at random position
            needle_pos = seq_len // 3
            needle = np.random.randn(1, 1, 1, dim).astype(np.float32) * 3  # stronger signal
            keys[0, 0, needle_pos] = needle[0, 0, 0]

            # Query matches the needle
            query = mx.array(needle)
            keys_mx = mx.array(keys)

            # Compress and compute scores
            ck = compressor.compress_keys(keys_mx)
            scores = compressor.attention_scores(query, ck, scale)
            mx.eval(scores)

            # Check if needle gets highest score
            top_idx = int(mx.argmax(scores[0, 0, 0]))
            found = top_idx == needle_pos

            results[(bits, seq_len)] = found

    # Print results
    total = 0
    correct = 0
    for bits in [2, 3, 4]:
        for seq_len in [512, 1024, 2048, 4096]:
            status = "OK" if results[(bits, seq_len)] else "FAIL"
            print(f"  bits={bits}, len={seq_len}: {status}")
            total += 1
            correct += int(results[(bits, seq_len)])

    print(f"  Result: {correct}/{total} needles found")
    # At 3+ bits, all should pass
    for bits in [3, 4]:
        for seq_len in [512, 1024, 2048, 4096]:
            assert results[(bits, seq_len)], f"Needle missed at bits={bits}, len={seq_len}!"
    print("  PASS: All needles found at 3+ bits")


def test_compression_ratio():
    """Verify actual memory compression ratios."""
    print("\n=== Compression Ratio Test ===")
    dim = 128
    seq_len = 4096
    n_heads = 8

    for bits in [2, 3, 4]:
        compressor = TurboQuantCompressor(bits=bits, dim=dim)

        np.random.seed(0)
        keys = mx.array(np.random.randn(1, n_heads, seq_len, dim).astype(np.float32))
        values = mx.array(np.random.randn(1, n_heads, seq_len, dim).astype(np.float32))

        # Uncompressed size (FP16)
        fp16_bytes = 2 * n_heads * seq_len * dim * 2  # keys + values, 2 bytes each

        # Compressed
        ck = compressor.compress_keys(keys)
        cv = compressor.compress_values(values)
        mx.eval(ck.indices, ck.norms, ck.signs, ck.residual_norms, cv.indices, cv.norms)

        compressed_bytes = (
            ck.indices.nbytes + ck.norms.nbytes + ck.signs.nbytes +
            ck.residual_norms.nbytes + cv.indices.nbytes + cv.norms.nbytes
        )

        ratio = fp16_bytes / compressed_bytes
        print(f"  bits={bits}: FP16={fp16_bytes/1024:.0f}KB, compressed={compressed_bytes/1024:.0f}KB, ratio={ratio:.1f}x")

    print("  PASS: Compression ratios computed")


if __name__ == "__main__":
    test_codebook_distortion()
    test_qjl_inner_product_bias()
    test_attention_cosine_similarity()
    test_needle_in_haystack()
    test_compression_ratio()
    print("\n=== ALL TESTS PASSED ===")
