"""
Precompute Lloyd-Max optimal scalar quantizers for the Gaussian distribution
that approximates rotated unit vector coordinates.

After random orthogonal rotation, each coordinate of a d-dimensional unit vector
is approximately N(0, 1/d). We compute the optimal Lloyd-Max quantizer for N(0,1)
and scale by 1/sqrt(d) at runtime.

This gives dimension-independent codebooks — one per bit-width.
"""

import os
import numpy as np
from scipy import integrate
from scipy.stats import norm


def lloyd_max_gaussian(n_levels, sigma=1.0, max_iter=200, tol=1e-10):
    """Solve Lloyd-Max quantizer for Gaussian N(0, sigma^2).

    Args:
        n_levels: Number of quantization levels (2^bits)
        sigma: Standard deviation
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        centroids: (n_levels,) optimal reconstruction values
        boundaries: (n_levels+1,) decision boundaries
    """
    # Support: [-k*sigma, k*sigma] — covers 99.99% of mass
    k = 5.0
    support_min = -k * sigma
    support_max = k * sigma

    pdf = lambda x: norm.pdf(x, 0, sigma)

    # Initialize centroids uniformly
    centroids = np.linspace(support_min, support_max, n_levels + 2)[1:-1]

    for iteration in range(max_iter):
        # Update boundaries as midpoints of centroids
        boundaries = np.zeros(n_levels + 1)
        boundaries[0] = support_min
        boundaries[-1] = support_max
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        # Update centroids as conditional means
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
                continue

            num, _ = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=100)
            den, _ = integrate.quad(pdf, lo, hi, limit=100)

            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = (lo + hi) / 2.0

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = np.zeros(n_levels + 1)
    boundaries[0] = support_min
    boundaries[-1] = support_max
    for i in range(1, n_levels):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids, boundaries


def compute_distortion(sigma, centroids, boundaries):
    """Compute MSE distortion of a Gaussian quantizer."""
    pdf = lambda x: norm.pdf(x, 0, sigma)
    mse = 0.0
    for i in range(len(centroids)):
        lo, hi = boundaries[i], boundaries[i + 1]
        val, _ = integrate.quad(
            lambda x, c=centroids[i]: (x - c) ** 2 * pdf(x), lo, hi, limit=100
        )
        mse += val
    return mse


def generate_codebooks(output_dir):
    """Generate codebooks for N(0,1). Scale by 1/sqrt(d) at runtime."""
    os.makedirs(output_dir, exist_ok=True)

    bits_list = [1, 2, 3, 4]
    results = {}

    for bits in bits_list:
        n_levels = 2**bits
        print(f"Computing Lloyd-Max for N(0,1): bits={bits}, levels={n_levels}")

        centroids, boundaries = lloyd_max_gaussian(n_levels, sigma=1.0)
        distortion = compute_distortion(1.0, centroids, boundaries)

        print(f"  Centroids: {centroids}")
        print(f"  Boundaries: {boundaries}")
        print(f"  MSE distortion: {distortion:.6e}")

        results[f"bits{bits}_centroids"] = centroids.astype(np.float32)
        results[f"bits{bits}_boundaries"] = boundaries.astype(np.float32)

    output_path = os.path.join(output_dir, "codebooks.npz")
    np.savez_compressed(output_path, **results)
    print(f"\nSaved to {output_path}")
    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "mlx_turboquant", "data")
    generate_codebooks(data_dir)
