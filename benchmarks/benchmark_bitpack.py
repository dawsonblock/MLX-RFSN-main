#!/usr/bin/env python3
"""
RFSN v10 - Bitpack Benchmarks.
Measures pack/unpack throughput, effective GB/s, and compression ratio.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import mlx.core as mx

from rfsn_v10.bitpack import BitPackedQuantizer


def get_metadata() -> dict:
    return {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_pack(bits: int, n: int, iterations: int = 10) -> dict:
    mx.random.seed(42)
    max_val = (1 << bits) - 1
    x = mx.random.randint(0, max_val + 1, (n,), dtype=mx.uint32)
    mx.eval(x)

    # Warmup
    packed, nv = BitPackedQuantizer.pack(x, bits)
    mx.eval(packed)

    timings = []
    for _ in range(iterations):
        t0 = time.monotonic()
        packed, nv = BitPackedQuantizer.pack(x, bits)
        mx.eval(packed)
        t1 = time.monotonic()
        timings.append(t1 - t0)

    packed_bytes = int(packed.size) * 4
    original_bytes = n * 4
    throughput_codes_per_s = n / (sum(timings) / len(timings))

    return {
        "bits": bits,
        "n_codes": n,
        "pack_latency_ms": (sum(timings) / len(timings)) * 1000.0,
        "throughput_codes_per_s": throughput_codes_per_s,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "compression_ratio": packed_bytes / original_bytes if original_bytes > 0 else 0,
    }


def benchmark_unpack(bits: int, n: int, iterations: int = 10) -> dict:
    mx.random.seed(42)
    max_val = (1 << bits) - 1
    x = mx.random.randint(0, max_val + 1, (n,), dtype=mx.uint32)
    packed, nv = BitPackedQuantizer.pack(x, bits)
    mx.eval(packed)

    timings = []
    for _ in range(iterations):
        t0 = time.monotonic()
        unpacked = BitPackedQuantizer.unpack(packed, nv, bits)
        mx.eval(unpacked)
        t1 = time.monotonic()
        timings.append(t1 - t0)

    throughput_codes_per_s = n / (sum(timings) / len(timings))

    return {
        "bits": bits,
        "n_codes": n,
        "unpack_latency_ms": (sum(timings) / len(timings)) * 1000.0,
        "throughput_codes_per_s": throughput_codes_per_s,
    }


def main():
    print("=" * 60)
    print("RFSN v10 Bitpack Benchmarks")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    bit_widths = [3, 8]

    results = {"metadata": meta, "pack": [], "unpack": []}

    for bits in bit_widths:
        print(f"\n--- bits={bits} ---")
        for n in sizes:
            p = benchmark_pack(bits, n)
            print(f"  pack n={n:>10}: {p['pack_latency_ms']:.3f} ms, "
                  f"ratio={p['compression_ratio']:.3f}, "
                  f"{p['throughput_codes_per_s']/1e6:.1f}M codes/s")
            results["pack"].append(p)

            u = benchmark_unpack(bits, n)
            print(f"  unpack n={n:>10}: {u['unpack_latency_ms']:.3f} ms, "
                  f"{u['throughput_codes_per_s']/1e6:.1f}M codes/s")
            results["unpack"].append(u)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
