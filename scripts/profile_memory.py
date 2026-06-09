#!/usr/bin/env python3
"""
RFSN v10 - Memory Profiling Utility.
Tracks MLX/Metal memory usage across store/retrieve operations.
"""
from __future__ import annotations

import json
import platform
import tempfile
from datetime import datetime, timezone

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.memory_guard import MemoryGuard


def get_active_memory_bytes() -> int:
    """Safely query MLX active memory."""
    try:
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
            return int(mx.metal.get_active_memory())
        if hasattr(mx, 'get_active_memory'):
            return int(mx.get_active_memory())
    except Exception:
        pass
    return 0


def profile_shapes():
    shapes = [
        (1, 8, 128, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 32, 4096, 128),
    ]

    print(f"Hardware: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    results = []

    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=8, v_bits=3, use_incoherent=True,
            max_memory_gb=1.0, max_pinned_memory_gb=0.5, cache_dir=td,
        )

        guard = MemoryGuard(
            safety_margin_gb=0.5,
            soft_limit_gb=4.0,
            hard_limit_gb=6.0,
        )

        for shape in shapes:
            mx.random.seed(42)
            mem_before = get_active_memory_bytes()

            k = mx.random.normal(shape)
            v = mx.random.normal(shape)
            mx.eval(k, v)
            mem_after_tensor = get_active_memory_bytes()

            mgr.store("profile", k, v, shape[2])
            cache = mgr.active_caches["profile"]
            estimated = mgr._estimate_cache_bytes(cache)

            result = mgr.retrieve("profile", out_dtype=mx.float32)
            k_rec, v_rec = result
            mx.eval(k_rec, v_rec)
            mem_after_retrieve = get_active_memory_bytes()

            results.append({
                "shape": str(shape),
                "mem_before_bytes": mem_before,
                "mem_after_tensor_bytes": mem_after_tensor,
                "estimated_cache_bytes": estimated,
                "mem_after_retrieve_bytes": mem_after_retrieve,
                "guard_status": guard.get_status(),
            })

            print(f"  {shape}: "
                  f"tensor={mem_after_tensor - mem_before:,} bytes, "
                  f"cache_est={estimated:,} bytes, "
                  f"after_retrieve={mem_after_retrieve:,} bytes")

            # Clear for next iteration
            mgr.active_caches.clear()

    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    profile_shapes()
