# Baseline Zone

This zone documents the stable, proven components that newer candidates are compared against.

## Components

### `rfsn_v10`

- **k8_v5_gs32**: Keys 8-bit, values 5-bit, group size 32. Validated baseline.
- **k8_v5_gs64**: Keys 8-bit, values 5-bit, group size 64. Validated baseline.

These configs have passed regression tests and are used as the **BASELINE** status in the shootout.

### `mlx_lm_baseline`

Plain MLX-LM generation with no KV compression. This is the **CONTROL**.

### `mlx_lm_quantized_kv`

MLX-LM's built-in quantized KV cache (if available in your installed version). This is also a **CONTROL**.

## Decision Rule

If a new candidate cannot beat at least one baseline on a useful axis (memory, speed, or quality), it does not promote.
