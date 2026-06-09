# M5 MLX Wheel Platform Performance

Date: 2026-05-13
Scope: vMLX Python/Electron bundled runtime on M5 Max, especially Qwen3.6 affine JANG bundles.

## Finding

The same `mlx==0.31.2` package version can ship materially different Metal runtime artifacts depending on the macOS wheel tag. On this M5 Max/Tahoe host:

- `macosx_14_0_arm64` bundle: Qwen3.6-27B `JANG_4M` prefill measured about 256-288 prompt tokens/s.
- `macosx_26_0_arm64` bundle: the same model, same `mlx-lm==0.31.3`, same direct `generate_step()` harness measured about 870-972 prompt tokens/s.

The difference is not the Electron UI, vMLX scheduler, `prefillBatchSize`, or the model files. Direct `mlx_lm.generate_step()` reproduced the slow path with the older wheel and the fast path with the native wheel.

## Evidence

Exact model:

```text
/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK
```

Slow app bundle before the fix:

```text
mlx Tag: cp312-cp312-macosx_14_0_arm64
mlx-metal Tag: py3-none-macosx_14_0_arm64
3005 prompt tokens: 288.0 PP tok/s
12005 prompt tokens: 256.2 PP tok/s
```

Fast rebuilt bundle after the fix:

```text
mlx Tag: cp312-cp312-macosx_26_0_arm64
mlx-metal Tag: py3-none-macosx_26_0_arm64
3005 prompt tokens: 971.8 PP tok/s
12005 prompt tokens: 870.5 PP tok/s
```

## Build Rule

`panel/scripts/bundle-python.sh` now defaults `VMLINUX_BUNDLE_MLX_PLATFORM=auto`:

- macOS 26+ build hosts use `macosx_26_0_arm64` to keep M5/Tahoe Metal kernels.
- older build hosts use `macosx_14_0_arm64`.
- compatibility releases can force the older wheel with:

```bash
VMLINUX_BUNDLE_MLX_PLATFORM=compat bash panel/scripts/bundle-python.sh
```

Do not reintroduce a hard-coded `macosx_14_0_arm64` default without accepting the M5 prompt-processing regression.
