# RFSN v10 Benchmark Results

**Platform**: Apple Silicon / MLX
**Date**: 2026-06-08

## KV Cache Quality (8-bit Grouped Quantization)

| Shape | K Bits | V Bits | Size ratio | K Cosine | V Cosine | Gate |
|-------|--------|--------|-------------|----------|----------|------|
| (1, 8, 2048, 64) | 8 | 8 | 0.265 | 0.99998 | 0.99998 | PASS |
| (1, 32, 4096, 128) | 8 | 8 | 0.266 | 0.99998 | 0.99998 | PASS |

## Quality Gates

- **Cosine similarity threshold**: 0.999
- **Key cosine actual**: 0.99998
- **Value cosine actual**: 0.99998
- **Gate result**: PASS

## Notes

- KV cache 8-bit grouped quantization preserves cosine similarity > 0.99998
- Compression factor ~3.75x (size_ratio 0.266) for typical transformer shapes
- All tested shapes pass the 0.999 cosine similarity threshold