# Polar Fused Baseline Benchmarks

| Candidate | Context | Prefill TPS | Decode TPS | KV Memory (MB) |
|-----------|---------|-------------|------------|----------------|
| mlx_fp16_baseline | 128 | 132.2 | 93.3 | 3.00 |
| rfsn_v10_k8_v5_gs64 | 128 | 0.0 | 10.4 | 0.00 |
| polar_naive_dequantized | 128 | 0.0 | 0.0 | 0.00 |
| mlx_fp16_baseline | 512 | 2862.8 | 84.3 | 6.00 |
| rfsn_v10_k8_v5_gs64 | 512 | 0.0 | 3.8 | 0.00 |
| polar_naive_dequantized | 512 | 0.0 | 0.0 | 0.00 |
| mlx_fp16_baseline | 1024 | 380.3 | 76.2 | 9.00 |
| rfsn_v10_k8_v5_gs64 | 1024 | 0.0 | 2.5 | 0.00 |
| polar_naive_dequantized | 1024 | 0.0 | 0.0 | 0.00 |
| mlx_fp16_baseline | 2048 | 1576.9 | 69.0 | 15.00 |
| rfsn_v10_k8_v5_gs64 | 2048 | 0.0 | 1.2 | 0.00 |
| polar_naive_dequantized | 2048 | 0.0 | 0.0 | 0.00 |