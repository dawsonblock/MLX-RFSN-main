# Real Generation Throughput Report

**Model:** Qwen/Qwen2.5-0.5B-Instruct  
**Seed:** 42  
**Configs:** baseline_fp16, k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive, experimental_hybrid

## Teacher-Forced Logit Equivalence

### baseline_fp16 @ 128 tokens (identity guard)

- **Positions checked:** 128
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000
- **Max abs logit delta:** 0.0000
- **Mean abs logit delta:** 0.0000

### baseline_fp16 @ 512 tokens (identity guard)

- **Positions checked:** 128
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000
- **Max abs logit delta:** 0.0000
- **Mean abs logit delta:** 0.0000

### baseline_fp16 @ 1024 tokens (identity guard)

- **Positions checked:** 128
- **Logit cosine vs FP16:** 1.0000
- **Top-5 overlap vs FP16:** 1.0000
- **KL vs FP16:** 0.000000
- **Max abs logit delta:** 0.0000
- **Mean abs logit delta:** 0.0000

### k8_v5_gs64 @ 128 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9556
- **Top-5 overlap vs FP16:** 0.7688
- **KL vs FP16:** 0.123498
- **Max abs logit delta:** 11.4453
- **Mean abs logit delta:** 0.8855

### k8_v5_gs64 @ 512 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9634
- **Top-5 overlap vs FP16:** 0.7328
- **KL vs FP16:** 0.079445
- **Max abs logit delta:** 15.0312
- **Mean abs logit delta:** 0.6330

### k8_v5_gs64 @ 1024 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9483
- **Top-5 overlap vs FP16:** 0.7406
- **KL vs FP16:** 0.034789
- **Max abs logit delta:** 13.0781
- **Mean abs logit delta:** 0.8210

### k8_v5_gs32 @ 128 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9555
- **Top-5 overlap vs FP16:** 0.7656
- **KL vs FP16:** 0.125121
- **Max abs logit delta:** 11.5781
- **Mean abs logit delta:** 0.8807

### k8_v5_gs32 @ 512 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9635
- **Top-5 overlap vs FP16:** 0.7313
- **KL vs FP16:** 0.079418
- **Max abs logit delta:** 14.9922
- **Mean abs logit delta:** 0.6306

### k8_v5_gs32 @ 1024 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9487
- **Top-5 overlap vs FP16:** 0.7313
- **KL vs FP16:** 0.034757
- **Max abs logit delta:** 13.0078
- **Mean abs logit delta:** 0.8163

### turbo_polar @ 128 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9006
- **Top-5 overlap vs FP16:** 0.5781
- **KL vs FP16:** 0.096738
- **Max abs logit delta:** 11.6094
- **Mean abs logit delta:** 1.1456

### turbo_polar @ 512 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.8638
- **Top-5 overlap vs FP16:** 0.5750
- **KL vs FP16:** 0.068500
- **Max abs logit delta:** 13.1406
- **Mean abs logit delta:** 1.5711

### turbo_polar @ 1024 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.8921
- **Top-5 overlap vs FP16:** 0.6000
- **KL vs FP16:** 0.038018
- **Max abs logit delta:** 13.8203
- **Mean abs logit delta:** 1.3028

### adaptive @ 128 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.8901
- **Top-5 overlap vs FP16:** 0.5859
- **KL vs FP16:** 0.126297
- **Max abs logit delta:** 11.3828
- **Mean abs logit delta:** 1.2032

### adaptive @ 512 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.8532
- **Top-5 overlap vs FP16:** 0.5891
- **KL vs FP16:** 0.057872
- **Max abs logit delta:** 13.8281
- **Mean abs logit delta:** 1.5967

### adaptive @ 1024 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.8930
- **Top-5 overlap vs FP16:** 0.6016
- **KL vs FP16:** 0.038081
- **Max abs logit delta:** 13.8125
- **Mean abs logit delta:** 1.2974

### experimental_hybrid @ 128 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9248
- **Top-5 overlap vs FP16:** 0.6625
- **KL vs FP16:** 0.130485
- **Max abs logit delta:** 11.0625
- **Mean abs logit delta:** 1.0596

### experimental_hybrid @ 512 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9490
- **Top-5 overlap vs FP16:** 0.7172
- **KL vs FP16:** 0.037331
- **Max abs logit delta:** 10.5000
- **Mean abs logit delta:** 1.0010

### experimental_hybrid @ 1024 tokens

- **Positions checked:** 128
- **Logit cosine vs FP16:** 0.9497
- **Top-5 overlap vs FP16:** 0.6813
- **KL vs FP16:** 0.020494
- **Max abs logit delta:** 11.1562
- **Mean abs logit delta:** 0.8829

## Free-Running Generation Divergence

### baseline_fp16 @ 128 tokens (identity)

- **Tokens/sec:** 54.89
- **Total E2E ms:** 2332.08
- **Compression ratio:** 1.00x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 1.0000

### baseline_fp16 @ 512 tokens (identity)

- **Tokens/sec:** 46.53
- **Total E2E ms:** 2750.77
- **Compression ratio:** 1.00x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 1.0000

### baseline_fp16 @ 1024 tokens (identity)

- **Tokens/sec:** 13.10
- **Total E2E ms:** 9773.14
- **Compression ratio:** 1.00x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 1.0000

### k8_v5_gs64 @ 128 tokens

- **Tokens/sec:** 41.43
- **Total E2E ms:** 3089.31
- **Compression ratio:** 2.23x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9572

### k8_v5_gs64 @ 512 tokens

- **Tokens/sec:** 42.31
- **Total E2E ms:** 3025.63
- **Compression ratio:** 2.23x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9616

### k8_v5_gs64 @ 1024 tokens

- **Tokens/sec:** 17.87
- **Total E2E ms:** 7163.71
- **Compression ratio:** 2.23x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9488

### k8_v5_gs32 @ 128 tokens

- **Tokens/sec:** 55.87
- **Total E2E ms:** 2290.98
- **Compression ratio:** 2.09x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9574

### k8_v5_gs32 @ 512 tokens

- **Tokens/sec:** 44.85
- **Total E2E ms:** 2853.93
- **Compression ratio:** 2.09x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9621

### k8_v5_gs32 @ 1024 tokens

- **Tokens/sec:** 23.62
- **Total E2E ms:** 5418.06
- **Compression ratio:** 2.09x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.9483

### turbo_polar @ 128 tokens

- **Tokens/sec:** 56.48
- **Total E2E ms:** 2266.13
- **Compression ratio:** 2.31x
- **First divergence position:** 9
- **Exact token match rate:** 14.06%
- **Logit cosine vs FP16:** 0.6537

### turbo_polar @ 512 tokens

- **Tokens/sec:** 48.38
- **Total E2E ms:** 2645.57
- **Compression ratio:** 2.32x
- **First divergence position:** 58
- **Exact token match rate:** 45.31%
- **Logit cosine vs FP16:** 0.6196

### turbo_polar @ 1024 tokens

- **Tokens/sec:** 24.66
- **Total E2E ms:** 5189.64
- **Compression ratio:** 2.32x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.8006

### adaptive @ 128 tokens

- **Tokens/sec:** 55.53
- **Total E2E ms:** 2305.21
- **Compression ratio:** 2.31x
- **First divergence position:** 9
- **Exact token match rate:** 14.06%
- **Logit cosine vs FP16:** 0.6500

### adaptive @ 512 tokens

- **Tokens/sec:** 49.82
- **Total E2E ms:** 2569.10
- **Compression ratio:** 2.32x
- **First divergence position:** 28
- **Exact token match rate:** 25.00%
- **Logit cosine vs FP16:** 0.5093

### adaptive @ 1024 tokens

- **Tokens/sec:** 24.28
- **Total E2E ms:** 5270.91
- **Compression ratio:** 2.32x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.8005

### experimental_hybrid @ 128 tokens

- **Tokens/sec:** 54.07
- **Total E2E ms:** 2367.32
- **Compression ratio:** 2.95x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.8532

### experimental_hybrid @ 512 tokens

- **Tokens/sec:** 50.24
- **Total E2E ms:** 2547.53
- **Compression ratio:** 2.97x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.8631

### experimental_hybrid @ 1024 tokens

- **Tokens/sec:** 23.20
- **Total E2E ms:** 5517.96
- **Compression ratio:** 2.98x
- **First divergence position:** None
- **Exact token match rate:** 100.00%
- **Logit cosine vs FP16:** 0.8903

