# Qwen2.5-1.5B-Instruct Experimental Comparison Summary

- **baseline_fp16**: cos_min=1.000000 top5=1.0000 kl=0.00e+00 ratio=1.00 status=reference
- **k8_v5_gs64**: cos_min=0.999385 top5=0.9875 kl=1.53e-06 ratio=2.23 status=candidate
- **k8_v5_gs32**: cos_min=0.999588 top5=0.9900 kl=1.30e-06 ratio=2.08 status=candidate
- **adaptive**: cos_min=0.999632 top5=0.9863 kl=1.32e-06 ratio=2.27 status=candidate
- **experimental_hybrid**: cos_min=0.999607 top5=0.9850 kl=1.27e-06 ratio=2.27 status=candidate