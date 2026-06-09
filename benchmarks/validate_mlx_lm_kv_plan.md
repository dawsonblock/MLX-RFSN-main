# MLX-LM Real Model Validation Plan (Main14/Main15)

## Purpose
Define an MLX-native real-model validation path that compares baseline logits
against compressed-KV logits without PyTorch/CPU roundtrips.

## Scope
- load model with mlx-lm
- run baseline decode with native MLX cache path
- run compressed KV path through RFSN manager
- compare logits/quality on Apple Silicon directly

## Proposed Flow
1. Load tokenizer + model through mlx-lm APIs.
2. Encode prompt and run context prefill.
3. Capture baseline next-token logits and selected cache state.
4. Compress/decompress KV with RFSN route (with and without sparse proxy mode).
5. Decode held-out tokens using reconstructed KV.
6. Measure and store comparison metrics.

## Required Metrics
- logit_cosine
- logit_max_abs_diff
- top1_token_match_rate
- top5_overlap
- perplexity_delta
- tokens_tested

## Artifact Contract
Write:
- artifacts/proof/main12/real_model_validation.json when run
or
- artifacts/proof/main12/real_model_validation_not_run.txt when skipped

## Acceptance Gates
- no fallback-in-strict route for claimed Metal path
- explicit run/not-run status in proof summary
- no quality overclaim when only synthetic tensor audits are available

## Notes
Main13 keeps the current optional real-model scaffold honest and does not claim
full MLX-native validation completion.
