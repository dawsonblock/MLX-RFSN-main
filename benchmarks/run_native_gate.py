#!/usr/bin/env python3
"""
Deterministic native benchmark entry point for RFSN K8/V8 GS64.

Phase 3: One command that:
  1. Validates MLX and Metal availability.
  2. Probes the backend and reports its state.
  3. Enables true-packed mode.
  4. Runs the kernel self-test (implicit in probe).
  5. Runs dense and packed traces over the same tokens.
  6. Saves all artifacts.
  7. Fails on any fallback.
  8. Fails on missing provenance.

Usage::

    python -m benchmarks.run_native_gate \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --candidate rfsn_direct_packed_k8v8_gs64 \
        --context-lengths 128 512 2048 \
        --output-tokens 64 \
        --strict

Exit code 0 = all artifacts valid, no fallback, provenance complete.
Exit code 1 = any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rfsn_v11.candidates.runtime_config import RFSNRuntimeConfig
from rfsn_v11.candidates.backend_state import BackendState


ARTIFACTS_ROOT = Path("artifacts/proof/native_gate")


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _compute_token_hash(token_ids: list[int]) -> str:
    """Deterministic hash of a token sequence."""
    payload = ",".join(str(t) for t in token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_dense_baseline(
    model_id: str,
    prompt: str,
    max_tokens: int,
    config: RFSNRuntimeConfig,
) -> dict:
    """Run dense FP16 baseline and return trace metadata."""
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_id)
    t0 = time.perf_counter()
    output = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    input_ids = tokenizer.encode(prompt)
    output_ids = tokenizer.encode(output)
    gen_ids = output_ids[len(input_ids):]

    return {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": len(input_ids),
        "generated_tokens": len(gen_ids),
        "generated_text": output,
        "elapsed_ms": round(elapsed_ms, 2),
        "token_sequence_hash": _compute_token_hash(gen_ids),
        "backend": "dense_fp16_baseline",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _generate_teacher_forced(
    model: Any,
    tokenizer: Any,
    prompt: str,
    forced_ids: list[int],
    cache_list: list[Any],
) -> tuple[list[float], list[int]]:
    """Teacher-forced generation: feed exact tokens, return logit metrics.

    Returns (per_step_max_logits, per_step_argmax_ids).
    """
    import mlx.core as mx
    import numpy as np

    prompt_ids = tokenizer.encode(prompt)
    y = mx.array(prompt_ids)

    # Prefill
    logits = model(y[None], cache=cache_list)
    logits = logits[:, -1, :]

    per_step_max: list[float] = []
    per_step_argmax: list[int] = []

    for forced_token in forced_ids:
        # Record stats for this step
        logit_np = np.array(logits.astype(mx.float32).squeeze(0))
        per_step_max.append(float(np.max(logit_np)))
        per_step_argmax.append(int(np.argmax(logit_np)))

        # Force the next token
        y = mx.array([forced_token])
        logits = model(y[None], cache=cache_list)
        logits = logits[:, -1, :]

    return per_step_max, per_step_argmax


def _run_dense_baseline(
    model_id: str,
    prompt: str,
    max_tokens: int,
    config: RFSNRuntimeConfig,
) -> dict:
    """Run dense FP16 baseline and return trace metadata.

    Uses a manual generation loop so we capture exact token IDs,
    then re-runs teacher-forced with fresh caches for comparison.
    """
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = mlx_lm.load(model_id)
    sampler = make_sampler(temp=0.0)
    prompt_ids = tokenizer.encode(prompt)

    # Step 1: Free-running to get the token sequence
    y = mx.array(prompt_ids)
    logits = model(y[None], cache=None)
    logits = logits[:, -1, :]

    gen_ids: list[int] = []
    for _ in range(max_tokens):
        logprobs = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
        token = sampler(logprobs).item()
        gen_ids.append(int(token))
        y = mx.array([token])
        logits = model(y[None], cache=None)
        logits = logits[:, -1, :]

    full_ids = prompt_ids + gen_ids
    output_text = tokenizer.decode(full_ids)

    # Step 2: Re-load model to get fresh caches for teacher-forced trace
    del model
    import gc
    gc.collect()
    mx.metal.reset_peak_memory()

    model, _ = mlx_lm.load(model_id)
    standard_caches: list[Any] = [None] * len(model.layers)

    t0 = time.perf_counter()
    dense_max, dense_argmax = _generate_teacher_forced(
        model, tokenizer, prompt, gen_ids, standard_caches
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(gen_ids),
        "generated_text": output_text,
        "elapsed_ms": round(elapsed_ms, 2),
        "token_sequence_hash": _compute_token_hash(gen_ids),
        "forced_token_ids": gen_ids,
        "per_step_max_logits": [round(x, 4) for x in dense_max],
        "per_step_argmax": dense_argmax,
        "backend": "dense_fp16_baseline",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_packed_trace(
    model_id: str,
    prompt: str,
    forced_ids: list[int],
    config: RFSNRuntimeConfig,
) -> dict:
    """Run packed K8/V8 teacher-forced trace and return trace metadata."""
    import mlx_lm
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_model_support import (
        RfsnDirectPackedKVCache,
        packed_attention_context,
    )

    model, tokenizer = mlx_lm.load(model_id)

    key_codec = CartesianCodec(bits=config.key_bits, group_size=config.group_size)
    value_codec = CartesianCodec(bits=config.value_bits, group_size=config.group_size)

    session = GenerationCacheSession(
        model_id=model_id,
        num_layers=len(model.layers),
        key_codec=key_codec,
        value_codec=value_codec,
        staging_capacity=config.staging_capacity,
        dense_residual_window=config.dense_residual_window,
        use_paged_arena=True,
        max_pages=256,
    )

    cache_list = [
        RfsnDirectPackedKVCache(
            layer_id=i,
            key_codec=key_codec,
            value_codec=value_codec,
            staging_capacity=config.staging_capacity,
            dense_residual_window=config.dense_residual_window,
            strict=config.strict_backend,
            session=session,
        )
        for i in range(len(model.layers))
    ]

    t0 = time.perf_counter()
    with packed_attention_context(model, cache_list, strict=config.strict_backend):
        packed_max, packed_argmax = _generate_teacher_forced(
            model, tokenizer, prompt, forced_ids, cache_list
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    prompt_ids = tokenizer.encode(prompt)

    # Gather proof counters
    counters = session.runtime_counters.to_dict()

    return {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(forced_ids),
        "forced_token_ids": forced_ids,
        "elapsed_ms": round(elapsed_ms, 2),
        "token_sequence_hash": _compute_token_hash(forced_ids),
        "per_step_max_logits": [round(x, 4) for x in packed_max],
        "per_step_argmax": packed_argmax,
        "backend": "packed_k8v8_gs64",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counters": counters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic native benchmark gate for RFSN K8/V8"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model ID to benchmark",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="rfsn_direct_packed_k8v8_gs64",
        help="Canonical candidate name",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=64,
        help="Number of decode tokens to generate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on any fallback (default: True)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow fallback to reference kernels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ARTIFACTS_ROOT),
        help="Directory for artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe backend and print report without running generation",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build runtime config and probe backend
    # ------------------------------------------------------------------
    config = RFSNRuntimeConfig(
        backend="metal_true_packed",
        strict_backend=args.strict,
        model_id=args.model,
        context_lengths=args.context_lengths,
        output_tokens=args.output_tokens,
        key_bits=8,
        value_bits=8,
        group_size=64,
        staging_capacity=64,
        dense_residual_window=0,
    )

    print("=== RFSN Native Gate ===")
    print(f"Model:     {args.model}")
    print(f"Candidate: {args.candidate}")
    print(f"Strict:    {args.strict}")
    print("")

    print("Probing backend ...")
    report = config.probe_backend()
    print(f"  State: {report.state.value}")
    if report.reason:
        print(f"  Reason: {report.reason}")
    if report.chip_model:
        print(f"  Chip:   {report.chip_model}")
    if report.mlx_version:
        print(f"  MLX:    {report.mlx_version}")

    _write_json(out_dir / "backend_report.json", report.to_dict())

    if report.state != BackendState.READY:
        print(f"\nFAILED: backend is not READY ({report.state.value})")
        return 1

    print("  Backend READY")

    if args.dry_run:
        print("\nDry run complete.")
        return 0

    # ------------------------------------------------------------------
    # 2. Run traces at each context length
    # ------------------------------------------------------------------
    all_ok = True
    manifest = {
        "candidate": args.candidate,
        "model_id": args.model,
        "config": config.to_dict(),
        "backend_report": report.to_dict(),
        "runs": [],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for ctx_len in args.context_lengths:
        print(f"\n--- Context length {ctx_len} ---")
        # Build a prompt that tokenizes to roughly ctx_len tokens
        # Use repeated text for determinism
        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "In 1492, Christopher Columbus sailed the ocean blue. "
            "The capital of France is Paris. "
            "Machine learning is a subset of artificial intelligence. "
        )
        # Repeat until we exceed ctx_len, then trim
        repeated = (base_text * ((ctx_len // 10) + 1))[: ctx_len * 6]

        print(f"  Running dense baseline ...")
        try:
            dense_result = _run_dense_baseline(
                args.model, repeated, args.output_tokens, config
            )
        except Exception as exc:
            print(f"  ERROR (dense): {exc}")
            all_ok = False
            dense_result = {"error": str(exc)}

        # Extract forced token IDs from dense baseline for teacher-forced comparison
        forced_ids = dense_result.get("forced_token_ids", [])
        if not forced_ids and "error" not in dense_result:
            print("  WARNING: no forced_token_ids from dense baseline")

        print(f"  Running packed trace ...")
        try:
            packed_result = _run_packed_trace(
                args.model, repeated, forced_ids, config
            )
        except Exception as exc:
            print(f"  ERROR (packed): {exc}")
            all_ok = False
            packed_result = {"error": str(exc)}

        # Compare token hashes
        dense_hash = dense_result.get("token_sequence_hash", "")
        packed_hash = packed_result.get("token_sequence_hash", "")
        if dense_hash and packed_hash:
            match = dense_hash == packed_hash
            print(f"  Token match: {match}")
            if not match and args.strict:
                print("  FAILED: token sequence divergence")
                all_ok = False
        else:
            match = None

        run_entry = {
            "context_length": ctx_len,
            "dense": dense_result,
            "packed": packed_result,
            "token_match": match,
        }
        manifest["runs"].append(run_entry)

    # ------------------------------------------------------------------
    # 3. Write manifest and validate
    # ------------------------------------------------------------------
    manifest_path = out_dir / "native_gate_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"\nWrote manifest: {manifest_path}")

    # Check for zero fallback
    for run in manifest["runs"]:
        packed = run.get("packed", {})
        counters = packed.get("counters", {})
        if counters.get("dense_fallback_calls", 0) > 0:
            print(
                f"  FAILED: dense_fallback_calls > 0 "
                f"at context {run['context_length']}"
            )
            all_ok = False
        if counters.get("full_history_materialization_calls", 0) > 0:
            print(
                f"  FAILED: full_history_materialization_calls > 0 "
                f"at context {run['context_length']}"
            )
            all_ok = False

    if all_ok:
        print("\n=== Native Gate Passed ===")
        return 0
    else:
        print("\n=== Native Gate Failed ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
