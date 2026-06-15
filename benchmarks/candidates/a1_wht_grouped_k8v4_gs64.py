"""Candidate A1: WHT + grouped symmetric quantization, keys 8-bit, values 5-bit, group_size=64.

This is the first compression candidate and the foundational baseline for all
subsequent candidates in the promotion ladder.

Algorithm
---------
Keys:
  1. Apply Walsh-Hadamard Transform over head_dim (makes distribution more
     Gaussian → better suited for symmetric quantization).
  2. Apply mx.quantize at 8 bits, group_size=64 (bit-packed, MLX-native).
  3. On fetch: mx.dequantize → inverse WHT.

Values:
  1. Apply WHT (WHT distributes energy more uniformly across coordinates).
  2. Apply mx.quantize at 5 bits, group_size=64 (bit-packed).
  3. On fetch: mx.dequantize → inverse WHT.

Design notes
------------
- WHT is applied via KeyQuant._apply_wht_pretransform (self-inverse).
- Sign preconditioning is NOT used in the cache because it is
  position-dependent: applying signs to incremental batches would require
  tracking global token offsets, which complicates the interface.
  WHT alone provides significant preconditioning benefit.
- mlx.quantize/dequantize provides bit-packed storage (real memory savings).
- Both K and V buffers are pre-allocated with step=256 like mlx_lm's
  QuantizedKVCache to amortise allocation cost.

Memory model (per layer, per token, head_dim=D, k=8, v=4, gs=64)
-----------------------------------------------------------------
Keys:   D*8/32 * 4 + D/64 * 4 + D/64 * 4  = D + D/8  bytes
         (packed codes)   (scales)  (biases)
Values: D*4/32 * 4 + D/64 * 4 + D/64 * 4  = 0.5*D + D/8 bytes
K+V total: D + D/8 + 0.5*D + D/8  = 1.75*D bytes
FP16 K+V : 2 * D * 2           = 4*D bytes
Ratio    : 1.75 / 4           = 0.4375 → 56.25% reduction (passes 30% gate)

For D=128: 224 bytes vs 512 FP16 bytes → compression_factor ≈ 2.29×

NOTE: This candidate uses MLX's mx.quantize which only supports bits 2,3,4,6,8.
The production runtime uses GroupedCartesianQuantizer which supports arbitrary bit widths including 5-bit.
This A1 benchmark therefore tests V4 due to MLX limitations, while the production runtime uses V5.

Candidate name: A1_wht_grouped_k8v4_gs64
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.candidates.base_candidate import BenchmarkCandidate
from benchmarks.schemas import CandidateResult

# Lazy import guard
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx.utils import tree_map
    _MLX_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# A1 KV Cache
# ---------------------------------------------------------------------------

class A1_WHT_GroupedKVCache:
    """WHT + grouped symmetric KV cache for one transformer layer.

    Implements the update_and_fetch(keys, values) interface expected by
    mlx_lm.utils.generate_step(prompt_cache=...).
    """

    step = 256

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 8,
        value_bits: int = 4,
        group_size: int = 64,
    ) -> None:
        if not _MLX_AVAILABLE:
            raise ImportError("mlx is required for A1_WHT_GroupedKVCache")

        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.offset = 0

        # Pre-allocated compressed buffers (None until first call)
        self.keys: tuple | None = None    # (w_q, scales, biases)
        self.values: tuple | None = None  # (w_q, scales, biases)

        # Timing accumulators (ms)
        self.compression_time_ms: float = 0.0
        self.decompression_time_ms: float = 0.0

        # Lazy-loaded WHT from KeyQuant
        self._key_quant = None

    def _get_key_quant(self) -> Any:
        if self._key_quant is None:
            from rfsn_v11.quant.key_quant import KeyQuant
            # WHT only — no sign preconditioning (see module docstring)
            self._key_quant = KeyQuant(
                bits=self.key_bits,
                group_size=self.group_size,
                use_wht=True,
                use_incoherent_signs=False,
            )
        return self._key_quant

    def _apply_wht(self, x: mx.array) -> mx.array:
        """Apply WHT to last dimension.  Self-inverse."""
        kq = self._get_key_quant()
        return kq._apply_wht_pretransform(x)

    def _ensure_capacity(
        self,
        B: int,
        H: int,
        new_steps: int,
        k_dim: int,
        v_dim: int,
        dtype: mx.Dtype,
    ) -> None:
        prev = self.offset
        if self.keys is not None and (prev + new_steps) <= self.keys[0].shape[-2]:
            return

        capacity = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (B, H, capacity)

        def _packed_dim(bits: int, dim: int) -> int:
            return dim * bits // 32

        def _group_dim(dim: int) -> int:
            return dim // self.group_size

        def _init_buf(bits: int, dim: int) -> tuple:
            packed = mx.zeros((*shape, _packed_dim(bits, dim)), dtype=mx.uint32)
            scales = mx.zeros((*shape, _group_dim(dim)), dtype=dtype)
            biases = mx.zeros((*shape, _group_dim(dim)), dtype=dtype)
            return (packed, scales, biases)

        def _expand_buf(buf: tuple) -> tuple:
            new_capacity = shape[-1]
            expanded = tuple(
                mx.concatenate(
                    [t, mx.zeros((*t.shape[:-1], new_capacity), dtype=t.dtype)],
                    axis=-1,
                )
                for t in buf
            )
            return expanded

        if self.keys is not None:
            if prev % self.step != 0:
                # Trim buffers to actual used size before expanding
                self.keys = tree_map(lambda t: t[..., :prev, :], self.keys)
                self.values = tree_map(lambda t: t[..., :prev, :], self.values)
            self.keys = _expand_buf(self.keys)
            self.values = _expand_buf(self.values)
        else:
            self.keys = _init_buf(self.key_bits, k_dim)
            self.values = _init_buf(self.value_bits, v_dim)

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compress new keys/values, store, and return full decompressed history.

        Parameters
        ----------
        keys, values : (B, H, T, D) float arrays

        Returns
        -------
        (decompressed_keys, decompressed_values) : (B, H, offset, D) float
        """
        B, H, T, D = keys.shape
        prev = self.offset
        self._ensure_capacity(B, H, T, D, values.shape[-1], keys.dtype)

        # --- Compress new tokens ---
        t0 = time.perf_counter()

        k_wht = self._apply_wht(keys)  # (B, H, T, D)
        v_wht = self._apply_wht(values)

        k_q, k_s, k_b = mx.quantize(
            k_wht.reshape(B * H * T, D), group_size=self.group_size, bits=self.key_bits
        )
        v_q, v_s, v_b = mx.quantize(
            v_wht.reshape(B * H * T, D), group_size=self.group_size, bits=self.value_bits
        )

        # Reshape back to (B, H, T, packed_dim)
        packed_k = _packed_dim(self.key_bits, D)
        packed_v = _packed_dim(self.value_bits, D)
        groups = D // self.group_size

        k_q = k_q.reshape(B, H, T, packed_k)
        k_s = k_s.reshape(B, H, T, groups)
        k_b = k_b.reshape(B, H, T, groups)
        v_q = v_q.reshape(B, H, T, packed_v)
        v_s = v_s.reshape(B, H, T, groups)
        v_b = v_b.reshape(B, H, T, groups)

        # Write into pre-allocated buffers
        self.keys[0][..., prev : prev + T, :] = k_q
        self.keys[1][..., prev : prev + T, :] = k_s
        self.keys[2][..., prev : prev + T, :] = k_b
        self.values[0][..., prev : prev + T, :] = v_q
        self.values[1][..., prev : prev + T, :] = v_s
        self.values[2][..., prev : prev + T, :] = v_b

        self.offset += T
        total_T = self.offset

        mx.eval(self.keys, self.values)
        self.compression_time_ms += (time.perf_counter() - t0) * 1000.0

        # --- Decompress full history ---
        t0 = time.perf_counter()

        k_buf_q = self.keys[0][..., :total_T, :]   # (B, H, total_T, packed_k)
        k_buf_s = self.keys[1][..., :total_T, :]
        k_buf_b = self.keys[2][..., :total_T, :]
        v_buf_q = self.values[0][..., :total_T, :]
        v_buf_s = self.values[1][..., :total_T, :]
        v_buf_b = self.values[2][..., :total_T, :]

        groups = D // self.group_size

        k_deq = mx.dequantize(
            k_buf_q.reshape(B * H * total_T, _packed_dim(self.key_bits, D)),
            k_buf_s.reshape(B * H * total_T, groups),
            k_buf_b.reshape(B * H * total_T, groups),
            group_size=self.group_size, bits=self.key_bits,
        ).reshape(B, H, total_T, D)

        v_deq = mx.dequantize(
            v_buf_q.reshape(B * H * total_T, _packed_dim(self.value_bits, D)),
            v_buf_s.reshape(B * H * total_T, groups),
            v_buf_b.reshape(B * H * total_T, groups),
            group_size=self.group_size, bits=self.value_bits,
        ).reshape(B, H, total_T, D)

        # Inverse WHT (self-inverse)
        k_out = self._apply_wht(k_deq)
        v_out = self._apply_wht(v_deq)

        mx.eval(k_out, v_out)
        self.decompression_time_ms += (time.perf_counter() - t0) * 1000.0

        return k_out, v_out

    def compressed_bytes(self) -> int:
        """Return current compressed buffer size in bytes."""
        if self.keys is None:
            return 0
        total_T = self.offset
        D = self.head_dim
        B_H = self.keys[0].shape[0] * self.keys[0].shape[1]
        k_bytes = B_H * total_T * (
            _packed_dim(self.key_bits, D) * 4  # uint32
            + D // self.group_size * 4          # scales float32
            + D // self.group_size * 4          # biases float32
        )
        v_bytes = B_H * total_T * (
            _packed_dim(self.value_bits, D) * 4
            + D // self.group_size * 4
            + D // self.group_size * 4
        )
        return k_bytes + v_bytes

    # ------------------------------------------------------------------
    # mlx_lm compat interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> tuple:
        if self.offset == (self.keys[0].shape[2] if self.keys else 0):
            return self.keys, self.values
        return (
            tree_map(lambda t: t[..., :self.offset, :], self.keys),
            tree_map(lambda t: t[..., :self.offset, :], self.values),
        )

    @state.setter
    def state(self, v: tuple) -> None:
        self.keys, self.values = v

    @property
    def meta_state(self) -> tuple:
        return (str(self.step), str(self.offset), str(self.group_size),
                str(self.key_bits), str(self.value_bits))

    @meta_state.setter
    def meta_state(self, v: tuple) -> None:
        self.step, self.offset, self.group_size, self.key_bits, self.value_bits = (
            int(x) for x in v
        )

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


def _packed_dim(bits: int, dim: int) -> int:
    return dim * bits // 32


# ---------------------------------------------------------------------------
# Candidate wrapper
# ---------------------------------------------------------------------------

class A1_WHT_Grouped(BenchmarkCandidate):
    """A1: WHT grouped symmetric quantization — keys 8-bit, values 5-bit, gs64.

    This is the first rung on the compression ladder.  It must:
    - Reduce KV memory by >= 30 %
    - Pass logit_cosine >= 0.995, top5_overlap >= 0.95
    - Pass attention_score_cosine >= 0.995
    """

    candidate_name = "A1_wht_grouped_k8v4_gs64"

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 4,
        group_size: int = 64,
    ) -> None:
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size

    def is_available(self) -> bool:
        if not _MLX_AVAILABLE:
            return False
        try:
            import mlx_lm  # noqa: F401

            from rfsn_v11.quant.key_quant import KeyQuant  # noqa: F401
            return True
        except ImportError:
            return False

    def run_on_model(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prompt_id: str,
        prompt: str,
        output_tokens: int = 100,
        seed: int = 42,
    ) -> CandidateResult:
        if not _MLX_AVAILABLE:
            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                error="mlx not available",
            )
        try:
            return self._run(model, tokenizer, model_id, prompt_id, prompt, output_tokens, seed)
        except Exception as exc:
            return CandidateResult(
                candidate_name=self.candidate_name,
                model_id=model_id,
                prompt_id=prompt_id,
                error=str(exc),
            )

    def _run(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prompt_id: str,
        prompt: str,
        output_tokens: int,
        seed: int,
    ) -> CandidateResult:
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import generate_step

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.0)

        # Detect model architecture
        head_dim = self._detect_head_dim(model)
        n_layers = self._detect_n_layers(model)

        # Build per-layer compressed KV caches
        caches = [
            A1_WHT_GroupedKVCache(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                group_size=self.group_size,
            )
            for _ in range(n_layers)
        ]

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="mlx")
        context_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(tokenizer.encode(prompt))
        if not hasattr(input_ids, "shape"):
            input_ids = mx.array(tokenizer.encode(prompt))[None]

        # --- Run generation ---
        t_start = time.perf_counter()
        first_token_time: float | None = None
        generated_tokens: list[int] = []
        all_logprobs: list[np.ndarray] = []

        prefill_tps: float = 0.0
        decode_tps: float = 0.0
        peak_memory_mb: float = 0.0

        for token, logprobs_mx in generate_step(
            input_ids[0],
            model,
            max_tokens=output_tokens,
            sampler=sampler,
            prompt_cache=caches,
        ):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            tok = int(token.item())
            generated_tokens.append(tok)
            try:
                lp = np.array(logprobs_mx).flatten().astype(np.float32)
                all_logprobs.append(lp)
            except Exception:
                pass
            if tok == tokenizer.eos_token_id:
                break

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        first_token_latency_ms = (
            (first_token_time - t_start) * 1000.0
            if first_token_time is not None else None
        )

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        n_gen = len(generated_tokens)
        if n_gen > 0 and first_token_latency_ms is not None:
            decode_time_s = (t_end - first_token_time) / max(n_gen - 1, 1)
            decode_tps = 1.0 / max(decode_time_s, 1e-9)
            prefill_time_s = first_token_latency_ms / 1000.0
            prefill_tps = context_length / max(prefill_time_s, 1e-9)

        # --- Peak memory from MLX ---
        try:
            peak_memory_mb = mx.metal.get_peak_memory() / (1024 ** 2)
        except Exception:
            peak_memory_mb = 0.0

        # --- Compression metrics from caches ---
        total_compressed_bytes = sum(c.compressed_bytes() for c in caches)
        compressed_kv_mb = total_compressed_bytes / (1024 ** 2)
        kv_cache_mb = self.estimate_kv_memory_mb(model, context_length + n_gen)

        total_comp_ms = sum(c.compression_time_ms for c in caches)
        total_decomp_ms = sum(c.decompression_time_ms for c in caches)

        compression_factor = (kv_cache_mb / max(compressed_kv_mb, 1e-9)) if compressed_kv_mb > 0 else None
        effective_bits = (
            (self.key_bits + self.value_bits) / 2.0
            * (1 + 2 * 4 / (self.group_size * self.key_bits / 8 * 2))  # approximate scale overhead
        )

        return CandidateResult(
            candidate_name=self.candidate_name,
            model_id=model_id,
            prompt_id=prompt_id,
            context_length=context_length,
            output_tokens=n_gen,
            preconditioner="wht",
            quantizer="grouped_sym",
            key_bits=float(self.key_bits),
            value_bits=float(self.value_bits),
            group_size=self.group_size,
            # Quality metrics: filled by run_a1.py comparison against baseline
            # Runtime
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            first_token_latency_ms=first_token_latency_ms,
            total_latency_ms=total_latency_ms,
            compression_time_ms=total_comp_ms,
            decompression_time_ms=total_decomp_ms,
            # Memory
            peak_memory_mb=peak_memory_mb,
            kv_cache_memory_mb=kv_cache_mb,
            compressed_kv_memory_mb=compressed_kv_mb,
            metadata_memory_mb=0.0,
            compression_factor=compression_factor,
            effective_bits_per_kv_element=effective_bits,
            generated_text=generated_text,
            notes=f"WHT + grouped sym k{self.key_bits}/v{self.value_bits} gs{self.group_size}",
        )

    @staticmethod
    def _detect_head_dim(model: Any) -> int:
        args = getattr(model, "args", None)
        if args:
            hd = getattr(args, "head_dim", None)
            if hd:
                return int(hd)
            hidden = getattr(args, "hidden_size", None) or getattr(args, "dim", 0)
            n_heads = getattr(args, "num_attention_heads", None) or getattr(args, "n_heads", 1)
            if hidden and n_heads:
                return hidden // n_heads
        return 128

    @staticmethod
    def _detect_n_layers(model: Any) -> int:
        args = getattr(model, "args", None)
        if args:
            n = getattr(args, "num_hidden_layers", None) or getattr(args, "n_layers", None)
            if n:
                return int(n)
        layers = getattr(model, "layers", [])
        return len(layers) if layers else 24
