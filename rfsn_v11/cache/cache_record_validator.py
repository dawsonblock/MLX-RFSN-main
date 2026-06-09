"""Cache-record validator for rfsn_v11.

Ported from vmlx-main/vmlx_engine/cache_record_validator.py.

Hard guard against malformed paged/L2 cache restores.  Validates cache
records BEFORE allocating MLX tensors so a corrupted on-disk block never
cascades into a multi-hundred-GB Metal buffer allocation.

Validator is intentionally cheap (operates on numpy/MLX shapes; no MLX kernel
launches) and idempotent — safe to call from both disk-read and
reconstruct-cache paths.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- Hard ceilings -----------------------------------------------------------
# Per-tensor cap: 4 GB. The largest legitimate single tensor in any rfsn_v11
# cache record tops out around 1-2 GB. 4 GB is a generous safety margin.
MAX_TENSOR_BYTES = 4 * 1024 ** 3

# Per-record cap: 16 GB. Even a 43-layer model at full quantization stays under 8 GB.
MAX_TOTAL_RECORD_BYTES = 16 * 1024 ** 3

# Per-dimension sanity cap.
MAX_TENSOR_DIM = 65536 * 4  # 256K defensive 4x

# Maximum tensor rank (ndim). KV caches are at most rank-4.
MAX_TENSOR_NDIM = 6

# Metadata/scalar ceilings.
MAX_CACHE_OFFSET = 2_000_000
MAX_CACHE_LAYERS = 1024
MAX_CACHE_GROUP_SIZE = 4096
ALLOWED_CACHE_BITS = {2, 3, 4, 8}


class CacheValidationError(ValueError):
    """Raised when a cache record is unsafe to restore."""


# =============================================================================
# Low-level helpers
# =============================================================================


def _tensor_byte_size(t: Any) -> int:
    """Compute byte size of an MLX/numpy tensor without triggering eval."""
    if t is None:
        return 0
    nbytes = getattr(t, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    shape = getattr(t, "shape", None)
    itemsize = getattr(t, "itemsize", None)
    if shape is None or itemsize is None:
        return 0
    n = 1
    for d in shape:
        n *= max(int(d), 0)
    return n * int(itemsize)


def _validate_tensor(
    t: Any,
    *,
    label: str,
    layer_idx: int,
) -> Tuple[bool, int, str]:
    """Validate a single tensor's shape and byte size.

    Returns:
        (ok, bytes, reason)
    """
    if t is None:
        return True, 0, ""
    shape = getattr(t, "shape", None)
    if shape is None:
        return True, 0, ""
    if len(shape) > MAX_TENSOR_NDIM:
        return False, 0, (
            f"layer {layer_idx} {label}: ndim={len(shape)} > {MAX_TENSOR_NDIM}"
        )
    for axis, dim in enumerate(shape):
        d = int(dim)
        if d < 0:
            return False, 0, f"layer {layer_idx} {label}: dim[{axis}]={d} < 0"
        if d > MAX_TENSOR_DIM:
            return False, 0, (
                f"layer {layer_idx} {label}: dim[{axis}]={d} > {MAX_TENSOR_DIM}"
            )
    nbytes = _tensor_byte_size(t)
    if nbytes > MAX_TENSOR_BYTES:
        return False, nbytes, (
            f"layer {layer_idx} {label}: {nbytes} bytes > {MAX_TENSOR_BYTES} cap"
        )
    return True, nbytes, ""


def _walk_tensors(obj: Any) -> Iterable[Any]:
    """Yield tensor-like objects nested inside dict/list/tuple state trees."""
    if obj is None:
        return
    if hasattr(obj, "shape"):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_tensors(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_tensors(v)


def _safe_int(value: Any, *, label: str) -> Tuple[bool, int, str]:
    try:
        if isinstance(value, bool):
            return False, 0, f"{label}: bool is not a valid integer"
        return True, int(value), ""
    except (TypeError, ValueError, OverflowError) as e:
        return False, 0, f"{label}: invalid integer {value!r}: {e}"


def _validate_int_range(
    value: Any,
    *,
    label: str,
    lo: int,
    hi: int,
) -> Tuple[bool, int, str]:
    ok, parsed, reason = _safe_int(value, label=label)
    if not ok:
        return False, 0, reason
    if parsed < lo or parsed > hi:
        return False, parsed, f"{label}: {parsed} outside [{lo}, {hi}]"
    return True, parsed, ""


def _decode_meta_sequence(meta: Any) -> list:
    if meta is None:
        return []
    if isinstance(meta, (list, tuple)):
        return list(meta)
    if isinstance(meta, dict):
        return []
    if isinstance(meta, str):
        stripped = meta.strip()
        if not stripped or stripped == "{}":
            return []
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, (list, tuple)):
                return list(decoded)
            return []
        except json.JSONDecodeError:
            return [p for p in stripped.replace(",", " ").split() if p]
    return []


def _validate_shape_list(
    shape: Any,
    *,
    label: str,
    bytes_per_elem: int = 2,
) -> Tuple[bool, str, int]:
    """Validate a JSON-decoded shape list and compute byte size.

    Returns:
        (ok, reason, nbytes)
    """
    if not isinstance(shape, list):
        return False, f"{label}: shape is not a list", 0
    if len(shape) > MAX_TENSOR_NDIM:
        return False, f"{label}: ndim={len(shape)} > {MAX_TENSOR_NDIM}", 0
    n = 1
    for axis, dim in enumerate(shape):
        if not isinstance(dim, int) or dim < 0:
            return False, f"{label}: dim[{axis}]={dim!r} invalid", 0
        if dim > MAX_TENSOR_DIM:
            return False, f"{label}: dim[{axis}]={dim} > {MAX_TENSOR_DIM}", 0
        n *= dim
    nbytes = n * bytes_per_elem
    if nbytes > MAX_TENSOR_BYTES:
        return False, f"{label}: {nbytes} bytes > {MAX_TENSOR_BYTES} cap", nbytes
    return True, "", nbytes


# =============================================================================
# Public validate_cache_record
# =============================================================================


def validate_cache_record(
    cache_data: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str]:
    """Validate a cache record's tensors before reconstruction.

    Walks the tagged-tuple block_data list produced by
    BlockAwarePrefixCache._extract_block_tensor_slice() and checks every
    tensor against the size caps.

    Args:
        cache_data: List of tagged tuples, one per layer.
        expected_num_layers: If given, enforces exact layer count.
        source: Label for diagnostic messages.

    Returns:
        ``(ok, reason)``
    """
    if cache_data is None:
        return True, ""  # Empty block is acceptable
    if not isinstance(cache_data, (list, tuple)):
        return False, f"{source}: cache_data is {type(cache_data).__name__}"

    if expected_num_layers is not None and len(cache_data) != expected_num_layers:
        return False, (
            f"{source}: layer count {len(cache_data)} != expected "
            f"{expected_num_layers}"
        )

    total_bytes = 0

    for layer_idx, entry in enumerate(cache_data):
        if not isinstance(entry, (tuple, list)) or not entry:
            continue

        tag = entry[0]

        if tag in ("kv", "rotating_kv"):
            for t_idx, t in enumerate(entry[1:3]):
                ok, nbytes, reason = _validate_tensor(
                    t, label=f"tag={tag}[{t_idx}]", layer_idx=layer_idx
                )
                if not ok:
                    return False, f"{source}: {reason}"
                total_bytes += nbytes

        elif tag == "quantized_kv":
            for bucket_idx, bucket in enumerate(entry[1:3]):
                if not isinstance(bucket, (list, tuple)):
                    continue
                for t_idx, t in enumerate(bucket):
                    ok, nbytes, reason = _validate_tensor(
                        t,
                        label=f"tag=quantized_kv[{bucket_idx}][{t_idx}]",
                        layer_idx=layer_idx,
                    )
                    if not ok:
                        return False, f"{source}: {reason}"
                    total_bytes += nbytes

        elif tag == "cumulative":
            state = entry[1] if len(entry) > 1 else None
            for t in _walk_tensors(state):
                ok, nbytes, reason = _validate_tensor(
                    t, label="tag=cumulative.state", layer_idx=layer_idx
                )
                if not ok:
                    return False, f"{source}: {reason}"
                total_bytes += nbytes

        elif tag == "cache_list" and len(entry) > 1:
            for sub_idx, sub in enumerate(entry[1] or []):
                if not isinstance(sub, (tuple, list)) or not sub:
                    continue
                for t in _walk_tensors(sub[1:]):
                    ok, nbytes, reason = _validate_tensor(
                        t,
                        label=f"tag=cache_list.sub[{sub_idx}]",
                        layer_idx=layer_idx,
                    )
                    if not ok:
                        return False, f"{source}: {reason}"
                    total_bytes += nbytes

        if total_bytes > MAX_TOTAL_RECORD_BYTES:
            return False, (
                f"{source}: accumulated {total_bytes} bytes > "
                f"{MAX_TOTAL_RECORD_BYTES} cap at layer {layer_idx}"
            )

    return True, ""


def reject_or_warn(
    cache_data: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> bool:
    """Validate cache_data and log a warning on failure.

    Returns:
        True if safe to use, False if record must be treated as a cache miss.
    """
    ok, reason = validate_cache_record(
        cache_data,
        expected_num_layers=expected_num_layers,
        source=source,
    )
    if not ok:
        logger.warning(
            "Cache record validation rejected %s: %s — treating as cache miss.",
            source,
            reason,
        )
    return ok


# =============================================================================
# validate_tq_native_metadata (used by tq_disk_store.deserialize_tq_cache)
# =============================================================================


def validate_tq_native_metadata(
    tensors: dict,
    metadata: dict,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str]:
    """Validate TurboQuant native disk metadata before any decode allocation.

    Guards against poisoned metadata (e.g. ``__tq_i_ck_shape__`` declaring a
    decoded KV tensor of hundreds of GB) that would pass safetensors header
    validation but blow up decode_keys().

    Args:
        tensors: Dict of named tensors from ``mx.load()``.
        metadata: String metadata dict from safetensors header.
        expected_num_layers: If given, enforces exact layer count.
        source: Label for diagnostic messages.

    Returns:
        ``(ok, reason)``
    """
    if not isinstance(metadata, dict):
        return False, f"TQ metadata is {type(metadata).__name__}, expected dict"

    ok, num_layers, reason = _validate_int_range(
        metadata.get("__num_layers__", "0"),
        label="__num_layers__",
        lo=0,
        hi=MAX_CACHE_LAYERS,
    )
    if not ok:
        return False, reason

    if expected_num_layers is not None and num_layers != expected_num_layers:
        return False, (
            f"__num_layers__ {num_layers} != expected {expected_num_layers} "
            f"(source={source})"
        )

    total_decoded = 0

    def _validate_tq_prefix(prefix: str, label: str) -> Tuple[bool, str]:
        nonlocal total_decoded
        required = (
            f"{prefix}_ck_indices_packed",
            f"{prefix}_ck_qjl_packed",
            f"{prefix}_ck_residual_norms",
            f"{prefix}_ck_vector_norms",
            f"{prefix}_cv_indices_packed",
            f"{prefix}_cv_vector_norms",
        )
        missing = [name for name in required if name not in tensors]
        if missing:
            return False, f"{label}: missing compressed tensors {missing}"

        for suffix in ("ck_shape", "cv_shape"):
            raw = metadata.get(f"__{prefix}_{suffix}__", "[]")
            try:
                shape = json.loads(raw)
            except json.JSONDecodeError as e:
                return False, f"{label}.{suffix}: invalid JSON {raw!r}: {e}"
            ok_shape, shape_reason, nbytes = _validate_shape_list(
                shape, label=f"{label}.{suffix}", bytes_per_elem=2
            )
            if not ok_shape:
                return False, shape_reason
            total_decoded += nbytes
            if total_decoded > MAX_TOTAL_RECORD_BYTES:
                return False, (
                    f"{label}: decoded total {total_decoded} bytes "
                    f"> {MAX_TOTAL_RECORD_BYTES}"
                )

        for suffix in ("ck_bits", "cv_bits", "key_bits", "value_bits"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_bits, bits, bit_reason = _validate_int_range(
                    metadata[key], label=f"{label}.{suffix}", lo=1, hi=16
                )
                if not ok_bits:
                    return False, bit_reason
                if bits not in ALLOWED_CACHE_BITS:
                    return False, (
                        f"{label}.{suffix}: {bits} not in {sorted(ALLOWED_CACHE_BITS)}"
                    )

        for suffix in ("offset", "compressed_tokens", "sink_tokens"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_int, _, int_reason = _validate_int_range(
                    metadata[key],
                    label=f"{label}.{suffix}",
                    lo=0,
                    hi=MAX_CACHE_OFFSET,
                )
                if not ok_int:
                    return False, int_reason

        for suffix in ("key_dim", "value_dim"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_dim, _, dim_reason = _validate_int_range(
                    metadata[key],
                    label=f"{label}.{suffix}",
                    lo=1,
                    hi=MAX_TENSOR_DIM,
                )
                if not ok_dim:
                    return False, dim_reason

        return True, ""

    for i in range(num_layers):
        cls_name = metadata.get(f"__layer_{i}_class__", "")
        if cls_name == "TurboQuantKVCache":
            ok_prefix, prefix_reason = _validate_tq_prefix(f"tq_{i}", f"layer {i}")
            if not ok_prefix:
                return False, prefix_reason

        if metadata.get(f"__layer_{i}_cache_list__") == "true":
            ok_count, sub_count, count_reason = _validate_int_range(
                metadata.get(f"__layer_{i}_cl_count__", "0"),
                label=f"layer {i}.cache_list_count",
                lo=0,
                hi=64,
            )
            if not ok_count:
                return False, count_reason
            for j in range(sub_count):
                sub_cls = metadata.get(f"__layer_{i}_cl_{j}_class__", "")
                if sub_cls == "TurboQuantKVCache":
                    ok_prefix, prefix_reason = _validate_tq_prefix(
                        f"cl_{i}_{j}", f"layer {i}.cache_list[{j}]"
                    )
                    if not ok_prefix:
                        return False, prefix_reason

        if metadata.get(f"__layer_{i}_quantized__") == "true":
            for suffix in ("qk_count", "qv_count"):
                key = f"__layer_{i}_{suffix}__"
                if key in metadata:
                    ok_count, _, count_reason = _validate_int_range(
                        metadata[key],
                        label=f"layer {i}.{suffix}",
                        lo=0,
                        hi=8,
                    )
                    if not ok_count:
                        return False, count_reason

        if metadata.get(f"__layer_{i}_cumulative__") == "true":
            key = f"__layer_{i}_state_count__"
            if key in metadata:
                ok_count, _, count_reason = _validate_int_range(
                    metadata[key],
                    label=f"layer {i}.state_count",
                    lo=0,
                    hi=64,
                )
                if not ok_count:
                    return False, count_reason

        meta_key = f"__layer_{i}_meta__"
        if meta_key in metadata:
            seq = _decode_meta_sequence(metadata[meta_key])
            if seq:
                ok_offset, _, offset_reason = _validate_int_range(
                    seq[0],
                    label=f"layer {i}.meta[0]",
                    lo=0,
                    hi=MAX_CACHE_OFFSET,
                )
                if not ok_offset:
                    return False, offset_reason

    return True, ""
