"""RFSN v11 cache key utilities.

Provides stable, config-stamped cache keys for block-level prefix caching.
Each key encodes the block content hash chained from its parent, plus all
loader-config dimensions (model_id, k_bits, v_bits, use_wht, CACHE_VERSION)
so records from incompatible configurations never collide.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .version import CACHE_VERSION


def compute_block_hash(token_ids: list[int], parent_hash: str) -> str:
    """Compute a SHA-256 chain hash for a block of token IDs.

    Each block's hash is derived from both its own token content AND the
    hash of the preceding block, forming an unforgeable chain.  Two blocks
    with identical tokens but different parent hashes produce different
    digests, preventing cross-context cache collisions.

    Args:
        token_ids: Ordered list of integer token IDs in this block.
        parent_hash: Hex-string hash of the preceding block, or the literal
            string ``"root"`` for the first block in a sequence.

    Returns:
        Lowercase hex-string SHA-256 digest of the chained hash.
    """
    # Inner hash: content of this block.
    inner = hashlib.sha256(bytes(token_ids)).digest()

    # Outer hash: chain inner with parent string for positional binding.
    outer = hashlib.sha256(inner + parent_hash.encode()).hexdigest()
    return outer


def make_cache_key(
    block_hash: str,
    model_id: str,
    k_bits: int,
    v_bits: float,
    use_wht: bool,
) -> str:
    """Construct a fully config-stamped cache key string.

    The key encodes every dimension that affects the on-disk representation:
    block content hash, model identity, quantization parameters, WHT flag, and
    the current ``CACHE_VERSION``.  A cache record is valid only when ALL
    components match exactly; any mismatch is a hard miss.

    Args:
        block_hash: Hex string from :func:`compute_block_hash`.
        model_id: Stable model identifier (e.g. ``"llama3"``).
        k_bits: Bit-width used for key quantization (e.g. ``8``).
        v_bits: Bit-width used for value quantization (e.g. ``4.0``).
        use_wht: Whether Walsh-Hadamard Transform was applied to keys.

    Returns:
        Colon-separated cache key string including ``CACHE_VERSION``.
    """
    return f"{block_hash}:{model_id}:{k_bits}:{v_bits}:{use_wht}:{CACHE_VERSION}"
