"""rfsn_v11 cache sub-package.

Public re-exports for the cache infrastructure:

  version              -- CACHE_VERSION constant
  cache_key            -- compute_block_hash, make_cache_key
  paged_cache          -- CacheBlock, KVCacheBlock, FreeKVCacheBlockQueue,
                          BlockHashToBlockMap, BlockTable, PagedCacheManager
  prefix_cache         -- BlockAwarePrefixCache
  tq_disk_store        -- serialize_tq_cache, deserialize_tq_cache,
                          is_tq_compressed_cache
  cache_record_validator -- validate_cache_record, validate_tq_native_metadata,
                            reject_or_warn, CacheValidationError
"""

from .version import CACHE_VERSION
from .cache_key import compute_block_hash as compute_block_hash_key, make_cache_key
from .paged_cache import (
    CacheBlock,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
    BlockHashToBlockMap,
    BlockTable,
    PagedCacheManager,
)
from .prefix_cache import BlockAwarePrefixCache
from .tq_disk_store import (
    is_tq_compressed_cache,
    serialize_tq_cache,
    deserialize_tq_cache,
)
from .cache_record_validator import (
    CacheValidationError,
    validate_cache_record,
    validate_tq_native_metadata,
    reject_or_warn,
)

__all__ = [
    # version
    "CACHE_VERSION",
    # cache_key
    "compute_block_hash_key",
    "make_cache_key",
    # paged_cache
    "CacheBlock",
    "KVCacheBlock",
    "FreeKVCacheBlockQueue",
    "BlockHashToBlockMap",
    "BlockTable",
    "PagedCacheManager",
    # prefix_cache
    "BlockAwarePrefixCache",
    # tq_disk_store
    "is_tq_compressed_cache",
    "serialize_tq_cache",
    "deserialize_tq_cache",
    # cache_record_validator
    "CacheValidationError",
    "validate_cache_record",
    "validate_tq_native_metadata",
    "reject_or_warn",
]
