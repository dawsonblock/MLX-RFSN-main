"""RFSN v11 cache version.

Bump CACHE_VERSION and add a comment describing the format change whenever
the on-disk cache format changes. Loading a cache with mismatched version
must hard-fail, not silently reconstruct incorrect tensors.
"""
# v1: Initial rfsn_v11 format.
#   Keys: WHT + grouped symmetric (k_bits=8, group_size=64)
#   Values: PolarQuant rotation + Lloyd-Max (v_bits=4, dim=128)
#   Metadata: model_id, k_bits, v_bits, use_wht, cache_version
CACHE_VERSION = 1
