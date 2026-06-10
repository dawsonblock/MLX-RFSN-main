"""External memory layer for MLX-RFSN.

This module provides memory integrations (Qdrant, TurboVec) that are
separate from KV-cache compression. Do not put TurboVec into KV cache.

Architecture:
    files/docs/code
        ↓
    chunking
        ↓
    embeddings
        ↓
    Qdrant or TurboVec
        ↓
    retrieved context
        ↓
    MLX-LM/vMLX model
"""
