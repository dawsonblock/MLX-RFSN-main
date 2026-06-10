"""Document chunking utilities for external memory.

Chunks files/docs/code into pieces suitable for embedding and retrieval.
"""
from __future__ import annotations

from collections.abc import Iterator


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> Iterator[str]:
    """Split text into overlapping chunks.

    Parameters
    ----------
    chunk_size
        Target character length per chunk.
    overlap
        Number of characters to overlap between chunks.

    Yields
    ------
    str
        Text chunk.
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start = end - overlap
        if start >= end:
            break
