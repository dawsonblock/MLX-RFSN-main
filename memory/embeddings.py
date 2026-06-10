"""Embedding utilities for external memory.

Provides a simple wrapper around sentence-transformers for generating
text embeddings. Falls back gracefully if the library is not installed.
"""
from __future__ import annotations

from typing import Any


class Embedder:
    """Simple sentence-transformers wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install -e '.[memory]'"
                )
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts)
        return [emb.tolist() for emb in embeddings]
