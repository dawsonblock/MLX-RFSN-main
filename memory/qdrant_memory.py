"""Qdrant-based external memory.

Use Qdrant first for reliability. It is the stable vector memory option.
"""
from __future__ import annotations

from typing import Any


class QdrantMemory:
    """Qdrant external memory client wrapper.

    This is a scaffold. Install with:
        pip install -e ".[memory]"
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "rfsn_memory") -> None:
        self.url = url
        self.collection = collection
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.url)
            except ImportError:
                raise ImportError(
                    "qdrant-client is not installed. "
                    "Install with: pip install -e '.[memory]'"
                )
        return self._client

    def ensure_collection(self, vector_size: int = 384) -> None:
        client = self._get_client()
        try:
            client.get_collection(self.collection)
        except Exception:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add(self, texts: list[str], embeddings: list[list[float]]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=i, vector=emb, payload={"text": txt})
            for i, (emb, txt) in enumerate(zip(embeddings, texts))
        ]
        client.upsert(collection_name=self.collection, points=points)

    def search(self, embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        client = self._get_client()
        results = client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=limit,
        )
        return [
            {"text": r.payload.get("text", ""), "score": r.score}
            for r in results
        ]
