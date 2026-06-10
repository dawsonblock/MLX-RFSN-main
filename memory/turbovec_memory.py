"""TurboVec-based compressed local memory.

Use TurboVec as a compressed local memory experiment.
Do not make TurboVec required.
"""
from __future__ import annotations

from typing import Any


class TurboVecMemory:
    """TurboVec external memory client wrapper.

    This is a scaffold. TurboVec integration is experimental.
    """

    def __init__(self, path: str = "./turbovec_store") -> None:
        self.path = path
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import turbovec  # type: ignore[import]
                self._client = turbovec.Client(self.path)
            except ImportError:
                raise ImportError(
                    "turbovec is not installed. "
                    "TurboVec memory is experimental and optional."
                )
        return self._client

    def add(self, texts: list[str], embeddings: list[list[float]]) -> None:
        self._get_client()  # raises if turbovec is not installed
        raise NotImplementedError("TurboVec add() not yet implemented")

    def search(
        self, embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        self._get_client()  # raises if turbovec not installed
        raise NotImplementedError(
            "TurboVec search() not yet implemented"
        )
