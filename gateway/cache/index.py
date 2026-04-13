"""pgvector ANN index wrapper (M2 stub)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorIndex:
    """Async interface to pgvector similarity index.

    Full implementation in M2.
    """

    def __init__(self, pool: Any | None = None) -> None:
        self._pool = pool

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.92,
    ) -> list[tuple[str, float]]:
        """Return list of (entry_id, similarity) for nearest neighbours.

        M1 stub: always returns empty list.
        """
        return []

    async def insert(self, entry_id: str, embedding: list[float]) -> None:
        """Insert a new embedding into the index (M1 stub: no-op)."""
        pass
