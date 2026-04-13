"""CacheStore: read/write cache entries via VectorIndex — M2 implementation."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from gateway.cache.index import VectorIndex
from gateway.models import CacheEntry

logger = logging.getLogger(__name__)


class CacheStore:
    """Async facade for semantic-cache read/write operations.

    Uses VectorIndex (pgvector or in-memory numpy) for similarity search and storage.
    """

    def __init__(
        self,
        index: VectorIndex | None = None,
        redis_client: Any | None = None,  # kept for compat, not used in M2
        pool: Any | None = None,
    ) -> None:
        self._index = index or VectorIndex(pool=pool)
        self._redis = redis_client  # reserved for future Redis layer

    async def init(self) -> None:
        """Initialise the underlying index (runs DDL if pgvector)."""
        await self._index.init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def read(
        self,
        embedding: list[float],
        threshold: float,
        domain: str = "unknown",
    ) -> CacheEntry | None:
        """Return the best cache hit (cosine sim ≥ threshold) or None.

        Expired entries are silently ignored (TTL eviction).
        On hit, increments hit_count asynchronously (fire-and-forget).
        """
        results = await self._index.search(embedding, threshold, limit=1)
        if not results:
            return None
        entry = results[0]

        # Double-check TTL (index search already filters, but be defensive)
        now = datetime.now(tz=timezone.utc)
        exp = entry.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < now:
            return None

        # Fire-and-forget hit_count increment
        asyncio.create_task(self._index.increment_hit_count(entry.id))
        return entry

    async def write(self, entry: CacheEntry) -> None:
        """Persist a new cache entry (upsert)."""
        await self._index.upsert(entry)

    # ------------------------------------------------------------------
    # Legacy API (M1 compat)
    # ------------------------------------------------------------------

    async def get(self, entry_id: str) -> CacheEntry | None:
        """Fetch by exact ID (not by similarity)."""
        if self._index._use_mem and self._index._mem:
            e = self._index._mem.get_by_id(entry_id)
            if e is None:
                return None
            now = datetime.now(tz=timezone.utc)
            exp = e.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return e if exp >= now else None
        return None

    async def put(self, entry: CacheEntry) -> None:
        """Alias for write."""
        await self.write(entry)

    async def flush(self, tenant_id: str | None = None) -> int:
        """Flush all cache entries. Returns count flushed."""
        if self._index._use_mem and self._index._mem:
            return self._index._mem.clear()
        return 0
