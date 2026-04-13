"""Cache entry storage (M2 stub)."""
from __future__ import annotations

import logging
from typing import Any

from gateway.models import CacheEntry

logger = logging.getLogger(__name__)


class CacheStore:
    """Async interface for reading/writing CacheEntry objects.

    Full implementation in M2 (Redis + Postgres backends).
    """

    def __init__(self, redis_client: Any | None = None, pool: Any | None = None) -> None:
        self._redis = redis_client
        self._pool = pool

    async def get(self, entry_id: str) -> CacheEntry | None:
        """Fetch a cache entry by ID (M1 stub: always returns None)."""
        return None

    async def put(self, entry: CacheEntry) -> None:
        """Store a cache entry (M1 stub: no-op)."""
        pass

    async def flush(self, tenant_id: str | None = None) -> int:
        """Flush all (or tenant-specific) cache entries.

        Returns number of entries flushed.
        """
        return 0
