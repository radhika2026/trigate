"""Prefix → backend locality map (M3, Redis-backed)."""
from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis

from gateway.config import get_settings
from gateway.scheduler.hasher import hash_pages

logger = logging.getLogger(__name__)

_POOL: aioredis.ConnectionPool | None = None


def _get_pool() -> aioredis.ConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = aioredis.ConnectionPool.from_url(
            get_settings().redis_url,
            max_connections=50,
            decode_responses=True,
        )
    return _POOL


class LocalityMap:
    """Redis-backed map: prefix_hash → backend_id.

    All I/O is async and uses a shared connection pool.
    TTL-based expiry is handled by Redis natively.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        if redis_url is not None:
            self._pool: aioredis.ConnectionPool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=50,
                decode_responses=True,
            )
        else:
            self._pool = _get_pool()

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.Redis(connection_pool=self._pool)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def set(self, prefix_hash: str, backend_id: str, ttl_s: int = 3600) -> None:
        """Store prefix_hash → backend_id with the given TTL."""
        try:
            async with self._client() as client:
                await client.setex(f"kv:{prefix_hash}", ttl_s, backend_id)
        except Exception as exc:
            logger.warning("LocalityMap.set failed: %s", exc)

    async def get(self, prefix_hash: str) -> str | None:
        """Return backend_id for this hash, or None if missing/expired."""
        try:
            async with self._client() as client:
                value: Any = await client.get(f"kv:{prefix_hash}")
            return value  # None when missing
        except Exception as exc:
            logger.warning("LocalityMap.get failed: %s", exc)
            return None

    async def find_backend(self, page_hashes: list[str]) -> str | None:
        """Return backend_id for the longest-matching prefix hash.

        Iterates through page_hashes in order; last match wins (= longest prefix).
        Returns None if no hash is found in Redis.
        """
        if not page_hashes:
            return None
        result: str | None = None
        try:
            async with self._client() as client:
                for h in page_hashes:
                    val: Any = await client.get(f"kv:{h}")
                    if val is not None:
                        result = val  # keep updating → last/longest match wins
        except Exception as exc:
            logger.warning("LocalityMap.find_backend failed: %s", exc)
            return None
        return result

    async def record_request(
        self,
        token_ids: list[int],
        backend_id: str,
        ttl_s: int = 3600,
    ) -> None:
        """Compute page hashes for token_ids and store all of them in Redis."""
        pages = hash_pages(token_ids)
        if not pages:
            return
        try:
            async with self._client() as client:
                pipe = client.pipeline(transaction=False)
                for h in pages:
                    pipe.setex(f"kv:{h}", ttl_s, backend_id)
                await pipe.execute()
        except Exception as exc:
            logger.warning("LocalityMap.record_request failed: %s", exc)

    # ------------------------------------------------------------------
    # Legacy M1 API (backward compat)
    # ------------------------------------------------------------------

    async def get_backend(self, prefix_hash: str) -> str | None:
        """Legacy alias for get()."""
        return await self.get(prefix_hash)

    async def record(self, prefix_hash: str, backend_id: str) -> None:
        """Legacy alias for set()."""
        await self.set(prefix_hash, backend_id)
