"""pgvector HNSW index with pure-numpy in-memory fallback — M2 implementation."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from gateway.models import CacheEntry

logger = logging.getLogger(__name__)

# DDL executed at startup
_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cache_entries (
    id TEXT PRIMARY KEY,
    embedding vector(384) NOT NULL,
    response TEXT NOT NULL,
    model_id TEXT NOT NULL,
    quality FLOAT,
    hit_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS cache_embedding_hnsw
    ON cache_entries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


class _InMemoryIndex:
    """Thread-safe pure-numpy cosine similarity index.

    Used when pgvector is unavailable. No FAISS/OpenMP dependencies so it
    is safe to use inside asyncio event loops on macOS.
    """

    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def search(
        self,
        embedding: list[float],
        threshold: float,
        limit: int = 5,
    ) -> list[CacheEntry]:
        with self._lock:
            entries_snapshot = list(self._entries)

        if not entries_snapshot:
            return []

        query = np.array(embedding, dtype=np.float32)
        now = datetime.now(tz=timezone.utc)
        scored: list[tuple[float, CacheEntry]] = []

        for entry in entries_snapshot:
            exp = entry.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp < now:
                continue
            vec = np.array(entry.embedding, dtype=np.float32)
            # Both vectors are L2-normalised → dot product == cosine similarity
            sim = float(np.dot(query, vec))
            if sim >= threshold:
                scored.append((sim, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def increment_hit_count(self, entry_id: str) -> None:
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    e.hit_count += 1
                    break

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries = []
        return count

    def get_by_id(self, entry_id: str) -> CacheEntry | None:
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    return e
        return None


class VectorIndex:
    """Async vector index: uses pgvector HNSW when available, numpy otherwise."""

    def __init__(self, pool: Any | None = None) -> None:
        self._pool = pool
        self._mem: _InMemoryIndex | None = None
        self._use_mem = pool is None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Run DDL and set up schema. Falls back to in-memory if pool is None."""
        if self._pool is None:
            logger.info("No DB pool provided — using in-memory index")
            self._mem = _InMemoryIndex()
            self._use_mem = True
            return

        try:
            async with self._pool.acquire() as conn:
                # pgvector DDL must run statement-by-statement
                for stmt in _DDL.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        await conn.execute(stmt)
            self._use_mem = False
            logger.info("pgvector schema initialised")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector init failed (%s) — falling back to in-memory index", exc)
            self._mem = _InMemoryIndex()
            self._use_mem = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        embedding: list[float],
        threshold: float,
        limit: int = 5,
    ) -> list[CacheEntry]:
        """Return CacheEntry list with cosine similarity ≥ threshold."""
        if self._use_mem:
            return self._mem.search(embedding, threshold, limit) if self._mem else []

        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                # pgvector cosine distance = 1 - cosine_similarity
                min_dist = 1.0 - threshold
                rows = await conn.fetch(
                    """
                    SELECT id, embedding, response, model_id, quality,
                           hit_count, created_at, expires_at
                    FROM cache_entries
                    WHERE expires_at > now()
                      AND (embedding <=> $1::vector) <= $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    str(embedding),
                    min_dist,
                    limit,
                )
                return [_row_to_entry(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector search error: %s", exc)
            return []

    async def upsert(self, entry: CacheEntry) -> None:
        """Insert or update a cache entry."""
        if self._use_mem:
            if self._mem is not None:
                self._mem.add(entry)
            return

        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO cache_entries
                        (id, embedding, response, model_id, quality,
                         hit_count, created_at, expires_at)
                    VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding   = EXCLUDED.embedding,
                        response    = EXCLUDED.response,
                        model_id    = EXCLUDED.model_id,
                        quality     = EXCLUDED.quality,
                        expires_at  = EXCLUDED.expires_at
                    """,
                    entry.id,
                    str(entry.embedding),
                    entry.response,
                    entry.model_id,
                    entry.quality,
                    entry.hit_count,
                    entry.created_at,
                    entry.expires_at,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector upsert error: %s", exc)

    async def increment_hit_count(self, entry_id: str) -> None:
        """Atomically increment hit_count for a cache entry."""
        if self._use_mem:
            if self._mem:
                self._mem.increment_hit_count(entry_id)
            return
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = $1",
                    entry_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("hit_count increment error: %s", exc)

    # Legacy insert API (M1 compat)
    async def insert(self, entry_id: str, embedding: list[float]) -> None:
        """Legacy stub kept for backward compat."""
        pass


def _row_to_entry(row: Any) -> CacheEntry:
    """Convert asyncpg row to CacheEntry."""
    return CacheEntry(
        id=row["id"],
        embedding=list(row["embedding"]),
        response=row["response"],
        model_id=row["model_id"],
        quality=row["quality"],
        hit_count=row["hit_count"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )
