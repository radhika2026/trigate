"""Integration tests for CacheStore with testcontainers pgvector.

These tests require Docker. If Docker / testcontainers is unavailable they are
skipped automatically — the unit tests cover the FAISS path.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

try:
    from testcontainers.postgres import PostgresContainer  # type: ignore[import]
    import asyncpg  # type: ignore[import]
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@pytest.fixture(scope="module")
def pgvector_dsn():
    if not HAS_DEPS:
        pytest.skip("testcontainers / asyncpg not available")
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as pg:
            dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
            yield dsn
    except Exception:
        pytest.skip("Docker not available")


@pytest.fixture(scope="module")
async def pool(pgvector_dsn):
    p = await asyncpg.create_pool(pgvector_dsn)
    yield p
    await p.close()


@pytest.fixture(scope="module")
async def cache_store(pool):
    from gateway.cache.store import CacheStore
    s = CacheStore(pool=pool)
    await s.init()
    return s


def _entry(vec: list[float], ttl_s: int = 3600) -> "CacheEntry":  # type: ignore[name-defined]
    from gateway.models import CacheEntry
    now = datetime.now(tz=timezone.utc)
    return CacheEntry(
        id=str(uuid.uuid4()),
        embedding=vec,
        response="pg integration response",
        model_id="test-model",
        quality=0.9,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_s),
    )


@pytest.mark.asyncio
async def test_pgvector_write_read_hit(cache_store):
    import math
    vec = [0.0] * 384
    vec[0] = 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    vec = [x / norm for x in vec]

    e = _entry(vec)
    await cache_store.write(e)

    result = await cache_store.read(vec, threshold=0.9)
    assert result is not None
    assert result.response == "pg integration response"


@pytest.mark.asyncio
async def test_pgvector_expired_miss(cache_store):
    import math
    vec = [0.0] * 384
    vec[1] = 1.0  # different vector
    norm = math.sqrt(sum(x * x for x in vec))
    vec = [x / norm for x in vec]

    now = datetime.now(tz=timezone.utc)
    from gateway.models import CacheEntry
    expired = CacheEntry(
        id=str(uuid.uuid4()),
        embedding=vec,
        response="expired",
        model_id="m",
        quality=None,
        hit_count=0,
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    await cache_store.write(expired)
    result = await cache_store.read(vec, threshold=0.5)
    assert result is None
