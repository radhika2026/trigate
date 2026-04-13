"""Integration tests for VectorIndex pgvector path.

Requires Docker + pgvector/pgvector:pg16 image. Skipped if unavailable.
"""
from __future__ import annotations

import math
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
def pg_container():
    if not HAS_DEPS:
        pytest.skip("testcontainers / asyncpg not available")
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as pg:
            yield pg
    except Exception:
        pytest.skip("Docker not available")


@pytest.fixture(scope="module")
async def pg_pool(pg_container):
    dsn = pg_container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    pool = await asyncpg.create_pool(dsn)
    yield pool
    await pool.close()


@pytest.fixture(scope="module")
async def vector_index(pg_pool):
    from gateway.cache.index import VectorIndex
    idx = VectorIndex(pool=pg_pool)
    await idx.init()
    return idx


def _unit_vec(dim: int = 384, pos: int = 0) -> list[float]:
    v = [0.0] * dim
    v[pos] = 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _entry(vec: list[float]) -> "CacheEntry":  # type: ignore[name-defined]
    from gateway.models import CacheEntry
    now = datetime.now(tz=timezone.utc)
    return CacheEntry(
        id=str(uuid.uuid4()),
        embedding=vec,
        response="test response from pgvector",
        model_id="test",
        quality=1.0,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_pgvector_upsert_and_search(vector_index):
    vec = _unit_vec(pos=0)
    e = _entry(vec)
    await vector_index.upsert(e)

    results = await vector_index.search(vec, threshold=0.9)
    assert len(results) >= 1
    assert any(r.id == e.id for r in results)


@pytest.mark.asyncio
async def test_pgvector_below_threshold_miss(vector_index):
    vec_a = _unit_vec(pos=2)
    vec_b = _unit_vec(pos=3)  # orthogonal to vec_a
    e = _entry(vec_a)
    await vector_index.upsert(e)

    results = await vector_index.search(vec_b, threshold=0.99)
    # Should not match since they are orthogonal (sim ≈ 0)
    assert not any(r.id == e.id for r in results)


@pytest.mark.asyncio
async def test_pgvector_increment_hit_count(vector_index):
    vec = _unit_vec(pos=4)
    e = _entry(vec)
    await vector_index.upsert(e)
    await vector_index.increment_hit_count(e.id)
    # Search and check hit_count
    results = await vector_index.search(vec, threshold=0.9)
    matched = [r for r in results if r.id == e.id]
    assert matched
    assert matched[0].hit_count == 1
