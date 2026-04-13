"""Unit tests for CacheStore (in-memory FAISS mode)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gateway.cache.index import VectorIndex
from gateway.cache.store import CacheStore
from gateway.models import CacheEntry


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _entry(
    embedding: list[float],
    response: str = "cached response",
    ttl_s: int = 3600,
) -> CacheEntry:
    now = _now()
    import uuid
    return CacheEntry(
        id=str(uuid.uuid4()),
        embedding=embedding,
        response=response,
        model_id="test-model",
        quality=0.9,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_s),
    )


def _vec(n: int = 384, value: float = 0.0) -> list[float]:
    """Return a unit vector with value at index 0."""
    import math
    v = [0.0] * n
    v[0] = 1.0 if value == 0.0 else value
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


@pytest.fixture
async def store() -> CacheStore:  # type: ignore[misc]
    s = CacheStore()  # uses FAISS fallback
    await s.init()
    return s


# ---------------------------------------------------------------------------
# Basic read/write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_miss_empty_store(store):
    result = await store.read(_vec(), threshold=0.9)
    assert result is None


@pytest.mark.asyncio
async def test_write_then_read_hit(store):
    vec = _vec()
    e = _entry(vec, response="hello world")
    await store.write(e)
    result = await store.read(vec, threshold=0.9)
    assert result is not None
    assert result.response == "hello world"


@pytest.mark.asyncio
async def test_read_below_threshold_returns_none(store):
    import math
    # Insert a vector pointing at index 0
    v1 = [0.0] * 384
    v1[0] = 1.0
    e = _entry(v1)
    await store.write(e)

    # Query with a vector pointing at index 1 (orthogonal)
    v2 = [0.0] * 384
    v2[1] = 1.0
    result = await store.read(v2, threshold=0.99)
    assert result is None


@pytest.mark.asyncio
async def test_expired_entry_not_returned(store):
    vec = _vec()
    now = _now()
    e = CacheEntry(
        id="expired-1",
        embedding=vec,
        response="stale",
        model_id="m",
        quality=None,
        hit_count=0,
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),  # already expired
    )
    await store.write(e)
    # Use a fresh store to avoid other entries
    s2 = CacheStore()
    await s2.init()
    await s2.write(e)
    result = await s2.read(vec, threshold=0.5)
    assert result is None


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush(store):
    vec = _vec()
    await store.write(_entry(vec))
    count = await store.flush()
    assert count >= 1
    result = await store.read(vec, threshold=0.9)
    assert result is None


# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_and_get(store):
    vec = _vec()
    e = _entry(vec)
    await store.put(e)
    result = await store.get(e.id)
    assert result is not None
    assert result.id == e.id
