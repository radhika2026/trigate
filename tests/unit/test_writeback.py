"""Unit tests for M2.5/M2.6 — cache write-back (fire-and-forget)."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from gateway.cache import interceptor as ic
from gateway.cache.store import CacheStore
from gateway.models import GatewayHints, Message, RequestContext, TenantConfig


def _make_ctx(text: str = "What is 2+2?") -> RequestContext:
    return RequestContext(
        request_id="req-wb-1",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1", cache_ttl_s=3600),
        messages=[Message(role="user", content=text)],
    )


@pytest.fixture(autouse=True)
async def reset_store():
    """Reset module-level store before each test."""
    original = ic._store
    store = CacheStore()
    await store.init()
    ic.set_store(store)
    yield store
    ic.set_store(original)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# check_cache — MISS path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_cache_miss(reset_store):
    ctx = _make_ctx("A completely unique query never seen before xyz")
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False
    assert ctx.domain != ""


@pytest.mark.asyncio
async def test_check_cache_sets_domain(reset_store):
    ctx = _make_ctx("Write a Python function")
    await ic.check_cache(ctx)
    assert ctx.domain == "code"


# ---------------------------------------------------------------------------
# check_cache — HIT path (seed store manually)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_cache_hit(reset_store):
    from gateway.cache.embedder import embed, load_model
    load_model("BAAI/bge-small-en-v1.5")

    query = "Who wrote Hamlet?"
    vec = await embed(query)

    import uuid
    from gateway.models import CacheEntry
    now = datetime.now(tz=timezone.utc)
    entry = CacheEntry(
        id=str(uuid.uuid4()),
        embedding=vec,
        response="William Shakespeare wrote Hamlet.",
        model_id="test",
        quality=1.0,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    await reset_store.write(entry)

    ctx = _make_ctx(query)
    result = await ic.check_cache(ctx)
    assert result == "William Shakespeare wrote Hamlet."
    assert ctx.cache_hit is True


# ---------------------------------------------------------------------------
# store_in_cache — fire-and-forget, non-blocking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_in_cache_nonblocking(reset_store):
    from gateway.cache.embedder import load_model
    load_model("BAAI/bge-small-en-v1.5")

    ctx = _make_ctx("unique query for write-back test")
    ctx.chosen_model = "test-model"

    t0 = time.perf_counter()
    await ic.store_in_cache(ctx, "some response")
    delta_ms = (time.perf_counter() - t0) * 1000

    # store_in_cache itself (just create_task) should return near-instantly
    assert delta_ms < 50, f"store_in_cache took {delta_ms:.1f}ms — too slow"

    # Allow background task to complete
    await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Degraded mode: no store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_cache_no_store_degraded():
    ic.set_store(None)  # type: ignore[arg-type]
    ctx = _make_ctx("query without store")
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False


# ---------------------------------------------------------------------------
# disable_cache hint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_cache_disabled_by_hint(reset_store):
    ctx = _make_ctx("query with cache disabled")
    ctx.gateway_hints = GatewayHints(disable_cache=True)
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False
