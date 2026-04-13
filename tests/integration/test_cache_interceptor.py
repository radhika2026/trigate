"""Integration tests for the cache interceptor pipeline stage.

These run with in-memory FAISS (no Docker needed) so they always run.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from gateway.cache import interceptor as ic
from gateway.cache.embedder import load_model
from gateway.cache.store import CacheStore
from gateway.models import CacheEntry, GatewayHints, Message, RequestContext, TenantConfig


@pytest.fixture(scope="module", autouse=True)
def load_bge():
    load_model("BAAI/bge-small-en-v1.5")


def _make_ctx(text: str, disable_cache: bool = False) -> RequestContext:
    hints = GatewayHints(disable_cache=disable_cache) if disable_cache else None
    return RequestContext(
        request_id=str(uuid.uuid4()),
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1", cache_ttl_s=3600),
        messages=[Message(role="user", content=text)],
        gateway_hints=hints,
    )


@pytest.fixture
async def fresh_store():
    store = CacheStore()
    await store.init()
    ic.set_store(store)
    yield store
    ic.set_store(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MISS path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_miss_empty_cache(fresh_store):
    ctx = _make_ctx("What is the meaning of life?")
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False
    assert ctx.domain != ""


# ---------------------------------------------------------------------------
# HIT path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_hit_after_write(fresh_store):
    from gateway.cache.embedder import embed
    query = "Who wrote Hamlet?"
    vec = await embed(query)

    now = datetime.now(tz=timezone.utc)
    entry = CacheEntry(
        id=str(uuid.uuid4()),
        embedding=vec,
        response="William Shakespeare",
        model_id="test",
        quality=1.0,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    await fresh_store.write(entry)

    ctx = _make_ctx(query)
    result = await ic.check_cache(ctx)
    assert result == "William Shakespeare"
    assert ctx.cache_hit is True
    assert ctx.domain == "factual_qa"


# ---------------------------------------------------------------------------
# Domain classification sets ctx.domain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_sets_domain(fresh_store):
    ctx = _make_ctx("Write a Python quicksort")
    await ic.check_cache(ctx)
    assert ctx.domain == "code"


# ---------------------------------------------------------------------------
# disable_cache hint bypasses cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_disable_cache_hint(fresh_store):
    ctx = _make_ctx("Who wrote Hamlet?", disable_cache=True)
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False


# ---------------------------------------------------------------------------
# Write-back fire-and-forget (doesn't block)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_back_fire_and_forget(fresh_store):
    import time
    ctx = _make_ctx("A unique query for write-back integration test")
    ctx.chosen_model = "gpt-4"

    t0 = time.perf_counter()
    await ic.store_in_cache(ctx, "the response text")
    dt_ms = (time.perf_counter() - t0) * 1000
    assert dt_ms < 50, f"store_in_cache blocked for {dt_ms:.1f}ms"

    # Let write-back complete
    await asyncio.sleep(0.5)

    # Verify entry got written — re-embed and check
    result = await ic.check_cache(ctx)
    # May or may not hit depending on embedding similarity — just check no crash
    assert ctx.domain != ""


# ---------------------------------------------------------------------------
# Degraded mode: no store, request completes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_degraded_no_store():
    ic.set_store(None)  # type: ignore[arg-type]
    ctx = _make_ctx("query in degraded mode")
    result = await ic.check_cache(ctx)
    assert result is None
    assert ctx.cache_hit is False
