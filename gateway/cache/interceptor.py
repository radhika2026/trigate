"""Cache interceptor: lookup → HIT/MISS → async write-back — M2 implementation."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from gateway.cache.domain import classify_domain, get_threshold
from gateway.cache.embedder import embed
from gateway.cache.store import CacheStore
from gateway.models import CacheEntry, RequestContext

logger = logging.getLogger(__name__)

# Module-level singleton store (initialised lazily / via lifespan)
_store: CacheStore | None = None


def set_store(store: CacheStore) -> None:
    """Inject the CacheStore to use. Called during app lifespan startup."""
    global _store
    _store = store


def get_store() -> CacheStore | None:
    return _store


def _build_text(ctx: RequestContext) -> str:
    """Extract the text to embed from a request context."""
    user_messages = [m.content for m in ctx.messages if m.role == "user"]
    return " ".join(user_messages) if user_messages else ""


async def check_cache(ctx: RequestContext) -> str | None:
    """Pipeline stage: look up semantic cache.

    Sets ctx.domain and ctx.cache_hit.
    Returns cached response string on HIT, None on MISS.
    When pgvector/store unavailable → returns None (degraded mode, no crash).
    """
    # Classify domain regardless (sets ctx.domain)
    domain = classify_domain(ctx)
    ctx.domain = domain

    # Skip cache if disabled by hint
    if ctx.gateway_hints and ctx.gateway_hints.disable_cache:
        ctx.cache_hit = False
        return None

    store = _store
    if store is None:
        ctx.cache_hit = False
        return None

    text = _build_text(ctx)
    if not text:
        ctx.cache_hit = False
        return None

    try:
        embedding = await embed(text)
        ctx.query_embedding = embedding

        threshold = get_threshold(domain)
        entry = await store.read(embedding, threshold, domain)

        if entry is not None:
            ctx.cache_hit = True
            logger.debug(
                "Cache HIT request_id=%s domain=%s", ctx.request_id, domain
            )
            return entry.response

    except Exception as exc:  # noqa: BLE001
        # Degraded mode: cache failure must not crash the request
        logger.warning("Cache lookup error (degraded mode): %s", exc)

    ctx.cache_hit = False
    return None


async def store_in_cache(ctx: RequestContext, response: str) -> None:
    """Fire-and-forget write-back. Called after streaming completes.

    Uses asyncio.create_task so it does NOT block TTFT.
    """
    asyncio.create_task(_write_back(ctx, response))


async def _write_back(ctx: RequestContext, response: str) -> None:
    """Background task: embed + write to cache store."""
    store = _store
    if store is None:
        return

    try:
        text = _build_text(ctx)
        if not text:
            return

        # Re-use already-computed embedding if available
        if ctx.query_embedding:
            embedding = ctx.query_embedding
        else:
            embedding = await embed(text)

        ttl_s: int = 86400  # default 1 day
        if ctx.tenant_config:
            ttl_s = ctx.tenant_config.cache_ttl_s

        now = datetime.now(tz=timezone.utc)
        entry = CacheEntry(
            id=str(uuid.uuid4()),
            embedding=embedding,
            response=response,
            model_id=ctx.chosen_model or "unknown",
            quality=None,
            hit_count=0,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_s),
        )
        await store.write(entry)
        logger.debug(
            "Cache write-back done request_id=%s domain=%s",
            ctx.request_id,
            ctx.domain,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write-back error: %s", exc)
