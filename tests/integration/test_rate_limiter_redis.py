"""Integration test: 61st request → 429 with real Redis."""
from __future__ import annotations

import asyncio

import pytest

try:
    import aioredis  # type: ignore[import]
    HAS_AIOREDIS = True
except ImportError:
    HAS_AIOREDIS = False

from gateway.pipeline.rate_limiter import RateLimiter


pytestmark = pytest.mark.skipif(not HAS_AIOREDIS, reason="aioredis not installed")


@pytest.mark.asyncio
async def test_61st_request_is_denied(redis_url: str) -> None:  # type: ignore[no-untyped-def]
    """The 60th request should succeed; the 61st should be denied."""
    redis = await aioredis.from_url(redis_url)
    rl = RateLimiter(redis)
    await rl.load_script()

    tenant_id = "integration-test-tenant"
    rpm_limit = 60

    # Flush previous bucket state
    await redis.delete(f"rl:rpm:{tenant_id}")

    results = []
    for _ in range(61):
        allowed = await rl.check(tenant_id, rpm_limit=rpm_limit)
        results.append(allowed)

    await redis.close()

    # First 60 should succeed
    assert all(results[:60]), "First 60 requests should be allowed"
    # 61st should be denied
    assert results[60] is False, "61st request should be denied (rate limited)"
