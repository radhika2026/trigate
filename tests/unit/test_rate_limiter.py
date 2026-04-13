"""Unit tests for rate_limiter.py: mock Redis, test bucket logic."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.pipeline.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_redis_mock(evalsha_return: int = 1) -> AsyncMock:
    """Return a mock Redis client."""
    redis = AsyncMock()
    redis.script_load = AsyncMock(return_value="abc123sha")
    redis.evalsha = AsyncMock(return_value=evalsha_return)
    return redis


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_script_caches_sha():
    redis = _make_redis_mock()
    rl = RateLimiter(redis)
    await rl.load_script()
    assert rl._sha == "abc123sha"
    # Called once
    redis.script_load.assert_called_once()


@pytest.mark.asyncio
async def test_allowed_request_returns_true():
    redis = _make_redis_mock(evalsha_return=1)
    rl = RateLimiter(redis)
    await rl.load_script()
    result = await rl.check("tenant1", rpm_limit=60)
    assert result is True


@pytest.mark.asyncio
async def test_denied_request_returns_false():
    redis = _make_redis_mock(evalsha_return=0)
    rl = RateLimiter(redis)
    await rl.load_script()
    result = await rl.check("tenant1", rpm_limit=60)
    assert result is False


@pytest.mark.asyncio
async def test_evalsha_called_with_correct_key():
    redis = _make_redis_mock()
    rl = RateLimiter(redis)
    await rl.load_script()
    await rl.check("tenant-xyz", rpm_limit=30)

    call_args = redis.evalsha.call_args
    # First positional: sha, second: numkeys=1, third: key name
    assert call_args[0][0] == "abc123sha"
    assert call_args[0][1] == 1
    assert call_args[0][2] == "rl:rpm:tenant-xyz"


@pytest.mark.asyncio
async def test_load_script_called_lazily_if_not_preloaded():
    """SHA is loaded lazily on first check() if load_script() was not called."""
    redis = _make_redis_mock()
    rl = RateLimiter(redis)
    assert rl._sha is None
    await rl.check("tenant1", rpm_limit=60)
    # Should have auto-loaded
    assert rl._sha is not None
    redis.script_load.assert_called_once()


@pytest.mark.asyncio
async def test_rate_limit_capacity_passed_to_evalsha():
    redis = _make_redis_mock()
    rl = RateLimiter(redis)
    await rl.load_script()
    await rl.check("tenant1", rpm_limit=120)

    call_args = redis.evalsha.call_args[0]
    capacity = call_args[3]  # ARGV[1]
    assert capacity == 120
