"""Token-bucket rate limiter backed by Redis Lua script."""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Lua token-bucket script:
# KEYS[1] = bucket key
# ARGV[1] = capacity (max tokens = rpm_limit)
# ARGV[2] = refill_rate (tokens per second = capacity / 60)
# ARGV[3] = now (unix timestamp float)
# ARGV[4] = cost (tokens consumed = 1)
# Returns 1 if allowed, 0 if denied
_LUA_TOKEN_BUCKET = """
local key       = KEYS[1]
local capacity  = tonumber(ARGV[1])
local rate      = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])
local cost      = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    tokens      = capacity
    last_refill = now
end

-- Refill
local elapsed = now - last_refill
local refilled = elapsed * rate
tokens = math.min(capacity, tokens + refilled)

if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 10)
    return 1
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 10)
    return 0
end
"""


class RateLimiter:
    """Async Redis-backed token-bucket rate limiter.

    The Lua script is loaded ONCE at startup via ``load_script``.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._sha: str | None = None

    async def load_script(self) -> None:
        """Load Lua script into Redis and cache its SHA."""
        self._sha = await self._redis.script_load(_LUA_TOKEN_BUCKET)
        logger.info("Rate-limiter Lua script loaded, SHA=%s", self._sha)

    async def check(
        self,
        tenant_id: str,
        rpm_limit: int,
        cost: int = 1,
    ) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if self._sha is None:
            # Fallback: load on first use
            await self.load_script()

        key = f"rl:rpm:{tenant_id}"
        capacity = rpm_limit
        rate = capacity / 60.0  # tokens per second
        now = time.time()

        result = await self._redis.evalsha(
            self._sha,
            1,
            key,
            capacity,
            rate,
            now,
            cost,
        )
        return bool(result)
