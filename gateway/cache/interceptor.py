"""Cache interceptor — checks semantic cache before forwarding (M2 stub)."""
from __future__ import annotations

import logging

from gateway.models import RequestContext

logger = logging.getLogger(__name__)


async def check_cache(ctx: RequestContext) -> str | None:
    """Check the semantic cache for a matching entry.

    Returns the cached response string if hit, else None.
    Full implementation in M2.
    """
    return None


async def store_in_cache(ctx: RequestContext, response: str) -> None:
    """Asynchronously store a response in the cache (fire-and-forget).

    Full implementation in M2.
    """
    pass
