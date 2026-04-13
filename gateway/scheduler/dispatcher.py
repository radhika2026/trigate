"""KV-aware dispatcher (M3 stub)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.backends.base import Backend

from gateway.models import RequestContext

logger = logging.getLogger(__name__)


async def dispatch(
    ctx: RequestContext,
    backends: list["Backend"],
) -> "Backend":
    """Select the best backend using KV locality (M1 stub: round-robin).

    Full implementation in M3.
    """
    if not backends:
        raise RuntimeError("No backends available")
    # M1: just pick first available
    return backends[0]
