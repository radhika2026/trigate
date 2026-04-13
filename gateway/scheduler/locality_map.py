"""Prefix → backend locality map (M3 stub)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LocalityMap:
    """Tracks which backend has a given KV prefix cached.

    Full implementation in M3 (Redis-backed).
    """

    async def get_backend(self, prefix_hash: str) -> str | None:
        """Return backend_id that has this prefix, or None (M1 stub)."""
        return None

    async def record(self, prefix_hash: str, backend_id: str) -> None:
        """Record that a backend now holds a prefix (M1 stub: no-op)."""
        pass
