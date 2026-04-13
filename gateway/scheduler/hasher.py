"""Prefix hasher for KV-cache locality (M3 stub)."""
from __future__ import annotations

import hashlib

from gateway.models import RequestContext


def compute_prefix_hash(ctx: RequestContext, prefix_len: int = 32) -> str:
    """Hash the first ``prefix_len`` token IDs to identify shared prefixes.

    Full implementation in M3.
    """
    tokens = ctx.token_ids[:prefix_len]
    raw = ",".join(str(t) for t in tokens)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
