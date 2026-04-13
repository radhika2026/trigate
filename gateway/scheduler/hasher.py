"""Prefix hasher for KV-cache locality (M3)."""
from __future__ import annotations

import hashlib
import struct

from gateway.config import get_settings


def hash_pages(token_ids: list[int], page_size: int | None = None) -> list[str]:
    """Compute rolling cumulative page hashes for a token sequence.

    Each page covers tokens[0:page_size*i] for i in 1..N.
    Hash input is raw bytes: each token encoded as 2-byte big-endian uint16.

    Args:
        token_ids: List of integer token IDs.
        page_size: Number of tokens per page. Defaults to settings.vllm_block_size.

    Returns:
        List of hex-string SHA-256 hashes, one per complete page.
        Empty list if token_ids is empty or shorter than one page.
    """
    if not token_ids:
        return []

    if page_size is None:
        page_size = get_settings().vllm_block_size

    num_pages = len(token_ids) // page_size
    hashes: list[str] = []

    for i in range(1, num_pages + 1):
        end = i * page_size
        # Encode tokens as 2-byte big-endian (cumulative window [0:end])
        raw = struct.pack(f">{end}H", *token_ids[:end])
        digest = hashlib.sha256(raw).hexdigest()
        hashes.append(digest)

    return hashes


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility with M1/M2 code that calls
# compute_prefix_hash(ctx).
# ---------------------------------------------------------------------------

def compute_prefix_hash(ctx: "RequestContext", prefix_len: int = 32) -> str:  # type: ignore[name-defined]  # noqa: F821
    """Legacy: Hash the first ``prefix_len`` token IDs.

    Kept for backward compatibility; M3+ code should use hash_pages().
    """
    from gateway.models import RequestContext  # local import avoids circular

    tokens = ctx.token_ids[:prefix_len]
    raw = struct.pack(f">{len(tokens)}H", *tokens) if tokens else b""
    return hashlib.sha256(raw).hexdigest()[:16]
