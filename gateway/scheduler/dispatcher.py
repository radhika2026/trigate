"""KV-aware dispatcher (M3)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from gateway.backends.base import Backend

from gateway.models import RequestContext
from gateway.scheduler.hasher import hash_pages
from gateway.scheduler.locality_map import LocalityMap

logger = logging.getLogger(__name__)

# Module-level shared instances (lazy-initialised)
_locality_map: LocalityMap | None = None
_rr_counter: int = 0

# Queue-depth probe state: backend_id → (depth, last_probe_ts)
_queue_depths: dict[str, tuple[int, float]] = {}
_PROBE_INTERVAL_S = 5.0
_PROBE_STALE_S = 10.0


def _get_locality_map() -> LocalityMap:
    global _locality_map
    if _locality_map is None:
        _locality_map = LocalityMap()
    return _locality_map


# ---------------------------------------------------------------------------
# Queue-depth probe
# ---------------------------------------------------------------------------

async def _probe_queue_depth(backend: "Backend") -> None:
    """Fetch vLLM /metrics and update _queue_depths for this backend."""
    # Only vLLM backends expose a metrics URL; use the backend_id as URL base
    # if the backend has a metrics_url attribute; otherwise skip.
    metrics_url: str | None = getattr(backend, "metrics_url", None)
    if metrics_url is None:
        return
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(metrics_url)
        depth = _parse_queue_depth(resp.text)
        _queue_depths[backend.backend_id] = (depth, time.monotonic())
    except Exception as exc:
        logger.debug("Queue-depth probe failed for %s: %s", backend.backend_id, exc)


def _parse_queue_depth(metrics_text: str) -> int:
    """Parse vllm_num_requests_waiting from Prometheus text format."""
    for line in metrics_text.splitlines():
        if line.startswith("vllm_num_requests_waiting"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(float(parts[-1]))
                except ValueError:
                    pass
    return 0


def _get_queue_depth(backend: "Backend") -> int | None:
    """Return queue depth if probe is fresh, else None."""
    entry = _queue_depths.get(backend.backend_id)
    if entry is None:
        return None
    depth, ts = entry
    if time.monotonic() - ts > _PROBE_STALE_S:
        return None  # stale
    return depth


async def _maybe_probe_backends(backends: list["Backend"]) -> None:
    """Fire background probe tasks if probe interval has elapsed."""
    now = time.monotonic()
    tasks = []
    for b in backends:
        entry = _queue_depths.get(b.backend_id)
        if entry is None or (now - entry[1]) >= _PROBE_INTERVAL_S:
            tasks.append(_probe_queue_depth(b))
    if tasks:
        # Fire-and-forget; do not await so dispatch is not blocked
        for coro in tasks:
            asyncio.create_task(coro)


# ---------------------------------------------------------------------------
# Backend selection helpers
# ---------------------------------------------------------------------------

def _round_robin(backends: list["Backend"]) -> "Backend":
    global _rr_counter
    backend = backends[_rr_counter % len(backends)]
    _rr_counter += 1
    return backend


def _least_queue(backends: list["Backend"]) -> "Backend":
    """Pick the backend with smallest known queue depth.

    Falls back to round-robin if no probes are available or all are stale.
    """
    candidates: list[tuple[int, "Backend"]] = []
    for b in backends:
        depth = _get_queue_depth(b)
        if depth is not None:
            candidates.append((depth, b))

    if not candidates:
        return _round_robin(backends)

    # Stable sort: pick minimum depth
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def select_backend(
    ctx: RequestContext,
    backends: list["Backend"],
    locality_map: LocalityMap | None = None,
) -> "Backend":
    """Select the best backend using KV-cache locality.

    1. Compute page hashes from ctx.token_ids.
    2. Query LocalityMap for longest-matching prefix.
    3. If hit → route to that backend, set ctx.kv_hit = True.
    4. Cold-start → distribute by queue depth (least-queue), fallback round-robin.

    Never raises; falls back to round-robin on any error.
    """
    if not backends:
        raise RuntimeError("No backends available")

    # Fire queue-depth probes in background (non-blocking)
    asyncio.create_task(_maybe_probe_backends(backends))

    lm = locality_map if locality_map is not None else _get_locality_map()

    # Build an index of available backends for fast lookup
    backend_index: dict[str, "Backend"] = {b.backend_id: b for b in backends}

    try:
        page_hashes = hash_pages(ctx.token_ids)
        if page_hashes:
            matched_id = await lm.find_backend(page_hashes)
            if matched_id and matched_id in backend_index:
                chosen = backend_index[matched_id]
                ctx.kv_hit = True
                ctx.chosen_backend = chosen.backend_id
                return chosen
    except Exception as exc:
        logger.warning("KV locality lookup failed, falling back: %s", exc)

    # Cold start: pick by queue depth
    try:
        chosen = _least_queue(backends)
    except Exception:
        chosen = _round_robin(backends)

    ctx.kv_hit = False
    ctx.chosen_backend = chosen.backend_id
    return chosen


async def dispatch(
    ctx: RequestContext,
    backends: list["Backend"],
) -> "Backend":
    """Dispatch a request to the best backend (public entry point).

    Alias for select_backend(); kept for backward compatibility with M1/M2.
    """
    return await select_backend(ctx, backends)


# ---------------------------------------------------------------------------
# KV write-back (M3.5)
# ---------------------------------------------------------------------------

def schedule_kv_writeback(
    token_ids: list[int],
    backend_id: str,
    locality_map: LocalityMap | None = None,
    ttl_s: int = 3600,
) -> None:
    """Fire-and-forget async write-back of prefix hashes to LocalityMap.

    Must be called *after* a request completes so the hashes are populated
    for subsequent requests. Never blocks streaming.
    """
    lm = locality_map if locality_map is not None else _get_locality_map()
    asyncio.create_task(lm.record_request(token_ids, backend_id, ttl_s))
