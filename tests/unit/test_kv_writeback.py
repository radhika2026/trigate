"""Unit tests for KV write-back (M3.5)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.models import Message, RequestContext, TenantConfig
from gateway.scheduler.dispatcher import schedule_kv_writeback, select_backend


def _make_backend(bid: str) -> MagicMock:
    b = MagicMock()
    b.backend_id = bid
    b.tier = "small"
    b.model_id = "mock"
    return b


def _make_ctx(token_ids: list[int] | None = None) -> RequestContext:
    return RequestContext(
        request_id="req-wb",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content="hello")],
        token_ids=token_ids or list(range(64)),
    )


class _TrackingLocalityMap:
    """LocalityMap that records calls for assertion."""

    def __init__(self, hit_on_second: bool = True) -> None:
        self.call_count = 0
        self.recorded: list[tuple[list[int], str]] = []
        self.hit_on_second = hit_on_second
        self._stored_backend: str | None = None

    async def find_backend(self, page_hashes):
        if self.hit_on_second and self.call_count >= 1 and self._stored_backend:
            return self._stored_backend
        return None

    async def record_request(self, token_ids, backend_id, ttl_s=3600):
        self.recorded.append((token_ids, backend_id))
        self._stored_backend = backend_id
        self.call_count += 1


class TestKvWriteback:
    @pytest.mark.asyncio
    async def test_writeback_populates_locality_map(self):
        """After schedule_kv_writeback, the locality map has the entry."""
        lm = _TrackingLocalityMap()
        token_ids = list(range(64))
        schedule_kv_writeback(token_ids, "b1", locality_map=lm)
        # Drain pending tasks
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(lm.recorded) == 1
        assert lm.recorded[0][1] == "b1"

    @pytest.mark.asyncio
    async def test_second_request_is_kv_hit(self):
        """After first request's write-back, second request should be a KV hit."""
        lm = _TrackingLocalityMap(hit_on_second=True)
        backends = [_make_backend("b1"), _make_backend("b2")]
        token_ids = list(range(64))

        # First request → cold start
        ctx1 = _make_ctx(token_ids)
        chosen1 = await select_backend(ctx1, backends, locality_map=lm)
        assert ctx1.kv_hit is False

        # Simulate write-back after first request completes
        schedule_kv_writeback(token_ids, chosen1.backend_id, locality_map=lm)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Second request with same tokens → should be a KV hit
        ctx2 = _make_ctx(token_ids)
        chosen2 = await select_backend(ctx2, backends, locality_map=lm)
        assert ctx2.kv_hit is True
        assert chosen2.backend_id == chosen1.backend_id

    @pytest.mark.asyncio
    async def test_writeback_does_not_block_streaming(self):
        """Write-back should complete asynchronously without blocking."""
        lm = _TrackingLocalityMap()
        token_ids = list(range(64))

        t0 = time.monotonic()
        schedule_kv_writeback(token_ids, "b1", locality_map=lm)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # The synchronous scheduling itself must be < 2ms
        assert elapsed_ms < 2.0, f"schedule_kv_writeback blocked for {elapsed_ms:.1f}ms"

        # Drain tasks
        await asyncio.sleep(0)
        assert len(lm.recorded) == 1

    @pytest.mark.asyncio
    async def test_writeback_uses_fire_and_forget(self):
        """schedule_kv_writeback must not await the record_request coroutine."""
        events: list[str] = []

        class _SlowLM:
            async def record_request(self, token_ids, backend_id, ttl_s=3600):
                events.append("start")
                await asyncio.sleep(0.1)  # simulate slow I/O
                events.append("end")

        lm = _SlowLM()
        schedule_kv_writeback(list(range(32)), "b1", locality_map=lm)
        # At this point, the coroutine hasn't even started yet
        assert "start" not in events

        # Let one tick pass — it should start but not finish
        await asyncio.sleep(0)
        # The task was created but the slow I/O hasn't completed
        assert "end" not in events
