"""Unit tests for gateway.scheduler.dispatcher (M3.4)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.models import GatewayHints, Message, RequestContext, TenantConfig
from gateway.scheduler.dispatcher import dispatch, select_backend
from gateway.scheduler.locality_map import LocalityMap


def _make_backend(bid: str) -> MagicMock:
    b = MagicMock()
    b.backend_id = bid
    b.tier = "small"
    b.model_id = "mock"
    return b


def _make_ctx(token_ids: list[int] | None = None) -> RequestContext:
    return RequestContext(
        request_id="req-1",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content="hello")],
        token_ids=token_ids or [],
    )


# ---------------------------------------------------------------------------
# Helper mock locality map
# ---------------------------------------------------------------------------

class _MockLocalityMap:
    def __init__(self, result: str | None = None) -> None:
        self.result = result
        self.recorded: list[tuple[list[int], str]] = []

    async def find_backend(self, page_hashes: list[str]) -> str | None:
        return self.result

    async def record_request(self, token_ids: list[int], backend_id: str, ttl_s: int = 3600) -> None:
        self.recorded.append((token_ids, backend_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectBackendKvHit:
    @pytest.mark.asyncio
    async def test_kv_hit_routes_to_correct_backend(self):
        backends = [_make_backend("b1"), _make_backend("b2")]
        ctx = _make_ctx(list(range(64)))  # 4 pages of 16 tokens

        lm = _MockLocalityMap(result="b2")
        chosen = await select_backend(ctx, backends, locality_map=lm)

        assert chosen.backend_id == "b2"
        assert ctx.kv_hit is True
        assert ctx.chosen_backend == "b2"

    @pytest.mark.asyncio
    async def test_kv_hit_not_set_on_cold_start(self):
        backends = [_make_backend("b1"), _make_backend("b2")]
        ctx = _make_ctx(list(range(64)))

        lm = _MockLocalityMap(result=None)
        chosen = await select_backend(ctx, backends, locality_map=lm)

        assert ctx.kv_hit is False

    @pytest.mark.asyncio
    async def test_unknown_backend_id_falls_back(self):
        """If locality map returns an ID not in backends list, fall back."""
        backends = [_make_backend("b1"), _make_backend("b2")]
        ctx = _make_ctx(list(range(32)))

        lm = _MockLocalityMap(result="b-unknown")
        chosen = await select_backend(ctx, backends, locality_map=lm)

        # Should fall back gracefully, not raise
        assert chosen.backend_id in {"b1", "b2"}
        assert ctx.kv_hit is False


class TestSelectBackendColdStart:
    @pytest.mark.asyncio
    async def test_cold_start_distributes_across_backends(self):
        """Multiple cold requests should not all go to backend[0]."""
        backends = [_make_backend(f"b{i}") for i in range(3)]
        chosen_ids: set[str] = set()

        lm = _MockLocalityMap(result=None)
        for i in range(9):
            ctx = _make_ctx()  # empty token_ids → no page hashes
            chosen = await select_backend(ctx, backends, locality_map=lm)
            chosen_ids.add(chosen.backend_id)

        # Round-robin should hit all 3 backends
        assert len(chosen_ids) == 3

    @pytest.mark.asyncio
    async def test_no_backends_raises(self):
        ctx = _make_ctx()
        with pytest.raises(RuntimeError, match="No backends available"):
            await select_backend(ctx, [], locality_map=_MockLocalityMap())

    @pytest.mark.asyncio
    async def test_single_backend_always_chosen(self):
        backends = [_make_backend("only")]
        ctx = _make_ctx()
        lm = _MockLocalityMap(result=None)
        chosen = await select_backend(ctx, backends, locality_map=lm)
        assert chosen.backend_id == "only"


class TestSelectBackendRedisUnavailable:
    @pytest.mark.asyncio
    async def test_falls_back_on_locality_map_error(self):
        """If locality map raises, dispatch should not crash."""
        class _BrokenLM:
            async def find_backend(self, page_hashes):
                raise ConnectionError("Redis down")
            async def record_request(self, *a, **kw):
                pass

        backends = [_make_backend("b1"), _make_backend("b2")]
        ctx = _make_ctx(list(range(32)))
        chosen = await select_backend(ctx, backends, locality_map=_BrokenLM())
        assert chosen.backend_id in {"b1", "b2"}


class TestDispatchAlias:
    @pytest.mark.asyncio
    async def test_dispatch_returns_backend(self):
        """dispatch() is a public alias for select_backend()."""
        backends = [_make_backend("b1")]
        ctx = _make_ctx()

        # Patch _get_locality_map to avoid Redis
        with patch("gateway.scheduler.dispatcher._get_locality_map") as mock_glm:
            lm = _MockLocalityMap(result=None)
            mock_glm.return_value = lm
            chosen = await dispatch(ctx, backends)

        assert chosen.backend_id == "b1"
