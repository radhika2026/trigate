"""End-to-end integration tests for dispatcher with real Redis (M3.4)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from testcontainers.redis import RedisContainer
    HAS_DOCKER = True
except Exception:
    HAS_DOCKER = False

from gateway.models import Message, RequestContext, TenantConfig
from gateway.scheduler.dispatcher import select_backend
from gateway.scheduler.locality_map import LocalityMap


@pytest.fixture(scope="module")
def redis_url_e2e():
    if not HAS_DOCKER:
        pytest.skip("testcontainers / Docker not available")
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


def _make_backend(bid: str) -> MagicMock:
    b = MagicMock()
    b.backend_id = bid
    b.tier = "small"
    b.model_id = "mock"
    return b


def _make_ctx(token_ids: list[int]) -> RequestContext:
    return RequestContext(
        request_id="e2e-req",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content="hi")],
        token_ids=token_ids,
    )


class TestDispatcherE2E:
    @pytest.mark.asyncio
    async def test_kv_locality_routing_e2e(self, redis_url_e2e):
        """Full round-trip: first request cold, second request KV hit."""
        lm = LocalityMap(redis_url=redis_url_e2e)
        backends = [_make_backend("ba"), _make_backend("bb"), _make_backend("bc")]
        token_ids = list(range(128))  # 8 pages

        # First request: cold start
        ctx1 = _make_ctx(token_ids)
        chosen1 = await select_backend(ctx1, backends, locality_map=lm)
        assert ctx1.kv_hit is False

        # Manually record (simulating write-back completing)
        await lm.record_request(token_ids, chosen1.backend_id)

        # Second request with same tokens: should be a KV hit
        ctx2 = _make_ctx(token_ids)
        chosen2 = await select_backend(ctx2, backends, locality_map=lm)
        assert ctx2.kv_hit is True
        assert chosen2.backend_id == chosen1.backend_id

    @pytest.mark.asyncio
    async def test_different_prefix_different_backend(self, redis_url_e2e):
        """Two different token sequences route independently."""
        lm = LocalityMap(redis_url=redis_url_e2e)
        backends = [_make_backend("bx"), _make_backend("by")]
        tokens_a = list(range(32))
        tokens_b = list(range(100, 132))

        # Seed backend bx for tokens_a
        await lm.record_request(tokens_a, "bx")
        # Seed backend by for tokens_b
        await lm.record_request(tokens_b, "by")

        ctx_a = _make_ctx(tokens_a)
        chosen_a = await select_backend(ctx_a, backends, locality_map=lm)
        assert ctx_a.kv_hit is True
        assert chosen_a.backend_id == "bx"

        ctx_b = _make_ctx(tokens_b)
        chosen_b = await select_backend(ctx_b, backends, locality_map=lm)
        assert ctx_b.kv_hit is True
        assert chosen_b.backend_id == "by"
