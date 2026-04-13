"""Integration tests for LocalityMap using testcontainers Redis (M3.3)."""
from __future__ import annotations

import asyncio
import time

import pytest
import pytest_asyncio

try:
    from testcontainers.redis import RedisContainer
    HAS_DOCKER = True
except Exception:
    HAS_DOCKER = False

from gateway.scheduler.locality_map import LocalityMap


@pytest.fixture(scope="module")
def redis_url_for_locality():
    if not HAS_DOCKER:
        pytest.skip("testcontainers / Docker not available")
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture
def lm(redis_url_for_locality):
    return LocalityMap(redis_url=redis_url_for_locality)


# ---------------------------------------------------------------------------
# Basic set/get
# ---------------------------------------------------------------------------

class TestSetGet:
    @pytest.mark.asyncio
    async def test_set_then_get(self, lm):
        await lm.set("hash-abc", "backend-1")
        result = await lm.get("hash-abc")
        assert result == "backend-1"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, lm):
        result = await lm.get("nonexistent-hash-xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrite(self, lm):
        await lm.set("hash-over", "b1")
        await lm.set("hash-over", "b2")
        result = await lm.get("hash-over")
        assert result == "b2"

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self, lm):
        await lm.set("hash-ttl-short", "b1", ttl_s=1)
        result_before = await lm.get("hash-ttl-short")
        assert result_before == "b1"
        await asyncio.sleep(1.5)
        result_after = await lm.get("hash-ttl-short")
        assert result_after is None


# ---------------------------------------------------------------------------
# find_backend
# ---------------------------------------------------------------------------

class TestFindBackend:
    @pytest.mark.asyncio
    async def test_find_returns_none_for_empty_list(self, lm):
        result = await lm.find_backend([])
        assert result is None

    @pytest.mark.asyncio
    async def test_find_returns_match(self, lm):
        await lm.set("page-hash-1", "b1")
        result = await lm.find_backend(["page-hash-1"])
        assert result == "b1"

    @pytest.mark.asyncio
    async def test_find_longest_prefix_wins(self, lm):
        """last match in the list wins = longest prefix."""
        await lm.set("ph-long-1", "b1")
        await lm.set("ph-long-2", "b2")
        # hashes ordered: [page1_hash, page2_hash]
        # page2 is longer prefix → b2 should win
        result = await lm.find_backend(["ph-long-1", "ph-long-2"])
        assert result == "b2"

    @pytest.mark.asyncio
    async def test_find_partial_match(self, lm):
        """Only first page matches → returns that backend."""
        await lm.set("ph-partial-1", "b1")
        result = await lm.find_backend(["ph-partial-1", "ph-partial-missing"])
        assert result == "b1"

    @pytest.mark.asyncio
    async def test_find_no_match_returns_none(self, lm):
        result = await lm.find_backend(["missing-hash-a", "missing-hash-b"])
        assert result is None


# ---------------------------------------------------------------------------
# record_request
# ---------------------------------------------------------------------------

class TestRecordRequest:
    @pytest.mark.asyncio
    async def test_record_stores_all_pages(self, lm):
        token_ids = list(range(64))  # 4 pages of 16
        await lm.record_request(token_ids, "b-record")
        # find_backend should now hit
        result = await lm.find_backend(
            # We use the hasher to get the expected page hashes
            __import__("gateway.scheduler.hasher", fromlist=["hash_pages"]).hash_pages(token_ids)
        )
        assert result == "b-record"

    @pytest.mark.asyncio
    async def test_record_empty_tokens_no_op(self, lm):
        # Should not raise; nothing stored
        await lm.record_request([], "b-empty")


# ---------------------------------------------------------------------------
# Performance (P99 round-trip < 2ms)
# ---------------------------------------------------------------------------

class TestPerformance:
    @pytest.mark.asyncio
    async def test_write_roundtrip_p99_under_2ms(self, lm):
        """P99 of set+get on localhost Redis must be < 2ms."""
        timings: list[float] = []
        for i in range(100):
            t0 = time.perf_counter()
            await lm.set(f"perf-hash-{i}", f"b{i % 3}")
            val = await lm.get(f"perf-hash-{i}")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)
            assert val == f"b{i % 3}"

        timings.sort()
        p99 = timings[98]  # 99th percentile of 100 samples
        assert p99 < 2.0, f"P99 round-trip was {p99:.2f}ms (expected < 2ms)"
