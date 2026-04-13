"""Unit tests for queue-depth probe (M3.6)."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.scheduler.dispatcher import (
    _get_queue_depth,
    _parse_queue_depth,
    _PROBE_STALE_S,
    _queue_depths,
)


class TestParseQueueDepth:
    def test_parses_waiting_metric(self):
        metrics = (
            "# HELP vllm_num_requests_waiting\n"
            "# TYPE vllm_num_requests_waiting gauge\n"
            "vllm_num_requests_waiting 5\n"
        )
        assert _parse_queue_depth(metrics) == 5

    def test_zero_when_metric_absent(self):
        assert _parse_queue_depth("") == 0
        assert _parse_queue_depth("some_other_metric 99\n") == 0

    def test_parses_float_representation(self):
        metrics = "vllm_num_requests_waiting 3.0\n"
        assert _parse_queue_depth(metrics) == 3

    def test_handles_multiline(self):
        metrics = "foo 1\nvllm_num_requests_waiting 7\nbar 2\n"
        assert _parse_queue_depth(metrics) == 7


class TestGetQueueDepth:
    def setup_method(self):
        # Clear stale state
        _queue_depths.clear()

    def test_returns_none_when_no_probe(self):
        b = MagicMock()
        b.backend_id = "no-probe"
        assert _get_queue_depth(b) is None

    def test_returns_depth_when_fresh(self):
        b = MagicMock()
        b.backend_id = "fresh-b"
        _queue_depths["fresh-b"] = (7, time.monotonic())
        assert _get_queue_depth(b) == 7

    def test_returns_none_when_stale(self):
        b = MagicMock()
        b.backend_id = "stale-b"
        old_ts = time.monotonic() - (_PROBE_STALE_S + 1)
        _queue_depths["stale-b"] = (3, old_ts)
        assert _get_queue_depth(b) is None

    def teardown_method(self):
        _queue_depths.clear()


class TestBackendWithMetricsUrl:
    @pytest.mark.asyncio
    async def test_probe_skipped_when_no_metrics_url(self):
        """Backends without metrics_url are silently skipped."""
        from gateway.scheduler.dispatcher import _probe_queue_depth
        b = MagicMock(spec=["backend_id", "tier", "model_id"])
        b.backend_id = "no-url-b"
        # No metrics_url attribute → should not raise
        await _probe_queue_depth(b)
        assert "no-url-b" not in _queue_depths

    @pytest.mark.asyncio
    async def test_probe_updates_queue_depth(self):
        """When metrics_url is present and reachable, depth is recorded."""
        from gateway.scheduler.dispatcher import _probe_queue_depth

        b = MagicMock()
        b.backend_id = "metrics-b"
        b.metrics_url = "http://fake/metrics"

        metrics_text = "vllm_num_requests_waiting 4\n"
        mock_resp = MagicMock()
        mock_resp.text = metrics_text

        with patch("gateway.scheduler.dispatcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await _probe_queue_depth(b)

        assert "metrics-b" in _queue_depths
        depth, ts = _queue_depths["metrics-b"]
        assert depth == 4

    def teardown_method(self, _):
        _queue_depths.clear()
