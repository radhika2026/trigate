"""Unit tests for observability/metrics.py: all 14 metrics present at /metrics."""
from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from gateway.observability.metrics import ALL_METRIC_NAMES


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_14_metrics_are_defined():
    """All 14 required metric names must be importable."""
    assert len(ALL_METRIC_NAMES) == 14


def test_all_metric_names_unique():
    assert len(set(ALL_METRIC_NAMES)) == len(ALL_METRIC_NAMES)


def test_all_metrics_importable():
    """Every metric object must be importable from metrics module."""
    import gateway.observability.metrics as m

    # Verify each named metric exists as an attribute
    for name in ALL_METRIC_NAMES:
        obj = getattr(m, name, None)
        assert obj is not None, f"Metric {name!r} not found in metrics module"


def test_metrics_appear_in_prometheus_output():
    """After importing, metric names should appear in generate_latest() output."""
    # Trigger import to register metrics
    import gateway.observability.metrics as _m  # noqa: F401

    output = generate_latest().decode()

    for name in ALL_METRIC_NAMES:
        assert name in output, f"Metric {name!r} not found in /metrics output"


def test_counter_labels_gateway_requests_total():
    from gateway.observability.metrics import gateway_requests_total

    # Should accept all required labels without error
    gateway_requests_total.labels(
        tier="small", cache_hit="false", kv_hit="false", domain="unknown"
    ).inc()


def test_histogram_labels_gateway_ttft_ms():
    from gateway.observability.metrics import gateway_ttft_ms

    gateway_ttft_ms.labels(tier="mid", cache_hit="false", kv_hit="false").observe(100.0)


def test_histogram_labels_gateway_total_ms():
    from gateway.observability.metrics import gateway_total_ms

    gateway_total_ms.labels(tier="frontier").observe(500.0)


def test_histogram_labels_gateway_cost_usd():
    from gateway.observability.metrics import gateway_cost_usd

    gateway_cost_usd.labels(tier="small").observe(0.001)


def test_counter_labels_gateway_cache_hit_total():
    from gateway.observability.metrics import gateway_cache_hit_total

    gateway_cache_hit_total.labels(domain="code").inc()


def test_counter_labels_gateway_kv_hits_total():
    from gateway.observability.metrics import gateway_kv_hits_total

    gateway_kv_hits_total.labels(backend_id="vllm-1").inc()


def test_counter_labels_gateway_bandit_arm():
    from gateway.observability.metrics import gateway_bandit_arm

    gateway_bandit_arm.labels(arm="0", tier="small").inc()


def test_gauge_labels_gateway_backend_queue_depth():
    from gateway.observability.metrics import gateway_backend_queue_depth

    gateway_backend_queue_depth.labels(backend_id="mock-1").set(3)


def test_histogram_no_labels_gateway_kv_pages_matched():
    from gateway.observability.metrics import gateway_kv_pages_matched

    gateway_kv_pages_matched.observe(4)


def test_histogram_no_labels_gateway_embed_latency_ms():
    from gateway.observability.metrics import gateway_embed_latency_ms

    gateway_embed_latency_ms.observe(12.5)


def test_histogram_scorer_version_label():
    from gateway.observability.metrics import gateway_complexity_ms

    gateway_complexity_ms.labels(scorer_version="v1").observe(0.5)


def test_histogram_no_labels_gateway_pipeline_ms():
    from gateway.observability.metrics import gateway_pipeline_ms

    gateway_pipeline_ms.observe(45.0)
