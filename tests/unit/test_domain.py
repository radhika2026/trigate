"""Unit tests for M2.2 — Domain classifier."""
from __future__ import annotations

import time

import pytest

from gateway.cache.domain import (
    DOMAIN_THRESHOLDS,
    classify_domain,
    get_threshold,
    reload_thresholds,
    _classify_text,
)
from gateway.models import Message, RequestContext, TenantConfig


def _make_ctx(text: str) -> RequestContext:
    return RequestContext(
        request_id="test-req",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content=text)],
    )


# ---------------------------------------------------------------------------
# Required test cases from spec
# ---------------------------------------------------------------------------

def test_classify_code():
    assert classify_domain("Write a Python quicksort") == "code"


def test_classify_factual_qa():
    assert classify_domain("Who wrote Hamlet?") == "factual_qa"


def test_classify_summarisation():
    assert classify_domain("Summarise this article") == "summarisation"


# ---------------------------------------------------------------------------
# Context-based classification
# ---------------------------------------------------------------------------

def test_classify_ctx_code():
    ctx = _make_ctx("Implement a Python quicksort")
    assert classify_domain(ctx) == "code"


def test_classify_ctx_factual_qa():
    ctx = _make_ctx("Who wrote Hamlet?")
    assert classify_domain(ctx) == "factual_qa"


def test_classify_ctx_summarisation():
    ctx = _make_ctx("Summarise this article about climate change")
    assert classify_domain(ctx) == "summarisation"


def test_classify_ctx_unknown():
    ctx = _make_ctx("Tell me something nice")
    assert classify_domain(ctx) == "unknown"


def test_classify_empty_ctx():
    ctx = _make_ctx("")
    assert classify_domain(ctx) == "unknown"


# ---------------------------------------------------------------------------
# All domain keywords hit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Write Python code", "code"),
    ("summarize this document", "summarisation"),
    ("who is the author of the book", "factual_qa"),
    ("based on the document, what happened", "rag_retrieval"),
    ("write a poem about the moon", "creative"),
    ("prove that this theorem holds", "reasoning"),
    ("random gibberish text here", "unknown"),
])
def test_classify_parametrize(text, expected):
    assert classify_domain(text) == expected


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_domain_thresholds_all_present():
    for domain in ["factual_qa", "summarisation", "rag_retrieval",
                   "reasoning", "code", "creative", "unknown"]:
        assert domain in DOMAIN_THRESHOLDS


def test_thresholds_monotonic_order():
    """Factual QA threshold < code threshold (domain-adaptive)."""
    assert DOMAIN_THRESHOLDS["factual_qa"] < DOMAIN_THRESHOLDS["code"]


def test_get_threshold_known_domain():
    t = get_threshold("code")
    assert t == DOMAIN_THRESHOLDS["code"]


def test_get_threshold_unknown_fallback():
    t = get_threshold("nonexistent_domain")
    assert t == DOMAIN_THRESHOLDS["unknown"]


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------

def test_reload_thresholds_from_env(monkeypatch):
    import json
    overrides = {"factual_qa": 0.70}
    monkeypatch.setenv("DOMAIN_THRESHOLDS", json.dumps(overrides))
    reload_thresholds()
    assert get_threshold("factual_qa") == 0.70
    # Cleanup
    monkeypatch.delenv("DOMAIN_THRESHOLDS", raising=False)
    reload_thresholds()


# ---------------------------------------------------------------------------
# Latency < 2ms
# ---------------------------------------------------------------------------

def test_classify_latency():
    n = 100
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        classify_domain("Write a Python quicksort")
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p99 = times[int(n * 0.99) - 1]
    assert p99 < 2.0, f"Domain classify P99 too slow: {p99:.3f}ms"
