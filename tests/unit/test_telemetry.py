"""Unit tests for observability/telemetry.py: one record per request, all fields present."""
from __future__ import annotations

import io
import json

import pytest

from gateway.models import Message, RequestContext, TenantConfig, TelemetryRecord
from gateway.observability import telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**kwargs) -> RequestContext:  # type: ignore[no-untyped-def]
    defaults = dict(
        request_id="req-abc-123",
        tenant_id="tenant1",
        tenant_config=TenantConfig(tenant_id="tenant1"),
        messages=[Message(role="user", content="Hello")],
        input_tokens=10,
        output_tokens=20,
        complexity_score=0.5,
        domain="factual_qa",
        chosen_tier="small",
        chosen_model="mock-small",
        ttft_ms=45.0,
        cost_usd=0.001,
    )
    defaults.update(kwargs)
    return RequestContext(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_writes_one_json_line():
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        ctx = _make_ctx()
        telemetry.emit(ctx, total_ms=200.0)
        output = buf.getvalue()
    finally:
        telemetry.reset_sink()

    lines = [l for l in output.strip().splitlines() if l]
    assert len(lines) == 1, f"Expected 1 line, got {len(lines)}: {lines}"


def test_emit_returns_telemetry_record():
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        ctx = _make_ctx()
        record = telemetry.emit(ctx, total_ms=200.0)
    finally:
        telemetry.reset_sink()

    assert isinstance(record, TelemetryRecord)


def test_emit_all_fields_present():
    """All TelemetryRecord fields must appear in emitted JSON."""
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        ctx = _make_ctx()
        telemetry.emit(ctx, total_ms=200.0)
        output = buf.getvalue().strip()
    finally:
        telemetry.reset_sink()

    data = json.loads(output)

    required_fields = [
        "request_id",
        "tenant_id",
        "timestamp",
        "input_tokens",
        "output_tokens",
        "complexity_score",
        "domain",
        "message_turns",
        "cache_hit",
        "kv_hit",
        "chosen_tier",
        "chosen_model",
        "ttft_ms",
        "total_ms",
        "throughput_tps",
        "cost_usd",
        "quality_score",
        "judge_model",
        "bandit_arm",
        "bandit_reward",
    ]
    for field in required_fields:
        assert field in data, f"Field {field!r} missing from telemetry record"


def test_emit_correct_values():
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        ctx = _make_ctx(
            request_id="my-request",
            tenant_id="my-tenant",
            input_tokens=50,
            output_tokens=100,
            complexity_score=0.75,
            domain="code",
            chosen_tier="frontier",
            chosen_model="gpt-4o",
            ttft_ms=123.0,
            cost_usd=0.05,
        )
        telemetry.emit(ctx, total_ms=500.0)
        output = buf.getvalue().strip()
    finally:
        telemetry.reset_sink()

    data = json.loads(output)

    assert data["request_id"] == "my-request"
    assert data["tenant_id"] == "my-tenant"
    assert data["input_tokens"] == 50
    assert data["output_tokens"] == 100
    assert data["complexity_score"] == 0.75
    assert data["domain"] == "code"
    assert data["chosen_tier"] == "frontier"
    assert data["chosen_model"] == "gpt-4o"
    assert data["ttft_ms"] == 123.0
    assert data["total_ms"] == 500.0
    assert abs(data["cost_usd"] - 0.05) < 1e-6


def test_emit_multiple_requests_emits_multiple_lines():
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        for i in range(5):
            ctx = _make_ctx(request_id=f"req-{i}")
            telemetry.emit(ctx, total_ms=float(i * 100))
        output = buf.getvalue()
    finally:
        telemetry.reset_sink()

    lines = [l for l in output.strip().splitlines() if l]
    assert len(lines) == 5


def test_emit_message_turns_count():
    buf = io.StringIO()
    telemetry.set_sink(buf)
    try:
        ctx = _make_ctx()
        ctx.messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="Bye"),
        ]
        telemetry.emit(ctx, total_ms=100.0)
        output = buf.getvalue().strip()
    finally:
        telemetry.reset_sink()

    data = json.loads(output)
    assert data["message_turns"] == 3
