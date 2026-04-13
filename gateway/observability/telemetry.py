"""Telemetry record emission — one JSON-lines record per request."""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone

from gateway.models import RequestContext, TelemetryRecord

logger = logging.getLogger(__name__)

# File-like sink; override in tests
_sink = sys.stdout


def set_sink(sink: object) -> None:
    """Override the telemetry sink (for testing)."""
    global _sink
    _sink = sink  # type: ignore[assignment]


def reset_sink() -> None:
    """Reset sink to stdout."""
    global _sink
    _sink = sys.stdout


def emit(ctx: RequestContext, total_ms: float, quality_score: float | None = None) -> TelemetryRecord:
    """Build and emit a TelemetryRecord as a JSON line.

    Returns the record (useful for testing).
    """
    tier = ctx.chosen_tier or "unknown"
    model = ctx.chosen_model or "unknown"
    ttft = ctx.ttft_ms or 0.0
    throughput = (ctx.output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    record = TelemetryRecord(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        timestamp=datetime.now(tz=timezone.utc),
        input_tokens=ctx.input_tokens,
        output_tokens=ctx.output_tokens,
        complexity_score=ctx.complexity_score,
        domain=ctx.domain,
        message_turns=len(ctx.messages),
        cache_hit=ctx.cache_hit,
        kv_hit=ctx.kv_hit,
        chosen_tier=tier,
        chosen_model=model,
        ttft_ms=ttft,
        total_ms=total_ms,
        throughput_tps=throughput,
        cost_usd=ctx.cost_usd,
        quality_score=quality_score,
        judge_model=None,
        bandit_arm=None,
        bandit_reward=None,
    )

    line = record.model_dump_json()
    try:
        print(line, file=_sink, flush=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to emit telemetry: %s", exc)

    return record
