"""OpenTelemetry tracer setup."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_tracer = None


def get_tracer(name: str = "trigate") -> object:
    """Return an OTel tracer. Falls back to no-op if OTel is not configured."""
    global _tracer
    if _tracer is not None:
        return _tracer

    try:
        from opentelemetry import trace  # type: ignore[import]
        _tracer = trace.get_tracer(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenTelemetry tracer not available: %s", exc)
        _tracer = _NoOpTracer()

    return _tracer


class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: object) -> _NoOpSpan:
        return _NoOpSpan()
