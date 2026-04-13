"""Deterministic mock backend — CI backbone."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from gateway.backends.base import Backend
from gateway.models import RequestContext

logger = logging.getLogger(__name__)

# Default response template
_DEFAULT_RESPONSE = "This is a mock response from {model_id}."

# Cost per token (USD)
_MOCK_COST_PER_TOKEN = 0.000002


class MockBackend(Backend):
    """Configurable fake backend for CI/testing.

    Parameters
    ----------
    backend_id:
        Unique ID string.
    tier:
        Tier name (small | mid | frontier).
    model_id:
        Model identifier.
    ttft_ms:
        Simulated time-to-first-token in milliseconds.
    output_tokens:
        Number of tokens in the simulated response.
    cost_per_token:
        USD cost per output token.
    error:
        If set, raise this exception instead of returning a response.
    response_template:
        Format string for the response (uses {model_id}).
    """

    def __init__(
        self,
        backend_id: str = "mock-1",
        tier: str = "small",
        model_id: str = "mock-small",
        ttft_ms: float = 50.0,
        output_tokens: int = 32,
        cost_per_token: float = _MOCK_COST_PER_TOKEN,
        error: Exception | None = None,
        response_template: str = _DEFAULT_RESPONSE,
    ) -> None:
        self._backend_id = backend_id
        self._tier = tier
        self._model_id = model_id
        self._ttft_ms = ttft_ms
        self._output_tokens = output_tokens
        self._cost_per_token = cost_per_token
        self._error = error
        self._response_template = response_template
        self._queue_depth = 0

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def _build_response(self) -> str:
        return self._response_template.format(model_id=self._model_id)

    def _update_ctx(self, ctx: RequestContext, response: str) -> None:
        words = response.split()
        ctx.output_tokens = max(self._output_tokens, len(words))
        ctx.cost_usd = ctx.output_tokens * self._cost_per_token
        ctx.chosen_backend = self._backend_id
        ctx.chosen_model = self._model_id
        if ctx.chosen_tier is None:
            ctx.chosen_tier = self._tier

    async def chat_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        """Yield tokens with simulated TTFT delay."""
        if self._error is not None:
            raise self._error

        self._queue_depth += 1
        try:
            # Simulate TTFT
            await asyncio.sleep(self._ttft_ms / 1000.0)
            import time
            ctx.ttft_ms = self._ttft_ms

            response = self._build_response()
            # Yield word-by-word to simulate streaming
            words = response.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield chunk
                await asyncio.sleep(0)  # yield control

            self._update_ctx(ctx, response)
        finally:
            self._queue_depth -= 1

    async def chat(self, ctx: RequestContext) -> str:
        """Return full response after TTFT delay."""
        if self._error is not None:
            raise self._error

        self._queue_depth += 1
        try:
            await asyncio.sleep(self._ttft_ms / 1000.0)
            import time
            ctx.ttft_ms = self._ttft_ms

            response = self._build_response()
            self._update_ctx(ctx, response)
            return response
        finally:
            self._queue_depth -= 1

    async def health(self) -> bool:
        return self._error is None


def make_mock_backends(configs: list[dict]) -> list[MockBackend]:  # type: ignore[type-arg]
    """Instantiate MockBackend objects from config dicts."""
    backends = []
    for cfg in configs:
        if cfg.get("type") == "mock":
            backends.append(
                MockBackend(
                    backend_id=cfg.get("id", "mock"),
                    tier=cfg.get("tier", "small"),
                    model_id=cfg.get("model_id", "mock"),
                    ttft_ms=float(cfg.get("ttft_ms", 50.0)),
                    output_tokens=int(cfg.get("output_tokens", 32)),
                )
            )
    return backends
