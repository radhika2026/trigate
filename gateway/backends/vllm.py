"""vLLM OpenAI-compatible adapter."""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx
from fastapi import HTTPException

from gateway.backends.base import Backend
from gateway.models import RequestContext

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """Adapter for vLLM's OpenAI-compatible HTTP API."""

    def __init__(
        self,
        backend_id: str,
        tier: str,
        model_id: str,
        base_url: str = "http://localhost:8001",
        timeout_s: float = 120.0,
    ) -> None:
        self._backend_id = backend_id
        self._tier = tier
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s),
            )
        return self._client

    def _build_payload(self, ctx: RequestContext, stream: bool) -> dict:  # type: ignore[type-arg]
        return {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.content} for m in ctx.messages],
            "stream": stream,
            "temperature": ctx.temperature,
            "max_tokens": ctx.max_tokens,
        }

    async def chat_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        """Stream tokens from vLLM; raise 503 on connection refused."""
        client = self._get_client()
        payload = self._build_payload(ctx, stream=True)
        ttft_set = False
        start = time.monotonic()

        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"vLLM error: {body.decode()}",
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if not ttft_set:
                            ctx.ttft_ms = (time.monotonic() - start) * 1000.0
                            ttft_set = True
                        ctx.output_tokens += len(content.split())
                        yield content

        except httpx.ConnectError as exc:
            raise HTTPException(status_code=503, detail=f"vLLM unreachable: {exc}") from exc

        ctx.chosen_backend = self._backend_id
        ctx.chosen_model = self._model_id

    async def chat(self, ctx: RequestContext) -> str:
        """Non-streaming call to vLLM."""
        client = self._get_client()
        payload = self._build_payload(ctx, stream=False)
        start = time.monotonic()

        try:
            resp = await client.post("/v1/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise HTTPException(status_code=503, detail=f"vLLM unreachable: {exc}") from exc

        if resp.status_code >= 400:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"vLLM error: {resp.text}",
            )

        data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        ctx.ttft_ms = (time.monotonic() - start) * 1000.0
        ctx.output_tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
        ctx.chosen_backend = self._backend_id
        ctx.chosen_model = self._model_id
        return content

    async def health(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False
