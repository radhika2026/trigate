"""OpenAI + Anthropic API adapter."""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

from fastapi import HTTPException

from gateway.backends.base import Backend
from gateway.models import RequestContext

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {429}


class OpenAIBackend(Backend):
    """Adapter for OpenAI and Anthropic APIs.

    Routes to Anthropic SDK when model starts with 'claude-'.
    """

    def __init__(
        self,
        backend_id: str,
        tier: str,
        model_id: str,
        openai_api_key: str = "",
        anthropic_api_key: str = "",
    ) -> None:
        self._backend_id = backend_id
        self._tier = tier
        self._model_id = model_id
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def model_id(self) -> str:
        return self._model_id

    def _is_anthropic(self) -> bool:
        return self._model_id.startswith("claude-")

    def _openai_client(self) -> "openai.AsyncOpenAI":  # type: ignore[name-defined]  # noqa: F821
        import openai  # type: ignore[import]
        return openai.AsyncOpenAI(api_key=self._openai_api_key)

    def _anthropic_client(self) -> "anthropic.AsyncAnthropic":  # type: ignore[name-defined]  # noqa: F821
        import anthropic  # type: ignore[import]
        return anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)

    async def chat_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        """Stream tokens; retry on 429."""
        if self._is_anthropic():
            async for chunk in self._anthropic_stream(ctx):
                yield chunk
        else:
            async for chunk in self._openai_stream(ctx):
                yield chunk

    async def _openai_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        import openai  # type: ignore[import]
        client = self._openai_client()
        messages = [{"role": m.role, "content": m.content} for m in ctx.messages]
        start = time.monotonic()
        ttft_set = False

        for attempt in range(_MAX_RETRIES):
            try:
                stream = await client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,  # type: ignore[arg-type]
                    stream=True,
                    temperature=ctx.temperature,
                    max_tokens=ctx.max_tokens,
                )
                async for chunk in stream:
                    delta_content = chunk.choices[0].delta.content or ""
                    if delta_content:
                        if not ttft_set:
                            ctx.ttft_ms = (time.monotonic() - start) * 1000.0
                            ttft_set = True
                        ctx.output_tokens += 1
                        yield delta_content
                ctx.chosen_backend = self._backend_id
                ctx.chosen_model = self._model_id
                return
            except openai.RateLimitError:
                if attempt < _MAX_RETRIES - 1:
                    await _backoff(attempt)
                else:
                    raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded")
            except openai.APIConnectionError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def _anthropic_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        import anthropic  # type: ignore[import]
        client = self._anthropic_client()
        messages = [{"role": m.role, "content": m.content} for m in ctx.messages if m.role != "system"]
        system = next((m.content for m in ctx.messages if m.role == "system"), "")
        start = time.monotonic()
        ttft_set = False

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=self._model_id,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=ctx.max_tokens,
                )
                if system:
                    kwargs["system"] = system

                async with client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        if not ttft_set:
                            ctx.ttft_ms = (time.monotonic() - start) * 1000.0
                            ttft_set = True
                        ctx.output_tokens += 1
                        yield text
                ctx.chosen_backend = self._backend_id
                ctx.chosen_model = self._model_id
                return
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES - 1:
                    await _backoff(attempt)
                else:
                    raise HTTPException(status_code=429, detail="Anthropic rate limit exceeded")
            except anthropic.APIConnectionError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def chat(self, ctx: RequestContext) -> str:
        """Non-streaming chat completion."""
        if self._is_anthropic():
            return await self._anthropic_chat(ctx)
        return await self._openai_chat(ctx)

    async def _openai_chat(self, ctx: RequestContext) -> str:
        import openai  # type: ignore[import]
        client = self._openai_client()
        messages = [{"role": m.role, "content": m.content} for m in ctx.messages]
        start = time.monotonic()

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,  # type: ignore[arg-type]
                    stream=False,
                    temperature=ctx.temperature,
                    max_tokens=ctx.max_tokens,
                )
                content = resp.choices[0].message.content or ""
                ctx.ttft_ms = (time.monotonic() - start) * 1000.0
                ctx.output_tokens = resp.usage.completion_tokens if resp.usage else len(content.split())
                ctx.chosen_backend = self._backend_id
                ctx.chosen_model = self._model_id
                return content
            except openai.RateLimitError:
                if attempt < _MAX_RETRIES - 1:
                    await _backoff(attempt)
                else:
                    raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded")
            except openai.APIConnectionError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ""  # unreachable

    async def _anthropic_chat(self, ctx: RequestContext) -> str:
        import anthropic  # type: ignore[import]
        client = self._anthropic_client()
        messages = [{"role": m.role, "content": m.content} for m in ctx.messages if m.role != "system"]
        system = next((m.content for m in ctx.messages if m.role == "system"), "")
        start = time.monotonic()

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=self._model_id,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=ctx.max_tokens,
                )
                if system:
                    kwargs["system"] = system

                resp = await client.messages.create(**kwargs)
                content = resp.content[0].text if resp.content else ""
                ctx.ttft_ms = (time.monotonic() - start) * 1000.0
                ctx.output_tokens = resp.usage.output_tokens if resp.usage else len(content.split())
                ctx.chosen_backend = self._backend_id
                ctx.chosen_model = self._model_id
                return content
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES - 1:
                    await _backoff(attempt)
                else:
                    raise HTTPException(status_code=429, detail="Anthropic rate limit exceeded")
            except anthropic.APIConnectionError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ""  # unreachable


async def _backoff(attempt: int) -> None:
    """Exponential backoff: 1s, 2s, 4s…"""
    import asyncio
    await asyncio.sleep(2 ** attempt)
