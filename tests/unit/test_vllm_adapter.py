"""Unit tests for vllm.py: streaming, headers, 503 on connection refused."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from gateway.backends.vllm import VLLMBackend
from gateway.models import Message, RequestContext, TenantConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> RequestContext:
    return RequestContext(
        request_id="test-req",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content="Hello")],
    )


def _make_backend(base_url: str = "http://localhost:8001") -> VLLMBackend:
    return VLLMBackend(
        backend_id="vllm-1",
        tier="frontier",
        model_id="llama-3-8b",
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

_SSE_CHUNKS = [
    'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n',
    'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}\n',
    "data: [DONE]\n",
]


@pytest.mark.asyncio
async def test_chat_stream_returns_tokens():
    backend = _make_backend()
    ctx = _make_ctx()

    mock_resp = AsyncMock()
    mock_resp.status_code = 200

    async def fake_aiter_lines():
        for chunk in _SSE_CHUNKS:
            yield chunk.strip()

    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.aread = AsyncMock(return_value=b"")

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)

    with patch.object(backend, "_get_client", return_value=mock_client):
        chunks = []
        async for token in backend.chat_stream(ctx):
            chunks.append(token)

    assert "Hello" in chunks
    assert " world" in chunks


@pytest.mark.asyncio
async def test_chat_stream_sets_ttft():
    backend = _make_backend()
    ctx = _make_ctx()

    mock_resp = AsyncMock()
    mock_resp.status_code = 200

    async def fake_aiter_lines():
        for chunk in _SSE_CHUNKS:
            yield chunk.strip()

    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.aread = AsyncMock(return_value=b"")

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)

    with patch.object(backend, "_get_client", return_value=mock_client):
        async for _ in backend.chat_stream(ctx):
            pass

    assert ctx.ttft_ms is not None
    assert ctx.ttft_ms >= 0.0


@pytest.mark.asyncio
async def test_chat_stream_503_on_connection_refused():
    backend = _make_backend()
    ctx = _make_ctx()

    mock_client = MagicMock()
    mock_client.stream = MagicMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch.object(backend, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPException) as exc_info:
            async for _ in backend.chat_stream(ctx):
                pass

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_chat_sets_chosen_model():
    backend = _make_backend()
    ctx = _make_ctx()

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"content": "Hi!"}}],
            "usage": {"completion_tokens": 5},
        }
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch.object(backend, "_get_client", return_value=mock_client):
        await backend.chat(ctx)

    assert ctx.chosen_model == "llama-3-8b"
    assert ctx.chosen_backend == "vllm-1"
