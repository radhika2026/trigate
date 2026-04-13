"""Unit tests for openai_api.py: mocked OpenAI response, retry on 429, Anthropic routing."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.backends.openai_api import OpenAIBackend
from gateway.models import Message, RequestContext, TenantConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(messages: list[Message] | None = None) -> RequestContext:
    if messages is None:
        messages = [Message(role="user", content="Hello, world!")]
    return RequestContext(
        request_id="test-req",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=messages,
    )


def _openai_backend(model_id: str = "gpt-4o") -> OpenAIBackend:
    return OpenAIBackend(
        backend_id="openai-1",
        tier="frontier",
        model_id=model_id,
        openai_api_key="sk-test",
        anthropic_api_key="ant-test",
    )


# ---------------------------------------------------------------------------
# OpenAI non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_response():
    backend = _openai_backend()
    ctx = _make_ctx()

    mock_usage = MagicMock()
    mock_usage.completion_tokens = 10

    mock_message = MagicMock()
    mock_message.content = "Hello there!"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    mock_create = AsyncMock(return_value=mock_resp)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch.object(backend, "_openai_client", return_value=mock_client):
        result = await backend.chat(ctx)

    assert result == "Hello there!"
    assert ctx.chosen_model == "gpt-4o"


@pytest.mark.asyncio
async def test_chat_retries_on_429():
    """Should retry up to _MAX_RETRIES times on RateLimitError."""
    import openai

    backend = _openai_backend()
    ctx = _make_ctx()

    mock_usage = MagicMock()
    mock_usage.completion_tokens = 5

    mock_message = MagicMock()
    mock_message.content = "Retried successfully"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    call_count = 0

    async def _create_side_effect(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise openai.RateLimitError("rate limit", response=MagicMock(), body=None)
        return mock_resp

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create_side_effect

    with patch.object(backend, "_openai_client", return_value=mock_client):
        with patch("gateway.backends.openai_api._backoff", AsyncMock()):
            result = await backend.chat(ctx)

    assert result == "Retried successfully"
    assert call_count == 3


# ---------------------------------------------------------------------------
# Anthropic routing
# ---------------------------------------------------------------------------


def test_is_anthropic_for_claude_model():
    backend = _openai_backend("claude-3-5-sonnet-20241022")
    assert backend._is_anthropic() is True


def test_is_not_anthropic_for_gpt_model():
    backend = _openai_backend("gpt-4o")
    assert backend._is_anthropic() is False


@pytest.mark.asyncio
async def test_chat_routes_to_anthropic_for_claude():
    """When model starts with claude-, _anthropic_chat should be called."""
    backend = _openai_backend("claude-3-haiku-20240307")
    ctx = _make_ctx()

    mock_content = MagicMock()
    mock_content.text = "Claude says hi"

    mock_usage = MagicMock()
    mock_usage.output_tokens = 5

    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_resp.usage = mock_usage

    mock_create = AsyncMock(return_value=mock_resp)
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    with patch.object(backend, "_anthropic_client", return_value=mock_client):
        with patch.object(backend, "_openai_client") as mock_openai:
            result = await backend.chat(ctx)

    # OpenAI client should NOT have been used
    mock_openai.assert_not_called()
    assert result == "Claude says hi"
    assert ctx.chosen_model == "claude-3-haiku-20240307"
