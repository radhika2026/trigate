"""Unit tests for tokenizer.py: counts match tiktoken; ctx.token_ids populated."""
from __future__ import annotations

import tiktoken
import pytest

from gateway.models import Message, RequestContext, TenantConfig
from gateway.pipeline.tokenizer import count_tokens, tokenize_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(messages: list[Message]) -> RequestContext:
    return RequestContext(
        request_id="test-req",
        tenant_id="tenant1",
        tenant_config=TenantConfig(tenant_id="tenant1"),
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_count_tokens_returns_list_of_ints():
    ids = count_tokens("Hello, world!", model="auto")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(t, int) for t in ids)


def test_count_tokens_matches_tiktoken():
    text = "The quick brown fox jumps over the lazy dog."
    enc = tiktoken.get_encoding("cl100k_base")
    expected = enc.encode(text)
    got = count_tokens(text, model="auto")
    assert got == expected


def test_tokenize_context_sets_token_ids():
    ctx = _make_ctx([Message(role="user", content="What is the capital of France?")])
    tokenize_context(ctx)
    assert len(ctx.token_ids) > 0
    assert all(isinstance(t, int) for t in ctx.token_ids)


def test_tokenize_context_sets_input_tokens():
    ctx = _make_ctx([Message(role="user", content="What is the capital of France?")])
    tokenize_context(ctx)
    assert ctx.input_tokens == len(ctx.token_ids)
    assert ctx.input_tokens > 0


def test_tokenize_context_empty_messages():
    ctx = _make_ctx([])
    tokenize_context(ctx)
    assert ctx.token_ids == []
    assert ctx.input_tokens == 0


def test_tokenize_context_multi_turn():
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi! How can I help?"),
        Message(role="user", content="Tell me about Python."),
    ]
    ctx = _make_ctx(messages)
    tokenize_context(ctx)
    assert ctx.input_tokens > 20  # sanity: multi-turn should be longer
    assert ctx.input_tokens == len(ctx.token_ids)


def test_tokenize_context_longer_input_has_more_tokens():
    short_ctx = _make_ctx([Message(role="user", content="Hi")])
    long_ctx = _make_ctx([Message(role="user", content="Hi " * 100)])

    tokenize_context(short_ctx)
    tokenize_context(long_ctx)

    assert long_ctx.input_tokens > short_ctx.input_tokens
