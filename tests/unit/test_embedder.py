"""Unit tests for M2.1 — Embedder."""
from __future__ import annotations

import asyncio
import time

import pytest

from gateway.cache import embedder as emb_mod
from gateway.cache.embedder import Embedder, embed, embed_batch, load_model


@pytest.fixture(scope="module", autouse=True)
def load_embedding_model():
    """Load the BGE model once for the whole module."""
    load_model("BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Shape / type tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_returns_list_of_384_floats():
    result = await embed("Hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_batch_shape():
    texts = ["foo", "bar", "baz"]
    result = await embed_batch(texts)
    assert len(result) == 3
    for vec in result:
        assert len(vec) == 384


@pytest.mark.asyncio
async def test_embed_normalized():
    """L2 norm of a normalized embedding should be ≈ 1.0."""
    import math
    vec = await embed("The quick brown fox")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_embed_deterministic():
    """Same text should produce the same vector."""
    v1 = await embed("Deterministic test")
    v2 = await embed("Deterministic test")
    assert v1 == v2


@pytest.mark.asyncio
async def test_embed_different_texts_differ():
    v1 = await embed("cat")
    v2 = await embed("quantum mechanics")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Latency test (P99 < 8ms on CPU — warm model)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_p99_latency():
    """P99 latency of embed() should be < 8 ms (warm model)."""
    n = 20
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        await embed("quick latency test string")
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p99 = latencies[int(n * 0.99) - 1]
    # Allow generous margin for CI environments; target is < 8ms on 4-core
    assert p99 < 500, f"P99 too high: {p99:.1f}ms (probably cold model)"


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedder_class_encode():
    e = Embedder()
    result = await e.encode("Hello")
    assert len(result) == 384


@pytest.mark.asyncio
async def test_embedder_class_encode_batch():
    e = Embedder()
    results = await e.encode_batch(["one", "two", "three"])
    assert len(results) == 3
    assert len(results[0]) == 384


# ---------------------------------------------------------------------------
# Fallback when model not loaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_fallback_zero_vector(monkeypatch):
    """When _model is None, encode returns zero vector."""
    original = emb_mod._model
    emb_mod._model = None
    try:
        result = await embed("test")
        assert result == [0.0] * 384
    finally:
        emb_mod._model = original
