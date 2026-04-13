"""Embedding model wrapper — M2 implementation."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level executor and model — loaded ONCE at startup.
# Executor is created lazily to avoid spawning threads before torch is imported
# (torch aborts if imported after threads are already running on macOS/MPS).
_executor: ThreadPoolExecutor | None = None
_model: Any = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_DIM = 384


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embedder")
    return _executor


def _get_model() -> Any:
    """Return the loaded model (must be called after load_model())."""
    return _model


def load_model(model_name: str = _MODEL_NAME) -> None:
    """Load the sentence-transformer model (blocking, call once at startup)."""
    global _model, _MODEL_NAME
    _MODEL_NAME = model_name
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded: %s", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load embedding model %s: %s", model_name, exc)


def _encode_sync(text: str) -> list[float]:
    """Blocking encode — runs in thread pool."""
    if _model is None:
        return [0.0] * _DIM
    vec: np.ndarray = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()  # type: ignore[return-value]


def _encode_batch_sync(texts: list[str]) -> list[list[float]]:
    """Blocking batch encode — runs in thread pool."""
    if _model is None:
        return [[0.0] * _DIM for _ in texts]
    vecs: np.ndarray = _model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()  # type: ignore[return-value]


async def embed(text: str) -> list[float]:
    """Async embed a single text. Returns a (384,) normalized float vector."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_get_executor(), _encode_sync, text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Async embed a batch of texts. Returns list of (384,) normalized vectors."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_get_executor(), _encode_batch_sync, texts)


# --------------------------------------------------------------------------
# Legacy class-based API kept for backward compat
# --------------------------------------------------------------------------

class Embedder:
    """Thin async wrapper around sentence-transformers."""

    def __init__(self, model_name: str = _MODEL_NAME, dim: int = _DIM) -> None:
        self.model_name = model_name
        self.dim = dim

    def load(self) -> None:
        """Load (or re-use) the global model."""
        load_model(self.model_name)

    async def encode(self, text: str) -> list[float]:
        """Return normalized embedding vector for text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), _encode_sync, text)

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Return normalized embedding vectors for texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), _encode_batch_sync, texts)
