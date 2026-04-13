"""Embedding model wrapper (M2 stub)."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")


class Embedder:
    """Thin async wrapper around sentence-transformers (loaded lazily).

    Full implementation in M2. For M1, returns zero-vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model: Any = None

    def _load_model(self) -> None:
        """Load model in worker thread (blocking)."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load embedding model %s: %s", self.model_name, exc)

    def _encode_sync(self, text: str) -> list[float]:
        if self._model is None:
            return [0.0] * self.dim
        vec: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()  # type: ignore[return-value]

    async def encode(self, text: str) -> list[float]:
        """Return normalized embedding vector for text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._encode_sync, text)
