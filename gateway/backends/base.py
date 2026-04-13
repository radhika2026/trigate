"""Abstract Backend protocol and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from gateway.models import RequestContext


class Backend(ABC):
    """Abstract base class for LLM backend adapters."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Unique identifier for this backend instance."""
        ...

    @property
    @abstractmethod
    def tier(self) -> str:
        """Tier: small | mid | frontier."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string (e.g. 'gpt-4o', 'mock-small')."""
        ...

    @abstractmethod
    async def chat_stream(self, ctx: RequestContext) -> AsyncIterator[str]:
        """Yield token chunks as they arrive (streaming)."""
        ...

    @abstractmethod
    async def chat(self, ctx: RequestContext) -> str:
        """Return full response string (non-streaming)."""
        ...

    @property
    def queue_depth(self) -> int:
        """Current queue depth for observability (default 0)."""
        return 0

    async def health(self) -> bool:
        """Return True if backend is reachable."""
        return True
