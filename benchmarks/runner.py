"""Benchmark runner (M5 stub)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark scenarios (full implementation in M5)."""

    def __init__(self, gateway_url: str = "http://localhost:8000") -> None:
        self.gateway_url = gateway_url
        self.results: list[dict[str, Any]] = []

    async def run_scenario(self, scenario: Any) -> dict[str, Any]:
        """Run a single benchmark scenario (M5 stub)."""
        logger.info("Running scenario: %s", scenario)
        return {}

    async def run_all(self) -> list[dict[str, Any]]:
        """Run all registered scenarios (M5 stub)."""
        return []
