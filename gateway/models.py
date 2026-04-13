"""Data schemas for TriGate gateway."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class GatewayHints(BaseModel):
    """Optional caller hints to influence routing."""
    prefer_tier: str | None = None
    prefer_model: str | None = None
    disable_cache: bool = False
    disable_kv: bool = False
    max_cost_usd: float | None = None


class TenantConfig(BaseModel):
    tenant_id: str
    rpm_limit: int = 60
    tpd_limit: int = 1_000_000
    max_cost_per_req: float = 0.05
    default_ttft_slo_ms: int = 2000
    cost_lambda: float = 0.3
    cache_ttl_s: int = 86400
    cache_threshold: float = 0.92
    allowed_tiers: list[str] = ["small", "mid", "frontier"]


class RequestContext(BaseModel):
    request_id: str
    tenant_id: str
    tenant_config: TenantConfig
    messages: list[Message]
    model_hint: str = "auto"
    stream: bool = True
    temperature: float = 0.7
    max_tokens: int = 1024
    gateway_hints: GatewayHints | None = None
    input_tokens: int = 0
    token_ids: list[int] = Field(default_factory=list)
    complexity_score: float = 0.5
    query_embedding: list[float] = Field(default_factory=list)
    domain: str = "unknown"
    cache_hit: bool = False
    kv_hit: bool = False
    chosen_tier: str | None = None
    chosen_backend: str | None = None
    chosen_model: str | None = None
    start_ts: float = Field(default_factory=time.monotonic)
    ttft_ms: float | None = None
    output_tokens: int = 0
    cost_usd: float = 0.0


class GatewayRequest(BaseModel):
    """OpenAI-compatible chat completions request body."""
    model: str = "auto"
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 1024
    gateway_hints: GatewayHints | None = None
    # Extra OpenAI fields (passed through)
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    user: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    id: str
    embedding: list[float]
    response: str
    model_id: str
    quality: float | None
    hit_count: int = 0
    created_at: datetime
    expires_at: datetime


class TelemetryRecord(BaseModel):
    request_id: str
    tenant_id: str
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    complexity_score: float
    domain: str
    message_turns: int
    cache_hit: bool
    kv_hit: bool
    chosen_tier: str
    chosen_model: str
    ttft_ms: float
    total_ms: float
    throughput_tps: float
    cost_usd: float
    quality_score: float | None
    judge_model: str | None
    bandit_arm: int | None
    bandit_reward: float | None
