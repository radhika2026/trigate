"""Configuration via pydantic-settings (env vars)."""
from __future__ import annotations

import json
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Postgres ---
    postgres_dsn: str = "postgresql://trigate:trigate@localhost:5432/trigate"

    # --- Auth ---
    # JSON string mapping API key → tenant_id, e.g. '{"sk-abc": "tenant1"}'
    tenant_keys: str = "{}"

    # --- Backends ---
    # JSON list of backend configs
    backends_config: str = (
        '[{"id":"mock-1","type":"mock","tier":"small","model_id":"mock-small"},'
        '{"id":"mock-2","type":"mock","tier":"mid","model_id":"mock-mid"},'
        '{"id":"mock-3","type":"mock","tier":"frontier","model_id":"mock-frontier"}]'
    )

    # --- vLLM ---
    vllm_base_url: str = "http://localhost:8001"

    # --- OpenAI ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Observability ---
    otel_exporter_otlp_endpoint: str = ""
    metrics_prefix: str = "gateway"

    # --- Embedding model ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Complexity ---
    complexity_scorer_version: str = "v1"

    # --- Thread pool ---
    executor_max_workers: int = 8

    # --- Domain-adaptive cache thresholds ---
    # JSON string overriding per-domain similarity thresholds, e.g.
    # '{"factual_qa": 0.85, "code": 0.93}'
    domain_thresholds: str = "{}"

    @field_validator("domain_thresholds")
    @classmethod
    def validate_domain_thresholds(cls, v: str) -> str:
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("domain_thresholds must be a JSON object")
        except json.JSONDecodeError as exc:
            raise ValueError(f"domain_thresholds is not valid JSON: {exc}") from exc
        return v

    def get_domain_thresholds(self) -> dict[str, float]:
        """Return domain threshold overrides (empty dict = use defaults)."""
        return json.loads(self.domain_thresholds)  # type: ignore[no-any-return]

    @field_validator("tenant_keys")
    @classmethod
    def validate_tenant_keys(cls, v: str) -> str:
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("tenant_keys must be a JSON object")
        except json.JSONDecodeError as exc:
            raise ValueError(f"tenant_keys is not valid JSON: {exc}") from exc
        return v

    @field_validator("backends_config")
    @classmethod
    def validate_backends_config(cls, v: str) -> str:
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("backends_config must be a JSON array")
        except json.JSONDecodeError as exc:
            raise ValueError(f"backends_config is not valid JSON: {exc}") from exc
        return v

    def get_tenant_keys(self) -> dict[str, str]:
        """Return mapping of API key → tenant_id."""
        return json.loads(self.tenant_keys)  # type: ignore[no-any-return]

    def get_backends_config(self) -> list[dict[str, Any]]:
        """Return list of backend config dicts."""
        return json.loads(self.backends_config)  # type: ignore[no-any-return]


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
