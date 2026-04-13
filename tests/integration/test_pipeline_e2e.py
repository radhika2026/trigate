"""Integration test: full pipeline e2e with mock backend."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from gateway.models import TenantConfig
from gateway.pipeline.auth import reset_tenant_store


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):  # type: ignore[no-untyped-def]
    """Set up a test tenant and mock backends."""
    import json

    monkeypatch.setenv(
        "TENANT_KEYS",
        json.dumps({"sk-integration-test": {"tenant_id": "integration-tenant", "rpm_limit": 100}}),
    )
    monkeypatch.setenv(
        "BACKENDS_CONFIG",
        json.dumps(
            [
                {"id": "mock-small", "type": "mock", "tier": "small", "model_id": "mock-small"},
                {"id": "mock-mid", "type": "mock", "tier": "mid", "model_id": "mock-mid"},
                {"id": "mock-frontier", "type": "mock", "tier": "frontier", "model_id": "mock-frontier"},
            ]
        ),
    )

    # Reset auth store so it picks up new env
    reset_tenant_store(None)

    # Reset settings singleton
    import gateway.config as cfg_module
    cfg_module._settings = None

    yield

    # Cleanup
    reset_tenant_store(None)
    cfg_module._settings = None


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    from gateway.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):  # type: ignore[no-untyped-def]
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "ts" in data


def test_ready_endpoint(client):  # type: ignore[no-untyped-def]
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert isinstance(data["backends"], list)


def test_chat_completions_with_valid_key(client):  # type: ignore[no-untyped-def]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer sk-integration-test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_with_invalid_key(client):  # type: ignore[no-untyped-def]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer sk-invalid"},
    )
    assert resp.status_code == 401


def test_chat_completions_missing_auth(client):  # type: ignore[no-untyped-def]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]},
    )
    assert resp.status_code == 401


def test_response_headers_present(client):  # type: ignore[no-untyped-def]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer sk-integration-test"},
    )
    assert resp.status_code == 200
    assert "x-gateway-cache" in resp.headers
    assert "x-gateway-kv-hit" in resp.headers
    assert "x-gateway-request-id" in resp.headers
    assert "x-gateway-tier" in resp.headers


def test_metrics_endpoint(client):  # type: ignore[no-untyped-def]
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "gateway_requests_total" in resp.text or "gateway_" in resp.text


def test_admin_cache_flush(client):  # type: ignore[no-untyped-def]
    resp = client.post("/admin/cache/flush")
    assert resp.status_code == 200
    assert resp.json()["flushed"] == 0
