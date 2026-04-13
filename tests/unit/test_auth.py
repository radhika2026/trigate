"""Unit tests for auth.py: valid key→200, invalid key→401, missing header→401."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.models import TenantConfig
from gateway.pipeline.auth import authenticate, reset_tenant_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_KEY = "sk-test-valid"
_INVALID_KEY = "sk-test-invalid"
_TENANT_CFG = TenantConfig(tenant_id="tenant1", rpm_limit=60)


def _mock_store() -> dict[str, TenantConfig]:
    return {_VALID_KEY: _TENANT_CFG}


# ---------------------------------------------------------------------------
# Tests via authenticate() coroutine directly (no HTTP layer needed)
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal request stub for testing authenticate()."""

    def __init__(self, auth_header: str = "") -> None:
        self.headers = {"Authorization": auth_header} if auth_header else {}


@pytest.fixture(autouse=True)
def setup_store():
    reset_tenant_store(_mock_store())
    yield
    reset_tenant_store(None)


@pytest.mark.asyncio
async def test_valid_key_returns_tenant_config():
    req = _FakeRequest(f"Bearer {_VALID_KEY}")
    cfg = await authenticate(req)  # type: ignore[arg-type]
    assert cfg.tenant_id == "tenant1"
    assert cfg.rpm_limit == 60


@pytest.mark.asyncio
async def test_invalid_key_raises_401():
    req = _FakeRequest(f"Bearer {_INVALID_KEY}")
    with pytest.raises(HTTPException) as exc_info:
        await authenticate(req)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_missing_header_raises_401():
    req = _FakeRequest("")
    with pytest.raises(HTTPException) as exc_info:
        await authenticate(req)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_malformed_header_raises_401():
    """Header without 'Bearer ' prefix."""
    req = _FakeRequest(_VALID_KEY)  # no "Bearer " prefix
    with pytest.raises(HTTPException) as exc_info:
        await authenticate(req)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_empty_bearer_token_raises_401():
    req = _FakeRequest("Bearer ")
    with pytest.raises(HTTPException) as exc_info:
        await authenticate(req)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401
