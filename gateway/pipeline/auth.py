"""Bearer-token authentication and TenantConfig resolution."""
from __future__ import annotations

import json
import logging
import os

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gateway.models import TenantConfig

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)

# In-memory tenant store (populated from TENANT_KEYS env var)
_TENANT_STORE: dict[str, TenantConfig] | None = None


def _load_tenant_store() -> dict[str, TenantConfig]:
    """Load tenant configuration from TENANT_KEYS environment variable.

    TENANT_KEYS is a JSON string:
    {
        "sk-abc": {"tenant_id": "tenant1", "rpm_limit": 120, ...},
        "sk-xyz": {"tenant_id": "tenant2"}
    }
    A minimal entry only needs tenant_id; all other fields use defaults.
    """
    global _TENANT_STORE
    if _TENANT_STORE is not None:
        return _TENANT_STORE

    raw = os.environ.get("TENANT_KEYS", "{}")
    try:
        data: dict[str, object] = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("TENANT_KEYS is not valid JSON; using empty tenant store")
        data = {}

    store: dict[str, TenantConfig] = {}
    for api_key, cfg in data.items():
        if isinstance(cfg, dict):
            store[api_key] = TenantConfig(**cfg)
        elif isinstance(cfg, str):
            # Simple mapping: key → tenant_id string
            store[api_key] = TenantConfig(tenant_id=cfg)
        else:
            logger.warning("Skipping malformed tenant entry for key %s", api_key)

    _TENANT_STORE = store
    return store


def reset_tenant_store(new_store: dict[str, TenantConfig] | None = None) -> None:
    """Reset the in-memory store (used in tests)."""
    global _TENANT_STORE
    _TENANT_STORE = new_store


async def authenticate(request: Request) -> TenantConfig:
    """Extract Bearer token, validate, return TenantConfig or raise HTTP 401."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    store = _load_tenant_store()
    tenant_cfg = store.get(token)
    if tenant_cfg is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return tenant_cfg
