"""Integration test fixtures: testcontainers Redis + Postgres."""
from __future__ import annotations

import pytest
import pytest_asyncio

# These fixtures require Docker; skip if Docker is unavailable.
# In CI, docker-compose.ci.yml starts the services.

try:
    from testcontainers.redis import RedisContainer  # type: ignore[import]
    from testcontainers.postgres import PostgresContainer  # type: ignore[import]
    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False


@pytest.fixture(scope="session")
def redis_container():
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not available")
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_container():
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not available")
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def redis_url(redis_container):  # type: ignore[no-untyped-def]
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest.fixture(scope="session")
def postgres_dsn(postgres_container):  # type: ignore[no-untyped-def]
    return postgres_container.get_connection_url()
