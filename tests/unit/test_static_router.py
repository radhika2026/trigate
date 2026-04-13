"""Unit tests for router/tiers.py: score→tier, model_hint override, allowed_tiers."""
from __future__ import annotations

import pytest

from gateway.models import Message, RequestContext, TenantConfig
from gateway.router.tiers import assign_tier, route_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    complexity: float,
    model_hint: str = "auto",
    allowed_tiers: list[str] | None = None,
) -> RequestContext:
    if allowed_tiers is None:
        allowed_tiers = ["small", "mid", "frontier"]
    return RequestContext(
        request_id="test",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1", allowed_tiers=allowed_tiers),
        messages=[Message(role="user", content="test")],
        complexity_score=complexity,
        model_hint=model_hint,
    )


# ---------------------------------------------------------------------------
# Score-based routing
# ---------------------------------------------------------------------------


def test_low_complexity_routes_to_small():
    ctx = _make_ctx(complexity=0.2)
    assert assign_tier(ctx) == "small"


def test_mid_complexity_routes_to_mid():
    ctx = _make_ctx(complexity=0.5)
    assert assign_tier(ctx) == "mid"


def test_high_complexity_routes_to_frontier():
    ctx = _make_ctx(complexity=0.8)
    assert assign_tier(ctx) == "frontier"


def test_boundary_below_0_35_is_small():
    ctx = _make_ctx(complexity=0.34)
    assert assign_tier(ctx) == "small"


def test_boundary_at_0_35_is_mid():
    ctx = _make_ctx(complexity=0.35)
    assert assign_tier(ctx) == "mid"


def test_boundary_at_0_65_is_frontier():
    ctx = _make_ctx(complexity=0.65)
    assert assign_tier(ctx) == "frontier"


# ---------------------------------------------------------------------------
# model_hint override
# ---------------------------------------------------------------------------


def test_model_hint_small_overrides_score():
    ctx = _make_ctx(complexity=0.9, model_hint="small")
    assert assign_tier(ctx) == "small"


def test_model_hint_frontier_overrides_score():
    ctx = _make_ctx(complexity=0.1, model_hint="frontier")
    assert assign_tier(ctx) == "frontier"


def test_model_hint_mid_overrides_score():
    ctx = _make_ctx(complexity=0.9, model_hint="mid")
    assert assign_tier(ctx) == "mid"


# ---------------------------------------------------------------------------
# allowed_tiers enforcement
# ---------------------------------------------------------------------------


def test_frontier_excluded_when_not_in_allowed_tiers():
    """If frontier is not allowed, high-complexity query should fall to mid."""
    ctx = _make_ctx(complexity=0.8, allowed_tiers=["small", "mid"])
    tier = assign_tier(ctx)
    assert tier == "mid"
    assert tier != "frontier"


def test_only_frontier_allowed_always_returns_frontier():
    ctx = _make_ctx(complexity=0.1, allowed_tiers=["frontier"])
    assert assign_tier(ctx) == "frontier"


def test_only_small_allowed_always_returns_small():
    ctx = _make_ctx(complexity=0.9, allowed_tiers=["small"])
    assert assign_tier(ctx) == "small"


# ---------------------------------------------------------------------------
# route_tier sets ctx.chosen_tier
# ---------------------------------------------------------------------------


def test_route_tier_sets_chosen_tier():
    ctx = _make_ctx(complexity=0.2)
    result = route_tier(ctx)
    assert result.chosen_tier == "small"
