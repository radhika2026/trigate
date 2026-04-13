"""Static tier router based on complexity score."""
from __future__ import annotations

import logging

from gateway.models import RequestContext, TenantConfig

logger = logging.getLogger(__name__)

# Complexity thresholds for tier assignment
_TIER_THRESHOLDS = {
    "small": 0.35,     # complexity < 0.35 → small
    "mid": 0.65,       # 0.35 ≤ complexity < 0.65 → mid
    "frontier": 1.01,  # complexity ≥ 0.65 → frontier
}


def assign_tier(ctx: RequestContext) -> str:
    """Assign a tier based on complexity score, model_hint, and allowed_tiers.

    Rules (in priority order):
    1. model_hint override — if hint names a specific tier, use it
    2. Score-based: < 0.35 → small, < 0.65 → mid, else frontier
    3. Restrict to tenant allowed_tiers: upgrade to next allowed tier if needed
    """
    tenant_cfg: TenantConfig = ctx.tenant_config
    allowed = set(tenant_cfg.allowed_tiers)

    # --- Rule 1: model_hint override ---
    hint = ctx.model_hint.lower()
    if hint in ("small", "mid", "frontier"):
        tier = hint
        if tier in allowed:
            return tier
        # If the hinted tier is not allowed, fall through to score-based
        logger.warning(
            "model_hint=%s not in tenant allowed_tiers=%s; using score", hint, allowed
        )

    # --- Rule 2: score-based ---
    score = ctx.complexity_score
    if score < 0.35:
        tier = "small"
    elif score < 0.65:
        tier = "mid"
    else:
        tier = "frontier"

    # --- Rule 3: enforce allowed_tiers ---
    # If assigned tier is not allowed, upgrade to next allowed tier
    tier_order = ["small", "mid", "frontier"]
    idx = tier_order.index(tier)
    while tier not in allowed:
        idx += 1
        if idx >= len(tier_order):
            # Fall back to highest allowed tier
            for t in reversed(tier_order):
                if t in allowed:
                    return t
            raise RuntimeError(f"No allowed tiers for tenant {tenant_cfg.tenant_id}")
        tier = tier_order[idx]

    return tier


def route_tier(ctx: RequestContext) -> RequestContext:
    """Set ctx.chosen_tier based on complexity and tenant config."""
    ctx.chosen_tier = assign_tier(ctx)
    return ctx
