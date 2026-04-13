"""Reward computation for the bandit router (M4 stub)."""
from __future__ import annotations

from gateway.models import RequestContext, TenantConfig


def compute_reward(ctx: RequestContext) -> float:
    """Compute a scalar reward in [0, 1] for a completed request.

    Balances TTFT SLO satisfaction, cost efficiency, and quality.
    Full implementation in M4.
    """
    cfg: TenantConfig = ctx.tenant_config

    # TTFT reward: 1 if within SLO, decays otherwise
    ttft = ctx.ttft_ms or 0.0
    slo = cfg.default_ttft_slo_ms
    ttft_reward = max(0.0, 1.0 - max(0.0, ttft - slo) / slo)

    # Cost reward: 1 if free, 0 at max_cost
    cost_reward = max(0.0, 1.0 - ctx.cost_usd / max(cfg.max_cost_per_req, 1e-9))

    # Blend: (1-lambda)*ttft + lambda*cost
    lam = cfg.cost_lambda
    return (1.0 - lam) * ttft_reward + lam * cost_reward
