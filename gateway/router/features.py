"""Feature vector construction for bandit (M4 stub)."""
from __future__ import annotations

import numpy as np

from gateway.models import RequestContext


def build_feature_vector(ctx: RequestContext) -> np.ndarray:
    """Build a feature vector for the bandit router.

    Full implementation in M4.
    """
    # M1 stub: return a simple 4-dim feature vector
    return np.array(
        [
            ctx.complexity_score,
            float(ctx.input_tokens) / 2048.0,
            float(ctx.cache_hit),
            float(ctx.kv_hit),
        ],
        dtype=np.float32,
    )
