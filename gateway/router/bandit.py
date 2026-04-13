"""LinUCB bandit router (M4 stub)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    arm_id: int
    tier: str
    model_id: str
    pulls: int = 0
    total_reward: float = 0.0


@dataclass
class BanditState:
    arms: list[BanditArm] = field(default_factory=list)
    alpha: float = 1.0  # exploration parameter


class LinUCBBandit:
    """LinUCB contextual bandit for model selection.

    Full implementation in M4. M1 stub: always selects arm 0.
    """

    def __init__(self, arms: list[BanditArm], feature_dim: int = 4, alpha: float = 1.0) -> None:
        self.arms = arms
        self.alpha = alpha
        self.feature_dim = feature_dim
        d = feature_dim
        self._A = [np.eye(d) for _ in arms]
        self._b = [np.zeros(d) for _ in arms]

    def select_arm(self, features: np.ndarray) -> int:
        """Select arm index (M1 stub: return 0)."""
        return 0

    def update(self, arm_idx: int, features: np.ndarray, reward: float) -> None:
        """Update arm parameters (M1 stub: track reward only)."""
        if 0 <= arm_idx < len(self.arms):
            self.arms[arm_idx].pulls += 1
            self.arms[arm_idx].total_reward += reward

    def get_state(self) -> BanditState:
        return BanditState(arms=list(self.arms), alpha=self.alpha)
