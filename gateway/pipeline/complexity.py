"""Complexity scorer — v1 heuristic implementation."""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from gateway.models import RequestContext


class ComplexityScorer(ABC):
    """Abstract base class for complexity scorers."""

    @abstractmethod
    def score(self, ctx: RequestContext) -> float:
        """Return a complexity score in [0, 1]."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for metrics labels."""
        ...


# ---------------------------------------------------------------------------
# Heuristic signals
# ---------------------------------------------------------------------------

# Keywords that indicate high-complexity tasks
_HIGH_COMPLEXITY_KEYWORDS = frozenset(
    [
        "recursive", "recursion", "algorithm", "implement", "design",
        "architecture", "explain", "analyse", "analyze", "compare",
        "optimize", "optimise", "trade-off", "tradeoff", "complexity",
        "big-o", "proof", "theorem", "derive", "step by step",
        "fibonacci", "dynamic programming", "graph", "tree", "sorting",
        "machine learning", "neural network", "transformer", "attention",
        "write a", "create a", "build a", "develop a",
        "refactor", "debug", "review", "critique",
        "summarise", "summarize", "translate", "paraphrase",
    ]
)

# Keywords that indicate low-complexity tasks
_LOW_COMPLEXITY_KEYWORDS = frozenset(
    [
        "what is", "who is", "when is", "where is", "define",
        "list", "name", "give me", "tell me", "show me",
        "yes or no", "true or false", "capital of",
    ]
)

# Penalty for very short messages (simple queries)
_SHORT_MSG_THRESHOLD = 20  # characters


def _heuristic_score(text: str) -> float:
    """Compute a heuristic complexity score in [0, 1]."""
    lowered = text.lower()

    # --- Base signals ---
    word_count = len(text.split())
    char_count = len(text)

    # Length signal: longer → more complex (log scale, capped)
    length_signal = min(1.0, math.log1p(word_count) / math.log1p(200))

    # Keyword signals
    high_kw_hits = sum(1 for kw in _HIGH_COMPLEXITY_KEYWORDS if kw in lowered)
    low_kw_hits = sum(1 for kw in _LOW_COMPLEXITY_KEYWORDS if kw in lowered)

    keyword_signal = min(1.0, high_kw_hits * 0.20) - min(0.5, low_kw_hits * 0.2)

    # Code signal: backticks, indentation patterns, brackets
    code_chars = len(re.findall(r"[`{}()\[\];]", text))
    code_signal = min(0.3, code_chars * 0.02)

    # Multi-part question signal (question marks, numbered lists)
    question_count = text.count("?")
    multipart_signal = min(0.2, (question_count - 1) * 0.1) if question_count > 1 else 0.0

    # Short simple query penalty
    short_penalty = 0.3 if char_count < _SHORT_MSG_THRESHOLD else 0.0

    raw = (
        0.20 * length_signal
        + 0.60 * keyword_signal
        + 0.12 * code_signal
        + 0.08 * multipart_signal
        - short_penalty
    )

    return max(0.0, min(1.0, raw))


class HeuristicComplexityScorer(ComplexityScorer):
    """Version-1 heuristic complexity scorer."""

    @property
    def version(self) -> str:
        return "v1"

    def score(self, ctx: RequestContext) -> float:
        """Score based on the last user message (or all messages concatenated)."""
        # Use last user message for scoring; fall back to all content
        user_msgs = [m for m in ctx.messages if m.role == "user"]
        if user_msgs:
            text = user_msgs[-1].content
        else:
            text = " ".join(m.content for m in ctx.messages)

        return _heuristic_score(text)


def score_complexity(ctx: RequestContext, scorer: ComplexityScorer | None = None) -> RequestContext:
    """Score the request and set ctx.complexity_score."""
    if scorer is None:
        scorer = HeuristicComplexityScorer()
    ctx.complexity_score = scorer.score(ctx)
    return ctx
