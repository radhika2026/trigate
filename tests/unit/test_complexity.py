"""Unit tests for complexity.py: score thresholds and ABC interface."""
from __future__ import annotations

import pytest

from gateway.models import Message, RequestContext, TenantConfig
from gateway.pipeline.complexity import (
    ComplexityScorer,
    HeuristicComplexityScorer,
    score_complexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(text: str) -> RequestContext:
    return RequestContext(
        request_id="test",
        tenant_id="t1",
        tenant_config=TenantConfig(tenant_id="t1"),
        messages=[Message(role="user", content=text)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_math_scores_below_0_3():
    ctx = _make_ctx("What is 2+2?")
    scorer = HeuristicComplexityScorer()
    score = scorer.score(ctx)
    assert score < 0.3, f"Expected < 0.3 for simple math, got {score}"


def test_coding_question_scores_above_0_6():
    ctx = _make_ctx(
        "Write a recursive Fibonacci function and explain time complexity"
    )
    scorer = HeuristicComplexityScorer()
    score = scorer.score(ctx)
    assert score > 0.6, f"Expected > 0.6 for coding question, got {score}"


def test_score_in_range():
    ctx = _make_ctx("Tell me about Napoleon.")
    scorer = HeuristicComplexityScorer()
    score = scorer.score(ctx)
    assert 0.0 <= score <= 1.0


def test_longer_question_higher_score():
    short_ctx = _make_ctx("Hi")
    long_ctx = _make_ctx(
        "Implement a graph traversal algorithm using depth-first search "
        "and breadth-first search in Python, then compare their time and "
        "space complexity with detailed step-by-step analysis."
    )
    scorer = HeuristicComplexityScorer()
    assert scorer.score(long_ctx) > scorer.score(short_ctx)


def test_scorer_has_version_v1():
    scorer = HeuristicComplexityScorer()
    assert scorer.version == "v1"


def test_score_complexity_updates_ctx():
    ctx = _make_ctx("What is 2+2?")
    result = score_complexity(ctx)
    assert result.complexity_score < 0.3


def test_scorer_abc_interface():
    """HeuristicComplexityScorer must implement ComplexityScorer ABC."""
    scorer = HeuristicComplexityScorer()
    assert isinstance(scorer, ComplexityScorer)


def test_custom_scorer_via_score_complexity():
    class ConstantScorer(ComplexityScorer):
        @property
        def version(self) -> str:
            return "const"

        def score(self, ctx: RequestContext) -> float:
            return 0.42

    ctx = _make_ctx("anything")
    result = score_complexity(ctx, scorer=ConstantScorer())
    assert result.complexity_score == 0.42
