"""Domain classification and per-domain similarity thresholds — M2 implementation."""
from __future__ import annotations

import json
import logging
import os
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.models import RequestContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds — intentionally lower for factual/summarisation to make
# domain-adaptive caching meaningful vs a flat threshold.
# ---------------------------------------------------------------------------
DOMAIN_THRESHOLDS: dict[str, float] = {
    "factual_qa": 0.88,
    "summarisation": 0.90,
    "rag_retrieval": 0.92,
    "reasoning": 0.94,
    "code": 0.95,
    "creative": 0.96,
    "unknown": 0.92,
}

# Runtime-mutable copy (supports hot-reload via SIGHUP)
_active_thresholds: dict[str, float] = dict(DOMAIN_THRESHOLDS)


def _load_thresholds_from_env() -> dict[str, float]:
    """Read DOMAIN_THRESHOLDS override from env var (JSON string)."""
    raw = os.environ.get("DOMAIN_THRESHOLDS", "")
    if not raw:
        return dict(DOMAIN_THRESHOLDS)
    try:
        overrides: dict[str, float] = json.loads(raw)
        merged = dict(DOMAIN_THRESHOLDS)
        merged.update(overrides)
        logger.info("Loaded DOMAIN_THRESHOLDS override: %s", overrides)
        return merged
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Invalid DOMAIN_THRESHOLDS env var, using defaults: %s", exc)
        return dict(DOMAIN_THRESHOLDS)


def reload_thresholds() -> None:
    """Reload thresholds from env — called on SIGHUP."""
    global _active_thresholds
    _active_thresholds = _load_thresholds_from_env()
    logger.info("Domain thresholds reloaded: %s", _active_thresholds)


# Register SIGHUP handler (no-op on platforms that don't support it)
try:
    signal.signal(signal.SIGHUP, lambda _sig, _frame: reload_thresholds())
except (OSError, AttributeError):
    pass  # Windows / restricted envs

# Initialise from env at import time
_active_thresholds = _load_thresholds_from_env()

# ---------------------------------------------------------------------------
# Keyword-based heuristic classifier (no ML training required for M2 CI)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "code": [
        "python", "javascript", "java", "c++", "c#", "ruby", "golang", "rust",
        "function", "class", "def ", "import ", "algorithm", "code", "coding",
        "programming", "script", "debug", "bug", "implement", "quicksort",
        "sort", "binary search", "data structure", "recursion", "loop",
        "variable", "array", "list", "dict", "object", "method", "compile",
        "syntax", "library", "framework", "api",
    ],
    "summarisation": [
        "summarise", "summarize", "tldr", "tl;dr", "summary", "brief", "overview",
        "condense", "key points", "main points", "abstract", "outline",
        "summarisation", "summarization",
    ],
    "factual_qa": [
        "what is", "what are", "who is", "who was", "who wrote", "who created",
        "when was", "when did", "where is", "where was", "which", "how many",
        "capital of", "define", "meaning of", "year of", "author of",
        "invented", "discovered", "born", "died",
    ],
    "reasoning": [
        "prove", "derive", "explain why", "reason", "logic", "deduce", "infer",
        "because", "therefore", "thus", "hypothesis", "argument", "evidence",
        "justify", "analyse", "analyze",
    ],
    "creative": [
        "write a poem", "write a story", "write a song", "creative", "fiction",
        "imagine", "fantasy", "narrative", "short story", "haiku", "rhyme",
    ],
    "rag_retrieval": [
        "based on the document", "according to", "from the context",
        "from the passage", "retrieve", "in the text", "the document says",
    ],
}

# Priority order (more specific first)
_PRIORITY_ORDER = [
    "rag_retrieval",
    "code",
    "summarisation",
    "creative",
    "factual_qa",
    "reasoning",
]


def _classify_text(text: str) -> str:
    """Heuristic classification of a raw text string. < 1 ms on CPU."""
    lowered = text.lower()
    for domain in _PRIORITY_ORDER:
        keywords = _DOMAIN_KEYWORDS[domain]
        if any(kw in lowered for kw in keywords):
            return domain
    return "unknown"


def classify_domain(ctx_or_text: "RequestContext | str") -> str:
    """Classify the domain of a request context OR a plain text string.

    Accepts either a RequestContext (for pipeline use) or a str (for tests /
    benchmarks), so callers don't need to construct a full context.

    Returns one of: factual_qa, summarisation, rag_retrieval, reasoning,
    code, creative, unknown.
    """
    if isinstance(ctx_or_text, str):
        return _classify_text(ctx_or_text)

    # RequestContext path — use last user message
    ctx = ctx_or_text
    user_texts = [m.content for m in ctx.messages if m.role == "user"]
    if not user_texts:
        return "unknown"
    combined = " ".join(user_texts)
    return _classify_text(combined)


def get_threshold(domain: str) -> float:
    """Return the similarity threshold for a domain (uses hot-reloadable config)."""
    return _active_thresholds.get(domain, _active_thresholds.get("unknown", 0.92))
