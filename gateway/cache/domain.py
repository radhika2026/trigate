"""Domain classification and per-domain similarity thresholds."""
from __future__ import annotations

DOMAIN_THRESHOLDS: dict[str, float] = {
    "factual_qa": 0.88,
    "summarisation": 0.90,
    "rag_retrieval": 0.92,
    "reasoning": 0.94,
    "code": 0.95,
    "creative": 0.96,
    "unknown": 0.92,
}

# Simple keyword-based domain classifier
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "code": ["function", "class", "def ", "import ", "algorithm", "code", "programming", "bug", "debug", "implement"],
    "reasoning": ["prove", "derive", "explain why", "reason", "logic", "deduce", "infer"],
    "summarisation": ["summarise", "summarize", "tldr", "summary", "brief", "overview"],
    "factual_qa": ["what is", "who is", "when was", "where is", "capital", "define"],
    "creative": ["write a poem", "write a story", "creative", "fiction", "imagine"],
    "rag_retrieval": ["based on the document", "according to", "from the context", "retrieve"],
}


def classify_domain(text: str) -> str:
    """Return domain string for text. Falls back to 'unknown'."""
    lowered = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return "unknown"


def get_threshold(domain: str) -> float:
    """Return the similarity threshold for a domain."""
    return DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["unknown"])
