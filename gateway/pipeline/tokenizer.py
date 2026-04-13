"""tiktoken wrapper — populates ctx.token_ids and ctx.input_tokens."""
from __future__ import annotations

import logging
from functools import lru_cache

import tiktoken

from gateway.models import RequestContext

logger = logging.getLogger(__name__)

# Default encoding for models not explicitly mapped
_DEFAULT_ENCODING = "cl100k_base"

# Map model prefixes to tiktoken encoding names
_MODEL_ENCODING_MAP: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-35": "cl100k_base",
    "text-davinci": "p50k_base",
    "claude": "cl100k_base",
    "mock": "cl100k_base",
    "auto": "cl100k_base",
}


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def _resolve_encoding(model: str) -> tiktoken.Encoding:
    model_lower = model.lower()
    for prefix, enc_name in _MODEL_ENCODING_MAP.items():
        if model_lower.startswith(prefix):
            return _get_encoding(enc_name)
    return _get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, model: str = "auto") -> list[int]:
    """Return token IDs for ``text`` using the encoding appropriate for ``model``."""
    enc = _resolve_encoding(model)
    return enc.encode(text)


def tokenize_context(ctx: RequestContext) -> RequestContext:
    """Populate ctx.token_ids and ctx.input_tokens from ctx.messages.

    All message content is concatenated with a simple role-prefix format
    matching how OpenAI counts tokens (approximately).
    """
    model = ctx.model_hint if ctx.model_hint != "auto" else "auto"
    enc = _resolve_encoding(model)

    all_ids: list[int] = []
    for msg in ctx.messages:
        # Role prefix tokens (approximate ChatML format)
        prefix = f"<|im_start|>{msg.role}\n"
        suffix = "<|im_end|>\n"
        all_ids.extend(enc.encode(prefix))
        all_ids.extend(enc.encode(msg.content))
        all_ids.extend(enc.encode(suffix))

    ctx.token_ids = all_ids
    ctx.input_tokens = len(all_ids)
    return ctx
