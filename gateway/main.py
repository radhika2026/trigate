"""TriGate — FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gateway.backends.base import Backend
from gateway.backends.mock import make_mock_backends
from gateway.config import get_settings
from gateway.models import GatewayRequest, RequestContext, TenantConfig
from gateway.observability import metrics as m
from gateway.observability import telemetry
from gateway.pipeline.auth import authenticate
from gateway.pipeline.complexity import HeuristicComplexityScorer, score_complexity
from gateway.pipeline.tokenizer import tokenize_context
from gateway.router.tiers import route_tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(title="TriGate", version="0.1.0")

# Global state
_thread_pool: ThreadPoolExecutor | None = None
_backends: list[Backend] = []
_scorer = HeuristicComplexityScorer()


@app.on_event("startup")
async def startup() -> None:
    global _thread_pool, _backends
    settings = get_settings()
    _thread_pool = ThreadPoolExecutor(max_workers=settings.executor_max_workers)

    # Load backends from config
    backend_cfgs = settings.get_backends_config()
    _backends = make_mock_backends([c for c in backend_cfgs if c.get("type") == "mock"])

    # For non-mock backends, we'd initialise vLLM / OpenAI adapters here (M2+)
    logger.info("TriGate started with %d backends", len(_backends))


@app.on_event("shutdown")
async def shutdown() -> None:
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Health / ready / metrics
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": time.time()})


@app.get("/ready")
async def ready() -> JSONResponse:
    backend_ids = [b.backend_id for b in _backends]
    return JSONResponse({"status": "ready", "backends": backend_ids, "cache": "ok"})


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@app.post("/admin/cache/flush")
async def flush_cache() -> JSONResponse:
    # M2: will call cache store flush
    return JSONResponse({"flushed": 0})


@app.get("/admin/bandit/state")
async def bandit_state() -> JSONResponse:
    # M4: will return real bandit state
    return JSONResponse({"arms": []})


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

_GATEWAY_HEADERS_MISS = {
    "X-Gateway-Cache": "MISS",
    "X-Gateway-KV-Hit": "false",
}


def _build_response_headers(ctx: RequestContext) -> dict[str, str]:
    return {
        "X-Gateway-Cache": "HIT" if ctx.cache_hit else "MISS",
        "X-Gateway-KV-Hit": "true" if ctx.kv_hit else "false",
        "X-Gateway-Model": ctx.chosen_model or "",
        "X-Gateway-Tier": ctx.chosen_tier or "",
        "X-Gateway-Cost-USD": str(ctx.cost_usd),
        "X-Gateway-TTFT-MS": str(ctx.ttft_ms or ""),
        "X-Gateway-Request-ID": ctx.request_id,
        "X-Gateway-Domain": ctx.domain,
    }


def _select_backend(ctx: RequestContext) -> Backend:
    tier = ctx.chosen_tier or "small"
    # Find matching tier backend
    for backend in _backends:
        if backend.tier == tier:
            return backend
    # Fallback to first available
    if _backends:
        return _backends[0]
    raise HTTPException(status_code=503, detail="No backends available")


async def _run_pipeline(
    ctx: RequestContext,
    pool: ThreadPoolExecutor | None,
) -> tuple[RequestContext, Backend]:
    """Run auth → tokenize → score → route pipeline."""
    loop = asyncio.get_event_loop()

    # Tokenize (CPU-bound → thread pool)
    if pool:
        ctx = await loop.run_in_executor(pool, tokenize_context, ctx)
    else:
        ctx = tokenize_context(ctx)

    # Score complexity (CPU-bound → thread pool)
    if pool:
        ctx = await loop.run_in_executor(pool, score_complexity, ctx, _scorer)
    else:
        ctx = score_complexity(ctx, _scorer)

    # Route to tier
    ctx = route_tier(ctx)

    # Select backend
    backend = _select_backend(ctx)
    ctx.chosen_backend = backend.backend_id
    ctx.chosen_model = backend.model_id

    return ctx, backend


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: GatewayRequest) -> Response:
    start_ts = time.monotonic()
    request_id = str(uuid.uuid4())

    # Auth
    tenant_cfg: TenantConfig = await authenticate(request)

    ctx = RequestContext(
        request_id=request_id,
        tenant_id=tenant_cfg.tenant_id,
        tenant_config=tenant_cfg,
        messages=body.messages,
        model_hint=body.model,
        stream=body.stream,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        gateway_hints=body.gateway_hints,
        start_ts=start_ts,
    )

    # Run pipeline
    pool = _thread_pool
    ctx, backend = await _run_pipeline(ctx, pool)

    if body.stream:
        # Streaming response
        async def stream_gen() -> AsyncIterator[bytes]:
            async for chunk in backend.chat_stream(ctx):
                data = f"data: {chunk}\n\n"
                yield data.encode()
            yield b"data: [DONE]\n\n"

        total_ms = (time.monotonic() - start_ts) * 1000.0
        headers = _build_response_headers(ctx)
        asyncio.create_task(
            asyncio.to_thread(
                telemetry.emit, ctx, total_ms
            )
        )
        return StreamingResponse(stream_gen(), media_type="text/event-stream", headers=headers)

    # Non-streaming
    response_text = await backend.chat(ctx)
    total_ms = (time.monotonic() - start_ts) * 1000.0

    headers = _build_response_headers(ctx)

    # Update metrics
    tier = ctx.chosen_tier or "unknown"
    cache_hit_str = str(ctx.cache_hit).lower()
    kv_hit_str = str(ctx.kv_hit).lower()

    m.gateway_requests_total.labels(
        tier=tier,
        cache_hit=cache_hit_str,
        kv_hit=kv_hit_str,
        domain=ctx.domain,
    ).inc()
    if ctx.ttft_ms is not None:
        m.gateway_ttft_ms.labels(
            tier=tier, cache_hit=cache_hit_str, kv_hit=kv_hit_str
        ).observe(ctx.ttft_ms)
    m.gateway_total_ms.labels(tier=tier).observe(total_ms)
    m.gateway_cost_usd.labels(tier=tier).observe(ctx.cost_usd)

    # Fire-and-forget telemetry
    asyncio.create_task(
        asyncio.to_thread(telemetry.emit, ctx, total_ms)
    )

    # Build OpenAI-compatible response
    openai_response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ctx.chosen_model or body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": ctx.input_tokens,
            "completion_tokens": ctx.output_tokens,
            "total_tokens": ctx.input_tokens + ctx.output_tokens,
        },
    }

    return JSONResponse(content=openai_response, headers=headers)
