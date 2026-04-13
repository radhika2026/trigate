# TriGate

Production-grade OpenAI-compatible LLM gateway with semantic caching, KV-cache-aware scheduling, and contextual bandit routing.

## Quick start

```bash
pip install -e ".[dev]"
uvicorn gateway.main:app --reload
```

## API

- `POST /v1/chat/completions` — OpenAI-compatible
- `GET /health`
- `GET /ready`
- `GET /metrics` — Prometheus format
- `POST /admin/cache/flush`
- `GET /admin/bandit/state`

## Dev

```bash
pytest tests/unit/ -x -q
```
