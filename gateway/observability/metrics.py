"""Prometheus metrics definitions — all 14 required metrics."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# 1. gateway_requests_total
# ---------------------------------------------------------------------------
gateway_requests_total = Counter(
    "gateway_requests_total",
    "Total number of requests handled",
    ["tier", "cache_hit", "kv_hit", "domain"],
)

# ---------------------------------------------------------------------------
# 2. gateway_ttft_ms
# ---------------------------------------------------------------------------
gateway_ttft_ms = Histogram(
    "gateway_ttft_ms",
    "Time to first token in milliseconds",
    ["tier", "cache_hit", "kv_hit"],
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000],
)

# ---------------------------------------------------------------------------
# 3. gateway_total_ms
# ---------------------------------------------------------------------------
gateway_total_ms = Histogram(
    "gateway_total_ms",
    "Total request latency in milliseconds",
    ["tier"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 15000, 30000],
)

# ---------------------------------------------------------------------------
# 4. gateway_cost_usd
# ---------------------------------------------------------------------------
gateway_cost_usd = Histogram(
    "gateway_cost_usd",
    "Per-request cost in USD",
    ["tier"],
    buckets=[0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
)

# ---------------------------------------------------------------------------
# 5. gateway_cache_similarity
# ---------------------------------------------------------------------------
gateway_cache_similarity = Histogram(
    "gateway_cache_similarity",
    "Cosine similarity score from semantic cache lookup",
    ["domain", "hit"],
    buckets=[0.5, 0.7, 0.8, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0],
)

# ---------------------------------------------------------------------------
# 6. gateway_cache_hit_total
# ---------------------------------------------------------------------------
gateway_cache_hit_total = Counter(
    "gateway_cache_hit_total",
    "Number of semantic cache hits",
    ["domain"],
)

# ---------------------------------------------------------------------------
# 7. gateway_kv_pages_matched
# ---------------------------------------------------------------------------
gateway_kv_pages_matched = Histogram(
    "gateway_kv_pages_matched",
    "Number of KV-cache pages matched per request",
    [],
    buckets=[0, 1, 2, 4, 8, 16, 32, 64, 128],
)

# ---------------------------------------------------------------------------
# 8. gateway_kv_hits_total
# ---------------------------------------------------------------------------
gateway_kv_hits_total = Counter(
    "gateway_kv_hits_total",
    "Number of KV-cache hits",
    ["backend_id"],
)

# ---------------------------------------------------------------------------
# 9. gateway_bandit_arm
# ---------------------------------------------------------------------------
gateway_bandit_arm = Counter(
    "gateway_bandit_arm",
    "Number of times each bandit arm was selected",
    ["arm", "tier"],
)

# ---------------------------------------------------------------------------
# 10. gateway_bandit_reward
# ---------------------------------------------------------------------------
gateway_bandit_reward = Histogram(
    "gateway_bandit_reward",
    "Bandit reward signal per request",
    ["arm"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---------------------------------------------------------------------------
# 11. gateway_backend_queue_depth
# ---------------------------------------------------------------------------
gateway_backend_queue_depth = Gauge(
    "gateway_backend_queue_depth",
    "Current request queue depth for each backend",
    ["backend_id"],
)

# ---------------------------------------------------------------------------
# 12. gateway_embed_latency_ms
# ---------------------------------------------------------------------------
gateway_embed_latency_ms = Histogram(
    "gateway_embed_latency_ms",
    "Embedding computation latency in milliseconds",
    [],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

# ---------------------------------------------------------------------------
# 13. gateway_complexity_ms
# ---------------------------------------------------------------------------
gateway_complexity_ms = Histogram(
    "gateway_complexity_ms",
    "Complexity scoring latency in milliseconds",
    ["scorer_version"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 25],
)

# ---------------------------------------------------------------------------
# 14. gateway_pipeline_ms
# ---------------------------------------------------------------------------
gateway_pipeline_ms = Histogram(
    "gateway_pipeline_ms",
    "Total pipeline processing latency in milliseconds",
    [],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

# ---------------------------------------------------------------------------
# Convenience: list of all metric names for testing
# ---------------------------------------------------------------------------
ALL_METRIC_NAMES = [
    "gateway_requests_total",
    "gateway_ttft_ms",
    "gateway_total_ms",
    "gateway_cost_usd",
    "gateway_cache_similarity",
    "gateway_cache_hit_total",
    "gateway_kv_pages_matched",
    "gateway_kv_hits_total",
    "gateway_bandit_arm",
    "gateway_bandit_reward",
    "gateway_backend_queue_depth",
    "gateway_embed_latency_ms",
    "gateway_complexity_ms",
    "gateway_pipeline_ms",
]
