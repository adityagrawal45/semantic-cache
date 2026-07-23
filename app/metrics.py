"""
metrics.py — Prometheus metrics for the semantic cache service.

Exposed at GET /metrics (see routes.py) for scraping by Prometheus, and
visualized via the pre-built Grafana dashboard in monitoring/grafana/.
"""

from prometheus_client import Counter, Gauge, Histogram

CACHE_HIT_TOTAL = Counter(
    "cache_hit_total",
    "Number of cache hits",
    labelnames=["provider", "model"],
)

CACHE_MISS_TOTAL = Counter(
    "cache_miss_total",
    "Number of cache misses",
    labelnames=["provider", "model"],
)

CACHE_LATENCY_SECONDS = Histogram(
    "cache_latency_seconds",
    "End-to-end request latency, split by whether it was a cache hit or miss",
    labelnames=["outcome"],  # "hit" | "miss"
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

COST_SAVINGS_DOLLARS = Counter(
    "cost_savings_dollars",
    "Approximate cumulative dollars saved by serving cache hits instead of "
    "calling the upstream provider",
)

CACHE_SIZE_ENTRIES = Gauge(
    "cache_size_entries",
    "Approximate number of live entries in the semantic cache",
)

SIMILARITY_SCORE = Histogram(
    "similarity_score",
    "Distribution of top-1 cosine similarity scores seen during lookups",
    buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99, 1.0),
)


def estimate_cost_savings(prompt_tokens: int, completion_tokens: int,
                           cost_per_1k_input: float, cost_per_1k_output: float) -> float:
    """Estimate the dollar cost that was *avoided* by serving a cache hit
    instead of calling the upstream LLM, and record it in the counter."""
    savings = (prompt_tokens / 1000.0) * cost_per_1k_input + \
              (completion_tokens / 1000.0) * cost_per_1k_output
    COST_SAVINGS_DOLLARS.inc(savings)
    return savings
