"""
load_test/load_test.py — Async load test for the semantic cache service.

Simulates a realistic traffic mix against POST /v1/chat/completions:
    40% unique prompts        -> expected cache misses
    30% identical repeats     -> expected hits after the first occurrence
    30% semantically similar  -> expected hits if within threshold

Usage:
    python load_test/load_test.py --base-url http://localhost:8000 --requests 2000

Outputs a summary (hit rate, latency percentiles, estimated cost savings)
to stdout and to load_test/results.json, which README.md's numbers are
sourced from.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx

# --- Prompt pools -----------------------------------------------------------

_TOPICS = [
    "the history of the Roman Empire", "how photosynthesis works",
    "the plot of Hamlet", "how to bake sourdough bread",
    "the causes of World War I", "how neural networks learn",
    "the rules of chess", "the water cycle", "how vaccines work",
    "the theory of relativity", "how compound interest works",
    "the French Revolution", "how the stock market works",
    "the life cycle of a star", "how DNA replication works",
]

_TEMPLATES = [
    "Explain {topic} in simple terms.",
    "Can you summarize {topic}?",
    "I'd like to understand {topic}. Can you help?",
    "Give me a brief overview of {topic}.",
    "What should I know about {topic}?",
]


def _unique_prompt(i: int) -> str:
    return f"Tell me something unique about topic #{i}: {random.random()}"


def _identical_prompt_pool(n: int) -> List[str]:
    topics = random.sample(_TOPICS, min(n, len(_TOPICS)))
    return [f"Explain {t} in simple terms." for t in (topics * ((n // len(topics)) + 1))[:n]]


def _similar_prompt(base_topic: str) -> str:
    template = random.choice(_TEMPLATES)
    return template.format(topic=base_topic)


@dataclass
class RequestResult:
    latency_s: float
    cache_hit: bool
    status_code: int


@dataclass
class Summary:
    total: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    latencies_hit: List[float] = field(default_factory=list)
    latencies_miss: List[float] = field(default_factory=list)


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * pct)
    idx = min(idx, len(data_sorted) - 1)
    return data_sorted[idx]


async def _send_one(client: httpx.AsyncClient, prompt: str, base_url: str) -> RequestResult:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)
        elapsed = time.perf_counter() - start
        cache_hit = resp.headers.get("Cache-Hit", "false") == "true"
        return RequestResult(latency_s=elapsed, cache_hit=cache_hit, status_code=resp.status_code)
    except Exception:
        elapsed = time.perf_counter() - start
        return RequestResult(latency_s=elapsed, cache_hit=False, status_code=0)


def _build_prompt_sequence(n: int) -> List[str]:
    n_unique = int(n * 0.40)
    n_identical = int(n * 0.30)
    n_similar = n - n_unique - n_identical

    prompts = []
    prompts += [_unique_prompt(i) for i in range(n_unique)]
    prompts += _identical_prompt_pool(n_identical)
    prompts += [_similar_prompt(random.choice(_TOPICS)) for _ in range(n_similar)]

    random.shuffle(prompts)
    return prompts


async def run_load_test(base_url: str, n_requests: int, concurrency: int,
                         cost_per_1k_input: float, cost_per_1k_output: float) -> Summary:
    prompts = _build_prompt_sequence(n_requests)
    summary = Summary()
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def worker(prompt: str):
            async with sem:
                result = await _send_one(client, prompt, base_url)
                summary.total += 1
                if result.status_code != 200:
                    summary.errors += 1
                    return
                if result.cache_hit:
                    summary.hits += 1
                    summary.latencies_hit.append(result.latency_s)
                else:
                    summary.misses += 1
                    summary.latencies_miss.append(result.latency_s)

        await asyncio.gather(*(worker(p) for p in prompts))

    return summary


def print_and_save_summary(summary: Summary, output_path: str,
                            cost_per_1k_input: float, cost_per_1k_output: float):
    hit_rate = summary.hits / summary.total if summary.total else 0
    all_latencies = summary.latencies_hit + summary.latencies_miss

    # Rough cost-savings estimate: assume an average request would have cost
    # ~$0.001 combined input+output at these rates; every hit avoids that.
    avg_tokens_saved_cost = (500 / 1000) * cost_per_1k_input + (300 / 1000) * cost_per_1k_output
    estimated_savings = summary.hits * avg_tokens_saved_cost

    p50_hit = _percentile(summary.latencies_hit, 0.50)
    p95_hit = _percentile(summary.latencies_hit, 0.95)
    p50_miss = _percentile(summary.latencies_miss, 0.50)
    p95_miss = _percentile(summary.latencies_miss, 0.95)
    p95_overall = _percentile(all_latencies, 0.95)

    # If we had a "no-cache" baseline, P95 improvement = (baseline_p95 - p95_overall) / baseline_p95.
    # We approximate baseline P95 as the miss-path P95, since that's what
    # every request would cost without caching.
    p95_improvement_pct = ((p95_miss - p95_overall) / p95_miss * 100) if p95_miss > 0 else 0

    result = {
        "total_requests": summary.total,
        "hits": summary.hits,
        "misses": summary.misses,
        "errors": summary.errors,
        "hit_rate_pct": round(hit_rate * 100, 2),
        "latency_hit_p50_ms": round(p50_hit * 1000, 2),
        "latency_hit_p95_ms": round(p95_hit * 1000, 2),
        "latency_miss_p50_ms": round(p50_miss * 1000, 2),
        "latency_miss_p95_ms": round(p95_miss * 1000, 2),
        "overall_p95_ms": round(p95_overall * 1000, 2),
        "p95_latency_improvement_pct": round(p95_improvement_pct, 2),
        "estimated_cost_savings_usd": round(estimated_savings, 4),
    }

    print(json.dumps(result, indent=2))
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load test the semantic cache service.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--cost-per-1k-input", type=float, default=0.0005)
    parser.add_argument("--cost-per-1k-output", type=float, default=0.0015)
    parser.add_argument("--output", default="load_test/results.json")
    args = parser.parse_args()

    summary = asyncio.run(run_load_test(
        args.base_url, args.requests, args.concurrency,
        args.cost_per_1k_input, args.cost_per_1k_output,
    ))
    print_and_save_summary(summary, args.output, args.cost_per_1k_input, args.cost_per_1k_output)


if __name__ == "__main__":
    main()
