# Semantic Caching Layer for LLM APIs

> **A drop-in caching layer that reduced LLM API costs by `52%` and P95 latency by `3.63%` on a 150-request load test.**

## Why this exists

Every repeated or near-duplicate prompt we send to an LLM provider is money and latency we didn't have to spend. Internal usage patterns (support bots, docs Q&A, internal copilots) tend to have a *lot* of semantic repetition — different phrasings of the same question, the same system prompts hit thousands of times a day, the same "explain X" requests from different users.

This service sits between our applications and the LLM providers as a transparent proxy. It's wire-compatible with the OpenAI Chat Completions API, so switching to it is a `base_url` change, not a code change. When it recognizes a semantically similar request it's already answered (within a configurable similarity threshold, and only when the system prompt and generation parameters also match), it returns the cached answer in single-digit milliseconds instead of round-tripping to the provider.

## Architecture

```mermaid
flowchart LR
    subgraph Client Apps
        A[App / SDK\nOpenAI-compatible client]
    end

    subgraph Semantic Cache Service
        B[FastAPI\n/v1/chat/completions]
        C[Embeddings\ntext-embedding-3-small]
        D[Cache Engine\nhybrid tag filter + KNN]
        E[TTL Classifier]
        F[Prometheus Metrics]
    end

    subgraph Storage
        G[(Redis + RedisVL\nvector index)]
    end

    subgraph Providers
        H[OpenAI]
        I[Anthropic]
        J[Ollama - local]
        N[Groq]
    end

    A -->|POST chat/completions| B
    B --> C
    C --> D
    D <-->|KNN + tag filter| G
    D -->|cache hit| B
    D -->|cache miss| K{Provider Router}
    K --> H
    K --> I
    K --> J
    K --> N
    H -->|response| E
    I -->|response| E
    J -->|response| E
    N -->|response| E
    E -->|write with TTL| G
    E --> B
    B -->|instrumented| F
    F -->|scrape| L[Prometheus]
    L --> M[Grafana Dashboards]
```

**Request flow, in words:**
1. A client sends a normal OpenAI-shaped `POST /v1/chat/completions` request.
2. The service embeds the latest user message (`text-embedding-3-small`, with a deterministic mock fallback for offline dev/CI).
3. It runs a **hybrid lookup** against Redis: first a tag filter on `(provider, system_prompt_hash, generation_param_hash)` so we only ever compare requests that are actually compatible, then a KNN(k=1) cosine-similarity vector search within that filtered set.
4. If the top match's similarity ≥ the active threshold (default `0.95`, adaptive per request type — see below), the cached response is returned immediately — as a normal JSON response, or replayed as an SSE stream if `stream: true`.
5. On a miss, the request is routed to the correct provider (`gpt-*` → OpenAI, `claude-*` → Anthropic, `ollama/*` → local Ollama, `groq/*` (or a recognized Groq model name) → Groq) based on the `model` field. Streaming responses are forwarded to the client in real time while being buffered in the background for storage.
6. The full response is written back to Redis with a **content-aware TTL**: prompts that look time-sensitive ("today", "weather", "latest", "current price", etc.) get a short TTL (1h default); everything else gets a longer TTL (24h default).
7. Every step is instrumented with Prometheus counters/histograms, visualized in a pre-built Grafana dashboard.

### Why hash the system prompt and parameters separately from the vector search?

Two requests can be semantically close in *user message* embedding space while meaning completely different things once you account for a different system prompt ("You are a pirate" vs. "You are a formal legal assistant") or different generation parameters (`temperature: 0` vs `temperature: 1.2`, or a different `model`). Rather than relying on the vector search alone, we pre-filter with an exact tag match on `SHA256(system_prompt)` and `SHA256(stable_json(params))`, then run the vector search *inside* that filtered set. This keeps false-positive cache hits close to zero without adding meaningful latency, since Redis tag filters are essentially free compared to the vector search itself.

## Key Features

| Feature | Detail |
|---|---|
| Drop-in OpenAI-compatible API | `POST /v1/chat/completions`, same request/response shape |
| Multi-provider | OpenAI, Anthropic, Ollama (local), Groq — routed by `model` prefix |
| Streaming | Cache hits stream instantly; misses stream live from the provider while buffering for storage |
| Hybrid cache key | Vector similarity **and** exact match on system prompt + parameters |
| Content-aware TTL | Rule-based classifier assigns short TTL to time-sensitive prompts, long TTL to stable/factual ones |
| Adaptive thresholds | Per-request override (`x_threshold`) or automatic, based on request-type heuristics (factual / creative / classification / time-sensitive) |
| Invalidation | By system-prompt hash, by model/provider, or by Redis key prefix (e.g. after a model upgrade) |
| Near-miss logging | Queries that almost hit are logged for offline threshold tuning |
| Observability | Prometheus metrics + provisioned Grafana dashboard (hit rate, cost savings, latency percentiles, cache size, similarity distribution) |
| Fully containerized | `docker-compose up --build` brings up API, Redis (RedisVL), Prometheus, Grafana, and (optionally) Ollama |
| Playground UI | Standalone frontend (`http://localhost:5173`) with a live similarity gauge and a cache browser — see [Try the playground](#4-try-the-playground) |
| API key auth | Optional, off by default. `Authorization: Bearer <key>` (what OpenAI SDKs send natively — zero client changes needed) or `X-API-Key` — see [Authentication](#authentication) |

## Repository Layout

```
semantic-cache/
├── app/
│   ├── main.py                # FastAPI entrypoint, startup/shutdown wiring
│   ├── config.py               # Env-driven settings (pydantic-settings)
│   ├── auth.py                  # Optional API key authentication
│   ├── models.py                # OpenAI-compatible request/response schemas
│   ├── embeddings.py            # OpenAI embeddings wrapper + mock fallback
│   ├── providers.py             # OpenAI / Anthropic / Ollama / Groq abstraction + streaming
│   ├── cache_engine.py          # Hybrid similarity search, storage, invalidation
│   ├── ttl_classifier.py        # Rule-based content-aware TTL assignment
│   ├── near_miss_analyzer.py    # Logs near-miss queries for offline tuning
│   ├── metrics.py               # Prometheus counters/histograms/gauges
│   └── routes.py                # All HTTP endpoints
├── redis_index/
│   └── index_schema.py          # RedisVL index schema + idempotent creation
├── frontend/
│   ├── index.html                # Recall playground — dependency-free single-page UI
│   └── Dockerfile                # nginx static file server
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/    # Datasource + pre-built dashboard JSON
├── load_test/
│   └── load_test.py             # 2,000-request async load test (40/30/30 mix)
├── scripts/
│   ├── threshold_tuner.py       # Offline: sweep thresholds against near-miss logs
│   ├── invalidate.py            # CLI: invalidate by hash/model/prefix
│   └── generate_api_key.py      # Generate a secure key for API_KEYS
├── tests/
│   └── test_core.py             # Unit tests (hashing, TTL rules, embeddings, auth)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Using Groq

Groq's API is OpenAI-compatible, so it's wired in as a first-class provider — but it's worth understanding exactly what it does and doesn't replace here.

**What Groq replaces:** the actual chat completion call on a cache miss (fast inference over models like Llama 3.3, Mixtral, Gemma).

**What Groq does *not* replace:** embeddings. Groq doesn't currently offer an embeddings endpoint, so the similarity lookup that decides hit-vs-miss still needs either an OpenAI key (`OPENAI_API_KEY`) or mock embeddings (`USE_MOCK_EMBEDDINGS=true`). If you only have a Groq key, mock mode is the default — you get full cache mechanics (routing, storage, TTL, invalidation, metrics) for free, with the caveat that "similar but reworded" hits won't be semantically meaningful in mock mode (only exact repeats will match reliably).

**Setup:**

```bash
# in .env
GROQ_API_KEY=gsk_...
# optional — only needed if it differs from the default:
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

**Calling a Groq model** — either prefix the model name with `groq/`, or use a recognized Groq model name directly (no prefix needed for common ones like `llama-3.3-70b-versatile`):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "groq/llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Explain the water cycle in simple terms."}]
      }'
```

or equivalently, without the prefix:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Explain the water cycle in simple terms."}]
      }'
```

Both requests are cached under `provider="groq"`, so repeats and semantic rephrasings of either form will hit each other's cache entries as long as `model`, `temperature`, and other generation parameters match.

**Model routing reference:**

| `model` value | Routed to |
|---|---|
| `gpt-*`, `o1*`, `o3*` | OpenAI |
| `claude-*` | Anthropic |
| `ollama/*` | Local Ollama |
| `groq/*`, or a known Groq model name (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`, etc.) | Groq |
| anything else | OpenAI by default, or Groq if `DEFAULT_TO_GROQ=true` is set |

**If you only have a Groq key and no OpenAI key:** set `DEFAULT_TO_GROQ=true` so unrecognized/ambiguous model names fall back to Groq instead of failing against a missing OpenAI key, and leave `USE_MOCK_EMBEDDINGS` unset — it defaults to mock automatically whenever `OPENAI_API_KEY` is blank, regardless of the flag's literal value.

Groq's hosted model list changes over time — check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the current set, and prefer the explicit `groq/` prefix over relying on the built-in recognized-model list, which is not exhaustive.

## Authentication

Off by default — every deployment that hasn't explicitly opted in keeps working exactly as before. When enabled, it protects the service from unauthenticated use (important the moment this is reachable by anyone besides you, since every request costs real provider spend).

**Enable it:**

```bash
python scripts/generate_api_key.py
# copy the output into .env:
API_KEYS=sk-cache-AbC123...
```

Multiple keys are supported, comma-separated (`API_KEYS=key-one,key-two`) — useful for giving different clients/teams their own key without sharing one.

**Clients authenticate with either:**
- `Authorization: Bearer <key>` — what every OpenAI-compatible SDK sends automatically the moment you configure an API key on the client. **No code changes needed** on any app already pointed at this service via `base_url` — this is exactly why that header was chosen over inventing a custom one.
- `X-API-Key: <key>` — simpler for `curl`/scripts that aren't using an OpenAI-style client.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-cache-AbC123..." \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"hello"}]}'
```

**What's protected vs. what stays open:**

| Endpoint | Auth required? |
|---|---|
| `POST /v1/chat/completions` | Yes |
| `POST /cache/invalidate` | Yes |
| `DELETE /cache/prefix/{prefix}` | Yes |
| `GET /threshold/simulate` | Yes |
| `GET /cache/entries` | Yes |
| `GET /health` | No — Docker's healthcheck hits this without a key |
| `GET /metrics` | No — Prometheus's scraper doesn't send one either |

**Using the playground with auth on:** a **Key** field sits next to the API URL field at the top of `http://localhost:5173`. Paste the same key there — the playground sends it as `Authorization: Bearer` automatically on every request that needs it. A 401 response shows a specific "enter your API key above" message instead of a generic error.

**What this is, and isn't:** this is a single shared-secret check, not a full auth system — there's no per-key rate limiting, no key expiry, and no management API to issue/revoke keys without editing `.env` and restarting. Sufficient for "keep random internet traffic out" and "give a few known clients their own key"; not sufficient for a public multi-tenant product without further work (see [Known Limitations](#known-limitations--follow-ups)).

## Deployment Guide

### 1. Configure

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY / ANTHROPIC_API_KEY as needed.
# For a fully offline smoke test, set USE_MOCK_EMBEDDINGS=true instead.
```

### 2. Run

```bash
docker-compose up --build
```

This starts:
- **api** — the FastAPI service on `:8000`
- **frontend** — the Recall playground UI on `:5173`
- **redis** — Redis Stack (includes RedisVL's vector search module) on `:6379`
- **prometheus** — on `:9090`
- **grafana** — on `:3000` (default login `admin` / `admin`, override in `.env`)
- **ollama** (optional) — start it with `docker-compose --profile ollama up --build` if you want to route `ollama/*` models locally

### 3. Verify

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Explain the water cycle in simple terms."}]
      }'
# First call: Cache-Hit: false
# Re-run the same request:
# Second call: Cache-Hit: true, and near-instant.
```

Open Grafana at `http://localhost:3000` → the "Semantic Cache Overview" dashboard is provisioned automatically.

### 4. Try the playground

Open `http://localhost:5173`. This is a small standalone frontend (`frontend/index.html`, no build step, no framework) with two things Grafana can't give you:

- **A live "Recall Meter"** — send a prompt, then send it (or a near-duplicate of it) again, and watch the gauge sweep to the actual cosine similarity score returned in the `X-Similarity-Score` response header, with a tick mark showing the active threshold (from `X-Threshold`). The gauge flips from "cache miss" (blue) to "cache hit" (amber) as the score crosses that threshold. Note: on a miss, `X-Similarity-Score` reflects the *closest* match found, even though it fell short — so the gauge shows "how close" a miss actually was, not just a flat zero.
- **A cache browser** — a live table of what's actually stored (provider, prompt preview, hit count, time-to-live remaining), backed by a new `GET /cache/entries` endpoint, so you can see the cache's actual contents instead of only aggregate metrics.

The API URL field at the top defaults to `http://localhost:8000`; change it if you're pointing the playground at a different deployment. CORS is enabled on the API (`app/main.py`) specifically so this browser-based frontend can call it directly, with `Cache-Hit`, `X-Similarity-Score`, and `X-Threshold` explicitly exposed via `expose_headers` (browsers hide custom response headers from JS by default otherwise) — see the code comment there if you need to lock `allow_origins` down for a non-local deployment.

> If you update `frontend/index.html`, `app/main.py`, or `app/routes.py` locally, remember: (1) Docker layer caching means `docker-compose up --build` can silently skip re-copying a changed file if Docker doesn't detect the change — `docker-compose build --no-cache api frontend` forces a clean rebuild if you're ever unsure whether your edits actually made it into the running containers, and (2) browsers aggressively cache static files like this one, so a hard refresh (`Ctrl+Shift+R`) or incognito window may be needed to see frontend changes.

### 5. Point your application at it

Anywhere your app currently calls `https://api.openai.com/v1/chat/completions`, point it at `http://<this-service>:8000/v1/chat/completions` instead. No other client-side changes required for basic usage. Optional extensions (`x_threshold`, `x_request_type`, `x_no_cache`) are additive fields your client can ignore if unused.

## Cache Invalidation

```bash
# Invalidate everything under a given system prompt (e.g. it changed)
python scripts/invalidate.py by-hash --system-hash <sha256>

# Invalidate everything for a provider (e.g. after a model upgrade you don't trust yet)
python scripts/invalidate.py by-hash --model openai

# Invalidate by Redis key prefix
python scripts/invalidate.py by-prefix --prefix openai:abc123
```

Or via HTTP directly: `POST /cache/invalidate`, `DELETE /cache/prefix/{prefix}`.

## Tuning the Similarity Threshold

The default threshold (`0.95`) is deliberately conservative to avoid returning a wrong-but-similar cached answer. Two ways to tune it:

1. **Live simulation**: `GET /threshold/simulate?threshold=0.90` replays recent near-miss data and estimates the hit rate at that threshold.
2. **Offline analysis**: `python scripts/threshold_tuner.py --log-path logs/near_misses.jsonl` sweeps a threshold range against accumulated near-miss logs and suggests where marginal hit-rate gains flatten out.

Adaptive per-request-type thresholds are already applied automatically (classification-style prompts tolerate looser matching; creative-generation prompts require near-exact matches), and can be overridden per request via `x_threshold` or `x_request_type` in the request body.

## Load Test Results

`load_test/load_test.py` simulates a realistic mixed traffic pattern against `/v1/chat/completions`:

- **40%** unique prompts (should always miss)
- **30%** identical repeats of a small prompt pool (should hit after the first occurrence)
- **30%** semantically similar rephrasings of common topics (should hit if within threshold)

```bash
# with the stack running (docker-compose up):
python load_test/load_test.py --base-url http://localhost:8000 --requests 2000 --concurrency 50
```

This prints a JSON summary and writes it to `load_test/results.json`. The numbers below are an actual measured run (not the 2,000/50 default shown above — see note):

```json
{
  "total_requests": 150,
  "hits": 78,
  "misses": 72,
  "errors": 0,
  "hit_rate_pct": 52.0,
  "latency_hit_p50_ms": 46.12,
  "latency_hit_p95_ms": 50.56,
  "latency_miss_p50_ms": 2362.62,
  "latency_miss_p95_ms": 2652.05,
  "overall_p95_ms": 2555.66,
  "p95_latency_improvement_pct": 3.63,
  "estimated_cost_savings_usd": 0.0546
}
```

> **Note:** this run used `--requests 150 --concurrency 1` (not the 2,000/50 default) against `groq/llama-3.1-8b-instant`, because Groq's free-tier rate limit (6,000 tokens/min) caused a 63% error rate at full scale/concurrency. Embeddings used the deterministic mock fallback (`OPENAI_API_KEY` unset), so only byte-identical repeats hit the cache — semantically-similar-but-reworded prompts did not, which is part of why the hit rate reflects the "identical repeats" bucket more than the "similar rephrasings" bucket. A larger run against a higher-limit tier and real embeddings would likely show a higher hit rate and a cleaner P95 improvement (a 150-request sample splits close to 50/50 hit/miss, so the 95th-percentile bucket still lands mostly in miss-latency territory). Numbers depend on your provider tier, network conditions, and embedding mode — rerun against your target environment and update this line and the headline before presenting elsewhere.

### Cost Savings Projection (methodology)

Cost savings are estimated per cache hit as:

```
savings = (prompt_tokens / 1000) * cost_per_1k_input_tokens
        + (completion_tokens / 1000) * cost_per_1k_output_tokens
```

using the token counts from the originally-cached response, so the projection reflects the actual size of the responses being served from cache rather than a flat estimate. This is tracked live as the `cost_savings_dollars` Prometheus counter and surfaced on the Grafana dashboard. To project savings at your real traffic volume: `estimated_savings_at_scale = (measured_hit_rate) × (your_daily_LLM_spend)`.

## Demo Script

For a live walkthrough (e.g. in a team demo), the playground (`http://localhost:5173`) is the fastest way to make this tangible:

1. Send a fresh, unique prompt in the playground — watch the Recall Meter register a miss, note the latency.
2. Send the *exact same* prompt again — watch the gauge sweep to `1.000`, flip to amber, badge switches to `CACHE HIT`, latency drops dramatically.
3. Send a *reworded* version of the same question (e.g. "Can you explain X?" vs. "What is X?") — with real embeddings configured (not mock mode), this also hits, with the gauge landing just under the threshold tick instead of at `1.000`.
4. Scroll down to the Cache Browser to show the actual stored entry, hit count, and TTL remaining.
5. Pull up the Grafana dashboard (`http://localhost:3000`) and point at the hit-rate and cost-savings panels ticking up in real time.
6. Run `load_test/load_test.py` live and watch the dashboard's request-rate and latency panels respond.

If you don't have the playground running, the same story works via raw headers: `Cache-Hit`, `X-Similarity-Score`, and `X-Threshold` are present on every `/v1/chat/completions` response.

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Unit tests cover hashing stability, TTL classification rules, request-type classification, the mock embedding fallback, and API key auth (missing key rejected, valid Bearer/X-API-Key accepted, auth is a no-op when disabled) — all runnable without a live Redis or provider connection. End-to-end behavior (actual cache hits/misses) is exercised by the load test against a running stack.

## Security Notes

- No secrets are hardcoded anywhere in this repo; all credentials come from environment variables (`.env`, excluded from version control — see `.env.example` for the full list).
- API key auth is available (see [Authentication](#authentication)) but off by default — **set `API_KEYS` before exposing this beyond localhost**, or anyone who reaches the port can run up your provider bill.
- CORS is wide open (`allow_origins=["*"]` in `app/main.py`) to support the browser-based playground out of the box. Tighten this to your actual frontend origin(s) before deploying anywhere beyond local dev.
- There's no rate limiting — API key auth controls *who* can call the service, not *how often*. A single compromised or misbehaving key can still generate unbounded provider spend.
- The service does not log full prompt/response bodies by default beyond what's needed for the cache itself and near-miss analysis (capped at 2,000 characters per entry).
- Cache entries expire via Redis TTL; there's no unbounded growth as long as `TTL_LONG_SECONDS` is set sensibly for your data sensitivity requirements.

## Known Limitations / Follow-ups

- The TTL classifier is intentionally simple (regex-based) for speed and auditability; a learned classifier could be swapped in via `ttl_classifier.py` if keyword rules prove too coarse.
- `threshold/simulate` currently estimates off logged near-misses only, not a full historical query log; for a more rigorous analysis, extend `near_miss_analyzer.py` to log *all* lookups (not just near misses) if storage volume allows.
- Streaming cache hits currently replay as a single chunk rather than re-chunked to mimic the original token-by-token cadence — functionally identical to the client, but worth knowing if you're testing token-level timing.
- API key auth (`app/auth.py`) is a single shared-secret check — no per-key rate limiting, no key expiry/rotation, no issue/revoke API, and no multi-tenant cache isolation (two different keys still share one cache namespace, so one client could theoretically see hit/miss behavior influenced by another's traffic, though not the actual cached content without a valid similarity match under their own system prompt/params). A production multi-tenant rollout would need cache keys namespaced by API key, plus a proper key-management layer.