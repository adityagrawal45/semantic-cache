# Semantic Caching Layer for LLM APIs — Documentation

This document explains what the service does, how it works, and how to run it.

## What this service does

This service is a smart cache for OpenAI-style chat requests. It receives a request, checks whether the same question has already been answered with the same context and settings, and reuses that answer if it can.

In plain language:
- If a new request is effectively the same as a previous one, it returns the cached answer.
- If the request is different, it forwards it to the upstream model provider, returns the live result, and saves that result for future reuse.

That means repeated or paraphrased questions can be answered faster and without calling the LLM provider again.

## How it works

1. The client sends `POST /v1/chat/completions`.
2. The service reads the request and extracts the system prompt and the last user message.
3. It computes hashes for the system prompt and the request parameters.
4. It converts the user message into an embedding vector.
5. It searches RedisVL for a prior response that:
   - used the same provider,
   - had the same system prompt,
   - had the same generation parameters,
   - and had a semantically similar user message.
6. If a match is good enough, it returns the cached answer.
7. If not, it sends the request to the upstream provider and stores the result in Redis.

## What it checks before returning cached data

The cache only returns a prior answer when both of these are true:
- The request context is the same (system prompt, model, temperature, stop tokens, etc.).
- The user message is semantically similar to a prior request.

This avoids returning cached answers when the meaning changes, even if the words look similar.

## Main files and purpose

- `app/main.py` — starts the FastAPI app and connects to Redis.
- `app/config.py` — loads settings from environment variables.
- `app/models.py` — defines request and response shapes.
- `app/embeddings.py` — gets embeddings from OpenAI or uses mock embeddings for local development.
- `app/providers.py` — sends requests to OpenAI, Anthropic, Ollama, or Groq.
- `app/cache_engine.py` — performs cache lookup and stores results.
- `app/ttl_classifier.py` — decides how long cached entries should live.
- `app/near_miss_analyzer.py` — logs near misses for tuning.
- `app/metrics.py` — defines Prometheus metrics.
- `app/routes.py` — implements the HTTP endpoints.
- `redis_index/index_schema.py` — defines the RedisVL index.
- `Dockerfile` and `docker-compose.yml` — run the service and supporting stack.

## API Reference

### `POST /v1/chat/completions`

This is the main endpoint and works like OpenAI's chat completion API. It also supports optional cache controls:
- `x_threshold` — override similarity threshold.
- `x_request_type` — hint whether the request is `factual`, `creative`, `classification`, or `time_sensitive`.
- `x_no_cache` — bypass the cache for this request.

The response includes `cache_hit` and `similarity_score` when the result came from cache.

### `POST /cache/invalidate`

Invalidate entries by `system_hash` or `model`.

### `DELETE /cache/prefix/{prefix}`

Delete all keys matching the specified prefix.

### `GET /threshold/simulate`

Estimate the hit rate for a different similarity threshold using logged near miss data.

### `GET /health`

Returns a health check response.

### `GET /metrics`

Returns Prometheus metrics.

## Configuration

Settings are loaded from environment variables. Important values include:
- `REDIS_URL`
- `REDIS_INDEX_NAME`
- `REDIS_PREFIX`
- `SIMILARITY_THRESHOLD`
- `TTL_SHORT_SECONDS`
- `TTL_LONG_SECONDS`
- `USE_MOCK_EMBEDDINGS`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`

If you do not have an OpenAI key, set `USE_MOCK_EMBEDDINGS=true` to run locally with deterministic mock embeddings.

## Deployment

1. Copy `.env.example` to `.env`.
2. Set provider keys or `USE_MOCK_EMBEDDINGS=true`.
3. Run:

```bash
docker-compose up --build
```

4. Verify with:

```bash
curl http://localhost:8000/health
```

The API is available at `http://localhost:8000`.

## Testing

Run unit tests with:

```bash
pytest tests/ -v
```

## What it does exactly

This service acts as a proxy between your application and an LLM provider. It does not require your code to change beyond pointing requests at a new URL.

On each request, it either:
- returns a cached response if the request matches a prior request closely enough, or
- forwards the request to the real provider and saves the response for later.

It decides matches using both exact request context and semantic similarity, which makes it safer than a plain cache based only on text or hashes.

## Practical benefit

Use this service when you want to reduce repeated LLM calls for similar questions and save both latency and cost without changing the request format.
