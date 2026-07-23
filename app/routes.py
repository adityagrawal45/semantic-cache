"""
routes.py — API surface for the semantic cache service.

Endpoints:
    POST   /v1/chat/completions      OpenAI-compatible drop-in endpoint
    POST   /cache/invalidate         Invalidate by system_hash / model
    DELETE /cache/prefix/{prefix}    Invalidate by key prefix
    GET    /threshold/simulate       Replay logs at a hypothetical threshold
    GET    /health                   Liveness/readiness probe
    GET    /metrics                  Prometheus scrape endpoint
"""

import json
import logging
import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.cache_engine import hash_params, hash_system_prompt
from app.config import get_settings
from app.embeddings import embed_text
from app.metrics import CACHE_LATENCY_SECONDS, estimate_cost_savings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    InvalidateRequest,
    InvalidateResponse,
    ThresholdSimulationResult,
)
from app.near_miss_analyzer import load_near_misses
from app.providers import resolve_provider
from app.ttl_classifier import classify_request_type

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


def _system_prompt_text(request: ChatCompletionRequest) -> str:
    return "\n".join(m.content for m in request.messages if m.role == "system")


def _last_user_prompt(request: ChatCompletionRequest) -> str:
    user_msgs = [m.content for m in request.messages if m.role == "user"]
    return user_msgs[-1] if user_msgs else ""


def _resolve_threshold(request: ChatCompletionRequest) -> float:
    """Adaptive threshold: explicit override > request-type heuristic > default."""
    if request.x_threshold is not None:
        return max(settings.min_similarity_threshold,
                    min(settings.max_similarity_threshold, request.x_threshold))

    req_type = request.x_request_type or classify_request_type(_last_user_prompt(request))
    # Classification-style requests tolerate looser matching; creative
    # generation wants near-exact matches since wording matters a lot.
    type_thresholds = {
        "classification": 0.90,
        "factual": settings.similarity_threshold,
        "time_sensitive": min(0.97, settings.max_similarity_threshold),
        "creative": min(0.98, settings.max_similarity_threshold),
    }
    return type_thresholds.get(req_type, settings.similarity_threshold)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest, response: Response):
    app_state = request.app.state
    cache_engine = app_state.cache_engine

    provider = resolve_provider(body.model)
    system_text = _system_prompt_text(body)
    user_prompt = _last_user_prompt(body)
    system_hash = hash_system_prompt(system_text)
    param_hash = hash_params(body.model_dump(exclude={"x_threshold", "x_request_type", "x_no_cache"}))
    threshold = _resolve_threshold(body)

    start = time.perf_counter()

    if not body.x_no_cache:
        embedding = await embed_text(user_prompt)
        cached_entry, score = await cache_engine.lookup(
            embedding=embedding,
            provider=provider.name,
            system_hash=system_hash,
            param_hash=param_hash,
            prompt_text=user_prompt,
            threshold=threshold,
        )

        if cached_entry is not None:
            elapsed = time.perf_counter() - start
            CACHE_LATENCY_SECONDS.labels(outcome="hit").observe(elapsed)
            usage = cached_entry.get("usage", {})
            estimate_cost_savings(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                settings.cost_per_1k_input_tokens,
                settings.cost_per_1k_output_tokens,
            )
            response.headers["Cache-Hit"] = "true"
            response.headers["X-Similarity-Score"] = f"{score:.4f}"

            if body.stream:
                return StreamingResponse(
                    _replay_cached_stream(cached_entry),
                    media_type="text/event-stream",
                    headers={"Cache-Hit": "true", "X-Similarity-Score": f"{score:.4f}"},
                )

            cached_entry["cache_hit"] = True
            cached_entry["similarity_score"] = score
            return ChatCompletionResponse(**cached_entry)
    else:
        embedding = None

    # --- Cache miss path ---
    response.headers["Cache-Hit"] = "false"

    if body.stream:
        return StreamingResponse(
            _stream_and_cache(body, provider, embedding, system_hash, param_hash, user_prompt, start),
            media_type="text/event-stream",
            headers={"Cache-Hit": "false"},
        )

    result = await provider.complete(body)
    elapsed = time.perf_counter() - start
    CACHE_LATENCY_SECONDS.labels(outcome="miss").observe(elapsed)

    if not body.x_no_cache:
        if embedding is None:
            embedding = await embed_text(user_prompt)
        await request.app.state.cache_engine.store(
            embedding=embedding,
            provider=provider.name,
            system_hash=system_hash,
            param_hash=param_hash,
            prompt_text=user_prompt,
            response_payload=result.model_dump(),
            model=body.model,
        )

    return result


async def _replay_cached_stream(cached_entry: dict):
    """Yield an OpenAI-style SSE stream reconstructed from a cached
    (non-streamed) response. Cache hits stream instantly since there's no
    upstream latency to wait on."""
    content = cached_entry["choices"][0]["message"]["content"]
    chunk = {
        "id": cached_entry.get("id", "cached"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": cached_entry.get("model", ""),
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    done_chunk = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_and_cache(body, provider, embedding, system_hash, param_hash, user_prompt, start_time):
    """Stream from the upstream provider while buffering the full text so
    it can be written to the cache once the stream completes."""
    buffer = []
    model = body.model
    chunk_id = f"chatcmpl-stream-{int(time.time()*1000)}"

    async for delta in provider.stream(body):
        buffer.append(delta)
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    done_chunk = {
        "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    elapsed = time.perf_counter() - start_time
    CACHE_LATENCY_SECONDS.labels(outcome="miss").observe(elapsed)

    full_text = "".join(buffer)
    if not body.x_no_cache:
        from app.cache_engine import CacheEngine  # local import avoids cycle at module load
        from app.models import ChatMessage, Choice, Usage

        approx_tokens = max(1, len(full_text) // 4)
        payload = ChatCompletionResponse(
            model=model,
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=full_text), finish_reason="stop")],
            usage=Usage(prompt_tokens=max(1, len(user_prompt) // 4),
                        completion_tokens=approx_tokens,
                        total_tokens=max(1, len(user_prompt) // 4) + approx_tokens),
        ).model_dump()

        emb = embedding or await embed_text(user_prompt)
        # cache_engine is attached to app.state at startup; grabbed via closure
        # would require request access, so we import a module-level accessor.
        from app.main import get_app_cache_engine
        engine = get_app_cache_engine()
        if engine is not None:
            await engine.store(
                embedding=emb, provider=provider.name, system_hash=system_hash,
                param_hash=param_hash, prompt_text=user_prompt,
                response_payload=payload, model=model,
            )


@router.post("/cache/invalidate", response_model=InvalidateResponse)
async def invalidate(request: Request, body: InvalidateRequest):
    engine = request.app.state.cache_engine
    deleted = await engine.invalidate_by_criteria(system_hash=body.system_hash, model=body.model)
    return InvalidateResponse(deleted_count=deleted, matched_criteria=body.model_dump(exclude_none=True))


@router.delete("/cache/prefix/{prefix}", response_model=InvalidateResponse)
async def invalidate_prefix(request: Request, prefix: str):
    engine = request.app.state.cache_engine
    deleted = await engine.invalidate_by_prefix(prefix)
    return InvalidateResponse(deleted_count=deleted, matched_criteria={"prefix": prefix})


@router.get("/threshold/simulate", response_model=ThresholdSimulationResult)
async def simulate_threshold(threshold: float = 0.90):
    """Replay logged near-miss (and implicitly, hit) data to estimate what
    the hit rate *would have been* at a different threshold. This is a
    lightweight approximation based on near-miss logs; for a rigorous
    analysis, use scripts/threshold_tuner.py against full query logs.
    """
    records = load_near_misses()
    would_have_hit = sum(1 for r in records if r["best_score"] >= threshold)
    total = len(records) or 1
    hit_rate = would_have_hit / total
    est_savings = would_have_hit * (
        settings.cost_per_1k_input_tokens + settings.cost_per_1k_output_tokens
    ) * 0.5  # rough per-request estimate

    return ThresholdSimulationResult(
        threshold=threshold,
        simulated_hit_rate=hit_rate,
        simulated_requests=total,
        simulated_hits=would_have_hit,
        estimated_cost_savings_usd=est_savings,
    )


@router.get("/health")
async def health():
    return {"status": "ok", "service": settings.app_name}


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
