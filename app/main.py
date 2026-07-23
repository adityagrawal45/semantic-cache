"""
main.py — FastAPI application entry point.

Boots the RedisVL index, wires up the CacheEngine, and mounts routes.
Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
or via Docker Compose (see docker-compose.yml).
"""

import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI

from app.cache_engine import CacheEngine
from app.config import get_settings
from app.routes import router

logging.basicConfig(level=getattr(logging, get_settings().log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

settings = get_settings()

# Module-level handle so streaming helpers in routes.py (which don't have
# direct `request` access after the generator starts) can reach the cache
# engine without threading it through every yield. Set once at startup.
_cache_engine_singleton: CacheEngine | None = None


def get_app_cache_engine() -> CacheEngine | None:
    return _cache_engine_singleton


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache_engine_singleton

    logger.info("Starting %s", settings.app_name)
    redis_client = redis.from_url(settings.redis_url, decode_responses=False)

    from redis_index.index_schema import get_or_create_index
    index = await get_or_create_index(redis_client)

    cache_engine = CacheEngine(redis_client=redis_client, index=index)
    app.state.cache_engine = cache_engine
    app.state.redis_client = redis_client
    _cache_engine_singleton = cache_engine

    logger.info("Startup complete. Redis: %s | Index: %s", settings.redis_url, settings.redis_index_name)
    yield

    logger.info("Shutting down %s", settings.app_name)
    await redis_client.aclose()


app = FastAPI(
    title="Semantic Caching Layer for LLM APIs",
    description="Drop-in OpenAI-compatible proxy that caches LLM responses by semantic similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "openai_compatible_endpoint": "/v1/chat/completions",
    }
