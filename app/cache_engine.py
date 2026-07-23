"""
cache_engine.py — Similarity search, cache storage, and key generation.

Design:
  Each cache entry is stored as a Redis hash (via RedisVL) with fields:
    - embedding      : the prompt's vector embedding (float32 blob)
    - system_hash    : SHA256 of the system prompt (exact-match pre-filter)
    - param_hash     : SHA256 of a stable serialization of (model, temperature,
                        max_tokens, top_p, stop, ...) (exact-match pre-filter)
    - provider       : provider name, e.g. "openai" (tag field)
    - prompt_text     : original prompt, for debugging/analytics
    - response_json   : the full serialized ChatCompletionResponse
    - created_at      : unix timestamp
    - hit_count       : integer, incremented on each hit
    - ttl_seconds     : the TTL that was assigned at write time

  Lookup strategy (hybrid filter + vector search, as specified):
    1. Build a Redis tag filter on (provider, system_hash, param_hash) so we
       only ever compare vectors within requests that are actually
       compatible (same system prompt + same generation parameters).
    2. Run a KNN(top_k=1) vector search restricted to that filter.
    3. If the top result's similarity >= active threshold, it's a hit.

  This keeps false-positive risk near zero: two prompts can only match if
  they were issued against the same system prompt and same parameters AND
  are semantically close enough.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.metrics import (
    CACHE_HIT_TOTAL,
    CACHE_MISS_TOTAL,
    CACHE_SIZE_ENTRIES,
    SIMILARITY_SCORE,
)
from app.near_miss_analyzer import log_near_miss
from app.ttl_classifier import classify_ttl

logger = logging.getLogger(__name__)
settings = get_settings()

# Fields that are excluded from the parameter hash because they don't affect
# the *content* of a valid cache match (e.g. streaming is a transport detail,
# and `user` is just an identifier for abuse tracking on the provider side).
_PARAM_HASH_EXCLUDE = {"messages", "stream", "user", "x_threshold", "x_request_type", "x_no_cache"}


def hash_system_prompt(system_prompt: str) -> str:
    """SHA256 of the system prompt (empty string if there isn't one)."""
    return hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest()


def hash_params(params: Dict[str, Any]) -> str:
    """SHA256 of a stable (sorted-key) JSON serialization of the generation
    parameters, excluding fields listed in _PARAM_HASH_EXCLUDE."""
    filtered = {k: v for k, v in params.items() if k not in _PARAM_HASH_EXCLUDE}
    stable = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def build_cache_key_suffix(provider: str, system_hash: str, param_hash: str, embedding_hash: str) -> str:
    """Human-readable Redis key *suffix* for a cache entry (RedisVL prepends
    the index's configured prefix automatically when the document is
    loaded with id_field='id', so this must NOT include that prefix).
    The embedding_hash suffix keeps keys unique even when two prompts
    share provider/system/param hashes but differ semantically."""
    return f"{provider}:{system_hash[:12]}:{param_hash[:12]}:{embedding_hash[:16]}"


def full_redis_key(suffix: str) -> str:
    """Reconstruct the full Redis key from a RedisVL 'id' field value."""
    return f"{settings.redis_prefix}{suffix}"


class CacheEngine:
    """Wraps a RedisVL SearchIndex to provide semantic cache get/set."""

    def __init__(self, redis_client, index):
        self._redis = redis_client
        self._index = index

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    async def lookup(
        self,
        embedding: list,
        provider: str,
        system_hash: str,
        param_hash: str,
        prompt_text: str,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Return (cached_entry, similarity_score) or (None, best_score)."""
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        active_threshold = threshold if threshold is not None else settings.similarity_threshold

        filter_expr = (
            (Tag("provider") == provider)
            & (Tag("system_hash") == system_hash)
            & (Tag("param_hash") == param_hash)
        )

        query = VectorQuery(
            vector=np.array(embedding, dtype=np.float32).tobytes(),
            vector_field_name="embedding",
            return_fields=["id", "response_json", "prompt_text", "hit_count", "created_at"],
            num_results=1,
            filter_expression=filter_expr,
        )

        try:
            results = await self._index.query(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("RedisVL query failed: %s", exc)
            CACHE_MISS_TOTAL.labels(provider=provider, model="unknown").inc()
            return None, None

        if not results:
            CACHE_MISS_TOTAL.labels(provider=provider, model="unknown").inc()
            return None, None

        top = results[0]
        # RedisVL returns cosine *distance* by default; convert to similarity.
        score = 1.0 - float(top.get("vector_distance", 1.0))
        SIMILARITY_SCORE.observe(score)

        if score >= active_threshold:
            CACHE_HIT_TOTAL.labels(provider=provider, model="unknown").inc()
            entry = json.loads(top["response_json"])
            if top.get("id"):
                await self._bump_hit_count(full_redis_key(top["id"]))
            return entry, score

        # Not a hit — log as a near miss if it's close enough to be
        # interesting for threshold tuning.
        log_near_miss(prompt_text, score, active_threshold, provider)
        CACHE_MISS_TOTAL.labels(provider=provider, model="unknown").inc()
        return None, score

    async def _bump_hit_count(self, redis_key: Optional[str]) -> None:
        if not redis_key:
            return
        try:
            await self._redis.hincrby(redis_key, "hit_count", 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not bump hit_count for %s: %s", redis_key, exc)

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    async def store(
        self,
        embedding: list,
        provider: str,
        system_hash: str,
        param_hash: str,
        prompt_text: str,
        response_payload: Dict[str, Any],
        model: str,
    ) -> str:
        """Persist a fresh (miss) response into the cache. Returns the key."""
        embedding_hash = hashlib.sha256(np.array(embedding, dtype=np.float32).tobytes()).hexdigest()
        key_suffix = build_cache_key_suffix(provider, system_hash, param_hash, embedding_hash)
        full_key = full_redis_key(key_suffix)
        ttl_seconds = classify_ttl(prompt_text)

        doc = {
            "id": key_suffix,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "system_hash": system_hash,
            "param_hash": param_hash,
            "provider": provider,
            "prompt_text": prompt_text[:2000],  # cap for storage sanity
            "response_json": json.dumps(response_payload),
            "created_at": time.time(),
            "hit_count": 0,
            "ttl_seconds": ttl_seconds,
        }

        try:
            await self._index.load([doc], id_field="id")
            await self._redis.expire(full_key, ttl_seconds)
            CACHE_SIZE_ENTRIES.inc()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to store cache entry %s: %s", full_key, exc)

        return full_key

    # ------------------------------------------------------------------ #
    # Invalidation
    # ------------------------------------------------------------------ #
    async def invalidate_by_criteria(self, system_hash: Optional[str] = None,
                                      model: Optional[str] = None) -> int:
        """Delete all entries matching the given system_hash and/or model
        (provider is treated loosely as "model family" here for simplicity).
        """
        from redisvl.query.filter import Tag

        filters = []
        if system_hash:
            filters.append(Tag("system_hash") == system_hash)
        if model:
            filters.append(Tag("provider") == model)

        if not filters:
            return 0

        combined = filters[0]
        for f in filters[1:]:
            combined = combined & f

        from redisvl.query import FilterQuery

        query = FilterQuery(filter_expression=combined, return_fields=["id"])
        results = await self._index.query(query)
        keys = [full_redis_key(r["id"]) for r in results]

        deleted = 0
        for k in keys:
            try:
                await self._redis.delete(k)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to delete key %s: %s", k, exc)

        CACHE_SIZE_ENTRIES.dec(deleted)
        return deleted

    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Delete all keys under the given Redis key prefix."""
        full_prefix = f"{settings.redis_prefix}{prefix}"
        deleted = 0
        async for k in self._redis.scan_iter(match=f"{full_prefix}*"):
            await self._redis.delete(k)
            deleted += 1
        CACHE_SIZE_ENTRIES.dec(deleted)
        return deleted
