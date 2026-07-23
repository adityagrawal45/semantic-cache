"""
redis_index/index_schema.py — RedisVL index definition + creation helper.

The index is created idempotently on service startup (see app/main.py's
lifespan handler). It can also be run standalone:

    python -m redis_index.index_schema
"""

import asyncio
import logging

from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_schema() -> IndexSchema:
    settings = get_settings()
    return IndexSchema.from_dict(
        {
            "index": {
                "name": settings.redis_index_name,
                "prefix": settings.redis_prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "provider", "type": "tag"},
                {"name": "system_hash", "type": "tag"},
                {"name": "param_hash", "type": "tag"},
                {"name": "prompt_text", "type": "text"},
                {"name": "response_json", "type": "text"},
                {"name": "created_at", "type": "numeric"},
                {"name": "hit_count", "type": "numeric"},
                {"name": "ttl_seconds", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": settings.embedding_dim,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


async def get_or_create_index(redis_client=None) -> AsyncSearchIndex:
    """Connect to Redis and ensure the vector index exists. Returns the
    RedisVL AsyncSearchIndex handle used by CacheEngine."""
    settings = get_settings()
    schema = build_schema()
    index = AsyncSearchIndex(schema)

    if redis_client is not None:
        await index.set_client(redis_client)
    else:
        await index.connect(settings.redis_url)

    if not await index.exists():
        logger.info("Creating RedisVL index '%s'", settings.redis_index_name)
        await index.create(overwrite=False)
    else:
        logger.info("RedisVL index '%s' already exists", settings.redis_index_name)

    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(get_or_create_index())
