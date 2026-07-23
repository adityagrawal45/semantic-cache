"""
embeddings.py — Wrapper around the embedding provider.

Uses OpenAI's text-embedding-3-small by default. Falls back to a
deterministic, hash-based mock embedding when no API key is configured
or USE_MOCK_EMBEDDINGS=true, so the service (and its tests / load tests)
can run fully offline.
"""

import hashlib
import logging
from typing import List

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client = None


def _get_openai_client():
    """Lazily construct the OpenAI client so importing this module never
    requires an API key to be present (important for mock/test mode)."""
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _mock_embedding(text: str, dim: int = None) -> List[float]:
    """Deterministic pseudo-embedding derived from a SHA256 hash of the text.

    Not semantically meaningful, but stable and dimension-correct, which is
    enough to exercise the full caching pipeline (exact repeats will hit;
    near-duplicates won't unless byte-identical) in environments without a
    real embedding API key, e.g. CI or local dev.
    """
    dim = dim or settings.embedding_dim
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand the 32-byte digest deterministically to `dim` floats in [-1, 1].
    rng = np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))
    vec = rng.uniform(-1, 1, size=dim)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


async def embed_text(text: str) -> List[float]:
    """Return an embedding vector for `text`.

    Falls back to a mock embedding if mock mode is enabled or no API key
    is configured, so the rest of the pipeline degrades gracefully instead
    of hard-failing.
    """
    if settings.use_mock_embeddings or not settings.openai_api_key:
        logger.debug("Using mock embedding (mock_mode=%s, has_key=%s)",
                     settings.use_mock_embeddings, bool(settings.openai_api_key))
        return _mock_embedding(text)

    try:
        client = _get_openai_client()
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding
    except Exception as exc:  # noqa: BLE001 - we want a resilient fallback
        logger.warning("Embedding API call failed (%s); falling back to mock embedding", exc)
        return _mock_embedding(text)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. Used mainly in tests
    and offline analysis; the live query path relies on RedisVL's own
    vector search for performance."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
