"""
tests/test_core.py — Unit tests for the hashing, TTL classification, and
embedding utilities that don't require a live Redis/provider connection.

Run with:
    pytest tests/ -v
"""

import pytest

from app.cache_engine import hash_params, hash_system_prompt
from app.embeddings import cosine_similarity, _mock_embedding
from app.ttl_classifier import classify_request_type, classify_ttl


def test_hash_system_prompt_is_stable():
    h1 = hash_system_prompt("You are a helpful assistant.")
    h2 = hash_system_prompt("You are a helpful assistant.")
    assert h1 == h2


def test_hash_system_prompt_differs_on_change():
    h1 = hash_system_prompt("You are a helpful assistant.")
    h2 = hash_system_prompt("You are a rude assistant.")
    assert h1 != h2


def test_hash_system_prompt_empty_is_stable():
    assert hash_system_prompt("") == hash_system_prompt(None)


def test_hash_params_ignores_excluded_fields():
    p1 = {"model": "gpt-4o-mini", "temperature": 0.7, "messages": [{"role": "user", "content": "a"}], "stream": True}
    p2 = {"model": "gpt-4o-mini", "temperature": 0.7, "messages": [{"role": "user", "content": "b"}], "stream": False}
    assert hash_params(p1) == hash_params(p2)


def test_hash_params_differs_on_temperature():
    p1 = {"model": "gpt-4o-mini", "temperature": 0.7}
    p2 = {"model": "gpt-4o-mini", "temperature": 0.2}
    assert hash_params(p1) != hash_params(p2)


def test_ttl_classifier_short_for_time_sensitive():
    short_ttl = classify_ttl("What's the weather today?")
    long_ttl = classify_ttl("Explain gravity.")
    assert short_ttl < long_ttl


def test_ttl_classifier_long_for_factual():
    ttl = classify_ttl("Explain how photosynthesis works.")
    assert ttl > classify_ttl("What's the latest news right now?")


def test_request_type_classification():
    assert classify_request_type("What's the weather now?") == "time_sensitive"
    assert classify_request_type("Write a poem about the sea") == "creative"
    assert classify_request_type("Classify this review as positive or negative") == "classification"
    assert classify_request_type("Explain how gravity works") == "factual"


def test_mock_embedding_deterministic():
    v1 = _mock_embedding("hello world", dim=64)
    v2 = _mock_embedding("hello world", dim=64)
    assert v1 == v2


def test_mock_embedding_differs_for_different_text():
    v1 = _mock_embedding("hello world", dim=64)
    v2 = _mock_embedding("goodbye world", dim=64)
    assert v1 != v2


def test_cosine_similarity_identical_vectors():
    v = _mock_embedding("some text", dim=32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_like_bounds():
    v1 = _mock_embedding("apples", dim=32)
    v2 = _mock_embedding("thermodynamics", dim=32)
    sim = cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0


def test_resolve_provider_routes_groq_prefix():
    from app.providers import resolve_provider
    provider = resolve_provider("groq/llama-3.3-70b-versatile")
    assert provider.name == "groq"


def test_resolve_provider_routes_known_groq_model_without_prefix():
    from app.providers import resolve_provider
    provider = resolve_provider("llama-3.3-70b-versatile")
    assert provider.name == "groq"


def test_resolve_provider_still_routes_openai_and_anthropic():
    from app.providers import resolve_provider
    assert resolve_provider("gpt-4o-mini").name == "openai"
    assert resolve_provider("claude-3-5-sonnet-20241022").name == "anthropic"
    assert resolve_provider("ollama/llama3").name == "ollama"
