"""
config.py — Centralized application settings.

All configuration is loaded from environment variables (optionally via a
.env file in development). Nothing here should ever contain a hardcoded
secret; see .env.example for the full list of expected variables.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---
    app_name: str = "semantic-cache"
    log_level: str = "INFO"
    api_port: int = 8000

    # --- Redis / RedisVL ---
    redis_url: str = "redis://redis:6379"
    redis_index_name: str = "llm_cache_idx"
    redis_prefix: str = "cache:"
    embedding_dim: int = 1536  # text-embedding-3-small

    # --- Similarity / caching behavior ---
    similarity_threshold: float = 0.95
    min_similarity_threshold: float = 0.80
    max_similarity_threshold: float = 0.999

    # --- TTL policy (seconds) ---
    ttl_short_seconds: int = 3600        # 1 hour — time-sensitive content
    ttl_long_seconds: int = 86400        # 24 hours — factual/stable content
    ttl_default_seconds: int = 21600     # 6 hours — fallback

    # --- Provider credentials ---
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://ollama:11434"
    groq_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # If true, any model that doesn't match a gpt-/claude-/ollama-/groq- prefix
    # falls back to Groq instead of OpenAI (handy if Groq is your only key).
    default_to_groq: bool = False

    # --- Embeddings ---
    embedding_model: str = "text-embedding-3-small"
    use_mock_embeddings: bool = False  # set True for local dev / CI without an API key

    # --- Cost estimation (USD per 1K tokens), used only for savings metrics ---
    cost_per_1k_input_tokens: float = 0.0005
    cost_per_1k_output_tokens: float = 0.0015

    # --- Prometheus ---
    metrics_path: str = "/metrics"

    # --- Near-miss logging ---
    near_miss_log_path: str = "logs/near_misses.jsonl"
    near_miss_lower_bound: float = 0.80  # log queries scoring between this and threshold


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse the environment only once."""
    return Settings()
