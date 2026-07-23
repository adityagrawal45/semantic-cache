"""
near_miss_analyzer.py — Logs "near miss" queries: cache lookups whose top
similarity score fell just below the active threshold. Reviewing these
periodically is how you tell whether the threshold is too strict (lots of
near misses that a human would call "the same question") or too loose.

Logs are appended as JSON Lines to `settings.near_miss_log_path` so they
can be fed directly into threshold_tuner.py or any offline notebook.
"""

import json
import logging
import os
import time
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _ensure_log_dir() -> None:
    log_dir = os.path.dirname(settings.near_miss_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def log_near_miss(
    prompt_text: str,
    best_score: float,
    threshold: float,
    model: str,
    matched_cache_key: Optional[str] = None,
) -> None:
    """Record a near-miss event if the best score falls in the "interesting"
    band between near_miss_lower_bound and the active threshold.
    """
    if not (settings.near_miss_lower_bound <= best_score < threshold):
        return

    record = {
        "timestamp": time.time(),
        "prompt_text": prompt_text,
        "best_score": best_score,
        "threshold": threshold,
        "model": model,
        "matched_cache_key": matched_cache_key,
    }

    try:
        _ensure_log_dir()
        with open(settings.near_miss_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write near-miss log: %s", exc)


def load_near_misses(limit: Optional[int] = None) -> list:
    """Read back logged near-miss records (most recent last)."""
    if not os.path.exists(settings.near_miss_log_path):
        return []
    with open(settings.near_miss_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[-limit:] if limit else records
