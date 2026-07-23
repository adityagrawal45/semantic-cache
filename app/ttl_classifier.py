"""
ttl_classifier.py — Rule-based classifier for content-aware TTL assignment.

Time-sensitive prompts (weather, news, "today", stock prices, etc.) get a
short TTL so stale answers don't linger. Everything else defaults to a
longer TTL. This is intentionally simple and fast (regex, no model call)
since it sits on the hot path of every cache write.
"""

import re

from app.config import get_settings

settings = get_settings()

# Keywords/phrases that strongly suggest the answer is time-sensitive.
_SHORT_TTL_PATTERNS = [
    r"\btoday\b", r"\btonight\b", r"\bcurrent(ly)?\b", r"\bnow\b",
    r"\bweather\b", r"\bforecast\b", r"\bnews\b", r"\blatest\b",
    r"\bstock price\b", r"\bexchange rate\b", r"\bscore\b",
    r"\bthis week\b", r"\bright now\b", r"\bup to date\b", r"\brecent(ly)?\b",
    r"\bbreaking\b", r"\blive\b",
]

_SHORT_TTL_REGEX = re.compile("|".join(_SHORT_TTL_PATTERNS), re.IGNORECASE)


def classify_ttl(prompt_text: str) -> int:
    """Return a TTL in seconds for the given prompt text.

    - Time-sensitive prompts -> ttl_short_seconds
    - Everything else -> ttl_long_seconds

    A future iteration could swap this for a small classifier model, but a
    regex pass is sub-millisecond and easy to audit/tune.
    """
    if _SHORT_TTL_REGEX.search(prompt_text or ""):
        return settings.ttl_short_seconds
    return settings.ttl_long_seconds


def classify_request_type(prompt_text: str) -> str:
    """Rough categorization used by the adaptive-threshold logic.

    Returns one of: "time_sensitive", "factual", "creative", "classification".
    """
    text = (prompt_text or "").lower()
    if _SHORT_TTL_REGEX.search(text):
        return "time_sensitive"
    if any(k in text for k in ("classify", "categorize", "label this", "is this")):
        return "classification"
    if any(k in text for k in ("write a story", "poem", "creative", "imagine", "brainstorm")):
        return "creative"
    return "factual"
