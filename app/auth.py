"""
auth.py — API key authentication.

Design:
  - Auth is entirely OFF by default (API_KEYS unset/empty), matching this
    project's "runs with zero config out of the box" philosophy from day
    one — nobody's existing local setup breaks by pulling this file in.
  - When API_KEYS is set (comma-separated), protected routes require a
    valid key via either:
      Authorization: Bearer <key>   — what every OpenAI-compatible SDK
                                       sends automatically. This means an
                                       app already pointed at this service
                                       via base_url starts authenticating
                                       with ZERO client code changes the
                                       moment you set an OpenAI-style API
                                       key on that client.
      X-API-Key: <key>              — simple alternative for curl/scripts
                                       that don't already send a Bearer
                                       token for another purpose.
  - Key comparison is constant-time (hmac.compare_digest) to avoid leaking
    timing information about how much of a guessed key was correct.
  - Not applied here: hashing/storing keys server-side, per-key rate
    limits, key expiry, or a management API to issue/revoke keys. This is
    intentionally the simplest thing that adds real access control; see
    the module docstring in routes.py for what a next iteration could add.
"""

import hmac
from functools import lru_cache
from typing import FrozenSet, Optional

from fastapi import Header, HTTPException, status

from app.config import get_settings


@lru_cache
def _configured_keys() -> FrozenSet[str]:
    """Parse API_KEYS into a set once. Cached like get_settings() itself —
    if you rotate keys, restart the service (or clear this cache) rather
    than expecting a hot reload; treating auth config as fixed-at-startup
    is a deliberate simplicity trade-off."""
    settings = get_settings()
    raw = settings.api_keys or ""
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def auth_enabled() -> bool:
    """Whether any API keys are configured. Exposed so other code (e.g. a
    startup log line, or the frontend's own status check) can report
    whether auth is active without duplicating the parsing logic."""
    return len(_configured_keys()) > 0


def _matches_any(candidate: str, valid_keys: FrozenSet[str]) -> bool:
    """Constant-time membership check. Deliberately does NOT short-circuit
    on the first match — every key is compared, so the total time taken
    doesn't vary based on which key (if any) matched, avoiding a timing
    side-channel that a naive `candidate in valid_keys` could leak."""
    matched = False
    for key in valid_keys:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency — attach via `dependencies=[Depends(require_api_key)]`
    on any route that should require authentication. Raises 401 if a key
    is required and missing/invalid; returns silently (no-op) if auth is
    disabled or the provided key is valid.
    """
    valid_keys = _configured_keys()
    if not valid_keys:
        return  # auth disabled — nothing to check

    candidate = None
    if authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
    elif x_api_key:
        candidate = x_api_key.strip()

    if not candidate or not _matches_any(candidate, valid_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or missing API key. Provide it via "
                "'Authorization: Bearer <key>' or 'X-API-Key: <key>'."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )