"""
models.py — Pydantic request/response schemas.

These mirror the OpenAI Chat Completions API shape closely enough that
existing OpenAI client SDKs can point at this service with only a
base_url change.
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None

    # --- Extensions specific to this cache layer ---
    # Per-request override of the similarity threshold, e.g. 0.90.
    x_threshold: Optional[float] = Field(default=None, alias="x_threshold")
    # Explicit request "type" hint (e.g. "factual", "creative", "classification")
    # used by the adaptive-threshold logic when the caller doesn't pass x_threshold.
    x_request_type: Optional[str] = Field(default=None, alias="x_request_type")
    # Bypass the cache entirely for this call.
    x_no_cache: bool = Field(default=False, alias="x_no_cache")

    class Config:
        populate_by_name = True


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)

    # --- Cache-specific metadata (also mirrored in response headers) ---
    cache_hit: bool = False
    similarity_score: Optional[float] = None
    cache_key: Optional[str] = None


class StreamChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


class InvalidateRequest(BaseModel):
    system_hash: Optional[str] = None
    model: Optional[str] = None
    tag: Optional[str] = None


class InvalidateResponse(BaseModel):
    deleted_count: int
    matched_criteria: Dict[str, Any]


class ThresholdSimulationResult(BaseModel):
    threshold: float
    simulated_hit_rate: float
    simulated_requests: int
    simulated_hits: int
    estimated_cost_savings_usd: float
