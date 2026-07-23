"""
providers.py — Provider abstraction for OpenAI, Anthropic, Ollama, and Groq.

Each provider implementation exposes:
    async def complete(request) -> ChatCompletionResponse
    async def stream(request) -> AsyncIterator[str]   # yields raw text deltas

routes.py picks a provider based on the `model` field prefix:
    gpt-*     -> OpenAIProvider
    claude-*  -> AnthropicProvider
    ollama/*  -> OllamaProvider
    groq/*    -> GroqProvider   (also matches known Groq-hosted model names
                                  directly, e.g. "llama-3.3-70b-versatile",
                                  even without the "groq/" prefix)

Note: Groq does not currently offer an embeddings endpoint. Semantic
similarity lookups still use OpenAI embeddings (or the mock embedding
fallback in app/embeddings.py) regardless of which provider answers the
actual chat completion — only the "miss" completion call is routed to Groq.
"""

import abc
import logging
import time
from typing import AsyncIterator, List

import httpx

from app.config import get_settings
from app.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    @abc.abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        ...


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = self._get_client()
        resp = await client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        choice = resp.choices[0]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=choice.message.content or ""),
                finish_reason=choice.finish_reason,
            )],
            usage=Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class GroqProvider(BaseProvider):
    """Groq's API is OpenAI-compatible (same /chat/completions shape), so we
    reuse the OpenAI SDK and simply point it at Groq's base_url with a Groq
    API key. Model names are passed through as-is (e.g. "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it") after
    stripping an optional "groq/" prefix if the caller included one.
    """
    name = "groq"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
        return self._client

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model[len("groq/"):] if model.startswith("groq/") else model

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = self._get_client()
        model = self._strip_prefix(request.model)
        resp = await client.chat.completions.create(
            model=model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        choice = resp.choices[0]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=choice.message.content or ""),
                finish_reason=choice.finish_reason,
            )],
            usage=Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        client = self._get_client()
        model = self._strip_prefix(request.model)
        stream = await client.chat.completions.create(
            model=model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    @staticmethod
    def _split_system(messages: List[ChatMessage]):
        system = "\n".join(m.content for m in messages if m.role == "system")
        rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return system, rest

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = self._get_client()
        system, messages = self._split_system(request.messages)
        resp = await client.messages.create(
            model=request.model,
            system=system or None,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 1024,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=resp.stop_reason,
            )],
            usage=Usage(
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        client = self._get_client()
        system, messages = self._split_system(request.messages)
        async with client.messages.stream(
            model=request.model,
            system=system or None,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 1024,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self):
        self._base_url = settings.ollama_base_url

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model[len("ollama/"):] if model.startswith("ollama/") else model

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = self._strip_prefix(request.model)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=120) as client:
            resp = await client.post("/api/chat", json={
                "model": model,
                "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                "stream": False,
                "options": {"temperature": request.temperature},
            })
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=prompt_eval,
                completion_tokens=eval_count,
                total_tokens=prompt_eval + eval_count,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        model = self._strip_prefix(request.model)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=120) as client:
            async with client.stream("POST", "/api/chat", json={
                "model": model,
                "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                "stream": True,
                "options": {"temperature": request.temperature},
            }) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    import json as _json
                    try:
                        payload = _json.loads(line)
                    except ValueError:
                        continue
                    delta = payload.get("message", {}).get("content")
                    if delta:
                        yield delta


_PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
    "groq": GroqProvider(),
}

# Known Groq-hosted model names, so requests work even if the caller doesn't
# prefix them with "groq/". Not exhaustive — Groq adds/retires models
# periodically; check https://console.groq.com/docs/models for the current
# list. Using the explicit "groq/" prefix is the more future-proof option.
_KNOWN_GROQ_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
}


def resolve_provider(model: str) -> BaseProvider:
    """Route a model string to the correct provider implementation."""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return _PROVIDERS["openai"]
    if model.startswith("claude-"):
        return _PROVIDERS["anthropic"]
    if model.startswith("ollama/"):
        return _PROVIDERS["ollama"]
    if model.startswith("groq/") or model in _KNOWN_GROQ_MODELS:
        return _PROVIDERS["groq"]

    # Sensible default: OpenAI, unless the operator has configured Groq as
    # the fallback (useful if Groq is the only key on hand).
    default = "groq" if settings.default_to_groq else "openai"
    logger.warning("Unrecognized model prefix '%s'; defaulting to %s provider", model, default)
    return _PROVIDERS[default]
