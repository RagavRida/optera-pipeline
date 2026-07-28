"""Small OpenAI vision client with API-reported token accounting."""
from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import model_info

logger = logging.getLogger(__name__)
_UA = "optera-pipeline/1.0 (+https://github.com/)"
API_TIMEOUT = int(os.environ.get("OPTERA_API_TIMEOUT", "240"))
API_RETRIES = int(os.environ.get("OPTERA_API_RETRIES", "5"))
API_MAX_BACKOFF = int(os.environ.get("OPTERA_API_MAX_BACKOFF", "20"))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    model: str
    raw: dict = field(default_factory=dict, repr=False)


class ProviderError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"user-agent": _UA, **headers})
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        return json.loads(response.read())


def _post_retry(url: str, payload: dict, headers: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(API_RETRIES):
        try:
            return _post(url, payload, headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            last_error = ProviderError(f"HTTP {exc.code}: {detail}")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = ProviderError(f"{type(exc).__name__}: {exc}")
        wait = min(2 ** attempt, API_MAX_BACKOFF) + random.random() * 1.5
        logger.warning("OpenAI request failed; retry %d/%d in %.1fs", attempt + 1, API_RETRIES, wait)
        time.sleep(wait)
    raise last_error or ProviderError("exhausted retries")


def call_vision(model: str, system: str, text: str,
                images: list[tuple[str, str]] | None = None,
                max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
    """Send images plus a JSON-only task to OpenAI Chat Completions."""
    if model_info(model)["provider"] != "openai":
        raise ProviderError(f"Only OpenAI models are supported; got {model!r}")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set (see openai.env.example)")

    parts = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        for mime, b64 in (images or [])
    ]
    parts.append({"type": "text", "text": text})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    started = time.time()
    data = _post_retry(f"{base}/chat/completions", payload, {
        "content-type": "application/json", "authorization": f"Bearer {key}",
    })
    usage = data.get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    choices = data.get("choices") or [{}]
    return LLMResponse(
        text=(choices[0].get("message") or {}).get("content") or "",
        model=model,
        raw=data,
        usage=Usage(
            input_tokens=max((usage.get("prompt_tokens", 0) or 0) - cached, 0),
            output_tokens=usage.get("completion_tokens", 0) or 0,
            cache_read_tokens=cached,
            latency_s=time.time() - started,
        ),
    )


def provider_available(model: str) -> bool:
    return model_info(model)["provider"] == "openai" and bool(os.environ.get("OPENAI_API_KEY"))
