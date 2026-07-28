"""Provider-agnostic vision client.

One `call_vision()` signature over Anthropic and OpenAI so that model routing is
a pricing decision rather than a rewrite. Deliberately uses only the stdlib -
no vendor SDKs - so `pip install` stays tiny and the repo runs anywhere.

Every call returns a normalised Usage with API-reported token counts. Nothing in
this codebase estimates tokens; estimated tokens make cost claims unfalsifiable.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from .config import model_info

# urllib's default User-Agent is blocked by the Cloudflare edge in front of some
# gateways (HTTP 1010). A conventional UA avoids a confusing hard failure.
_UA = "optera-pipeline/1.0 (+https://github.com/) curl/8.0.1"

# Configurable retry/timeout defaults for production environments.
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

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.latency_s + other.latency_s,
        )


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    model: str
    raw: dict = field(default_factory=dict, repr=False)


class ProviderError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict, timeout: int | None = None) -> dict:
    if timeout is None:
        timeout = API_TIMEOUT
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"user-agent": _UA, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_retry(url: str, payload: dict, headers: dict,
                attempts: int | None = None) -> dict:
    """Retry on 429/5xx with jittered exponential backoff.

    Overload responses are the norm when fanning out dozens of images
    concurrently; without this the cheap path looks unreliable rather than cheap.
    """
    if attempts is None:
        attempts = API_RETRIES
    last = None
    for i in range(attempts):
        try:
            return _post(url, payload, headers)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last = ProviderError(f"HTTP {e.code}: {detail}")
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529):
                wait = min(2 ** i, API_MAX_BACKOFF) + random.random() * 1.5
                logger.warning("HTTP %d from %s, retry %d/%d in %.1fs",
                               e.code, url.split("/")[-1], i + 1, attempts, wait)
                time.sleep(wait)
                continue
            raise last
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = ProviderError(f"{type(e).__name__}: {e}")
            wait = min(2 ** i, API_MAX_BACKOFF) + random.random() * 1.5
            logger.warning("%s on %s, retry %d/%d in %.1fs",
                           type(e).__name__, url.split("/")[-1], i + 1, attempts, wait)
            time.sleep(wait)
    raise last or ProviderError("exhausted retries")


# ---------------------------------------------------------------- Anthropic --
def _anthropic(model: str, system: str, images: list[tuple[str, str]], text: str,
               max_tokens: int, cache_system: bool, temperature: float) -> LLMResponse:
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ProviderError("ANTHROPIC_API_KEY is not set (see .env.example)")

    content: list[dict] = []
    for media_type, b64 in images:
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64}})
    content.append({"type": "text", "text": text})

    sys_block = [{"type": "text", "text": system}]
    if cache_system:
        # Only worth it above the vendor's minimum cacheable length; below that
        # the directive is silently ignored and we would report a phantom saving.
        sys_block[0]["cache_control"] = {"type": "ephemeral"}

    payload = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "system": sys_block, "messages": [{"role": "user", "content": content}],
    }
    t0 = time.time()
    data = _post_retry(f"{base}/v1/messages", payload, {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": key,
    })
    dt = time.time() - t0

    u = data.get("usage", {}) or {}
    out_text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return LLMResponse(
        text=out_text, model=model, raw=data,
        usage=Usage(
            input_tokens=u.get("input_tokens", 0) or 0,
            output_tokens=u.get("output_tokens", 0) or 0,
            cache_write_tokens=u.get("cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=u.get("cache_read_input_tokens", 0) or 0,
            latency_s=dt,
        ),
    )


# ------------------------------------------------------------------- OpenAI --
def _openai(model: str, system: str, images: list[tuple[str, str]], text: str,
            max_tokens: int, cache_system: bool, temperature: float) -> LLMResponse:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set (see .env.example)")

    parts: list[dict] = []
    for media_type, b64 in images:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{media_type};base64,{b64}"}})
    parts.append({"type": "text", "text": text})

    payload: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": parts}],
        "response_format": {"type": "json_object"},
    }
    # The reasoning-era models renamed the cap and fix temperature at 1.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    t0 = time.time()
    data = _post_retry(f"{base}/chat/completions", payload, {
        "content-type": "application/json",
        "authorization": f"Bearer {key}",
    })
    dt = time.time() - t0

    u = data.get("usage", {}) or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    choices = data.get("choices") or [{}]
    return LLMResponse(
        text=(choices[0].get("message") or {}).get("content") or "",
        model=model, raw=data,
        usage=Usage(
            # OpenAI folds cached tokens into prompt_tokens; split them out so
            # the cost function does not charge cached tokens at full rate.
            input_tokens=max((u.get("prompt_tokens", 0) or 0) - cached, 0),
            output_tokens=u.get("completion_tokens", 0) or 0,
            cache_write_tokens=0,
            cache_read_tokens=cached,
            latency_s=dt,
        ),
    )


_DISPATCH = {"anthropic": _anthropic, "openai": _openai}


def call_vision(model: str, system: str, text: str,
                images: list[tuple[str, str]] | None = None,
                max_tokens: int = 1024, cache_system: bool = False,
                temperature: float = 0.0) -> LLMResponse:
    """Single entry point for every model call in the pipeline.

    images: list of (media_type, base64) tuples.
    """
    provider = model_info(model)["provider"]
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ProviderError(f"No client implemented for provider {provider!r}")
    return fn(model, system, images or [], text, max_tokens, cache_system, temperature)


def provider_available(model: str) -> bool:
    p = model_info(model)["provider"]
    return bool(os.environ.get("ANTHROPIC_API_KEY") if p == "anthropic"
                else os.environ.get("OPENAI_API_KEY"))
