"""Small, resumable OpenAI Batch API client for independent document requests."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .providers import API_MAX_BACKOFF, API_RETRIES, API_TIMEOUT, ProviderError

_BASE = lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_UA = "optera-pipeline/1.0 (+https://github.com/)"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set (see openai.env.example)")
    return {"authorization": f"Bearer {key}", "user-agent": _UA, **(extra or {})}


def _request(method: str, url: str, body: bytes | None = None,
             headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, data=body, method=method,
                                     headers=_headers(headers))
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        return response.read()


def _request_retry(method: str, url: str, body: bytes | None = None,
                   headers: dict[str, str] | None = None) -> bytes:
    last: Exception | None = None
    for attempt in range(API_RETRIES):
        try:
            return _request(method, url, body, headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            last = ProviderError(f"HTTP {exc.code}: {detail}")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last
        except urllib.error.URLError as exc:
            last = ProviderError(f"{type(exc).__name__}: {exc}")
        time.sleep(min(2 ** attempt, API_MAX_BACKOFF) + random.random())
    raise last or ProviderError("batch request retries exhausted")


def _json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    raw = _request_retry(method, f"{_BASE()}{path}", body,
                         {"content-type": "application/json"} if body else None)
    return json.loads(raw)


def upload_jsonl(path: Path) -> dict[str, Any]:
    """Upload a JSONL request file with OpenAI's required ``batch`` purpose."""
    path = Path(path)
    boundary = f"----optera-{uuid.uuid4().hex}"
    data = path.read_bytes()
    chunks = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n").encode(),
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    raw = _request_retry("POST", f"{_BASE()}/files", b"".join(chunks), {
        "content-type": f"multipart/form-data; boundary={boundary}",
    })
    return json.loads(raw)


def create_batch(input_file_id: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    """Create a 24-hour Chat Completions batch from an uploaded JSONL file."""
    return _json("POST", "/batches", {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": metadata or {},
    })


def get_batch(batch_id: str) -> dict[str, Any]:
    return _json("GET", f"/batches/{batch_id}")


def download_file(file_id: str) -> bytes:
    return _request_retry("GET", f"{_BASE()}/files/{file_id}/content")


def submit_jsonl(path: Path, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    """Upload then submit a request JSONL file, returning durable batch state."""
    uploaded = upload_jsonl(path)
    batch = create_batch(str(uploaded["id"]), metadata=metadata)
    return {"input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch["status"]}


def collect_if_complete(batch_id: str, destination: Path) -> dict[str, Any]:
    """Persist output JSONL only after the batch completes; safe to call repeatedly."""
    batch = get_batch(batch_id)
    if batch.get("status") != "completed":
        return {"status": batch.get("status"), "batch_id": batch_id}
    output_id = batch.get("output_file_id")
    if not output_id:
        raise ProviderError(f"completed batch {batch_id} has no output_file_id")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(download_file(str(output_id)))
    return {"status": "completed", "batch_id": batch_id, "output": str(destination)}
