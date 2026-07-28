"""Safe, content-addressed cache for already-validated optimized results."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import prompts, schemas
from .config import CLASS_POLICY, EXTRACT_MODEL, ROUTER_MODEL
from .preflight import PreflightResult

CACHE_VERSION = "openai-optimized-v1"
MIN_EXTRACT_CONFIDENCE = 0.80
MIN_REFUSAL_CONFIDENCE = 0.95


def _fingerprint() -> str:
    """Hash every prompt/model/schema choice that can change an answer."""
    policy = {
        name: {"model": item.model, "max_dim": item.max_dim,
               "jpeg_q": item.jpeg_q, "max_tokens": item.max_tokens}
        for name, item in CLASS_POLICY.items()
    }
    material = {
        "version": CACHE_VERSION,
        "router_model": ROUTER_MODEL,
        "extract_model": EXTRACT_MODEL,
        "rulebook": prompts.RULEBOOK,
        "router_prompt": prompts.classification_prompt(),
        "schemas": {name: schemas.prompt_spec(name) for name in CLASS_POLICY},
        "policy": policy,
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class ResultCache:
    """Cache only high-confidence exact-image results.

    The cache is intentionally conservative: near-duplicate perceptual hashes
    are never cache keys, and a prompt/model/schema change creates a new cache
    namespace automatically.
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.fingerprint = _fingerprint()

    def _path(self, sha256: str) -> Path:
        return self.root / self.fingerprint / f"{sha256}.json"

    @staticmethod
    def cacheable(env: dict[str, Any]) -> bool:
        if env.get("status") == "extracted":
            return bool(env.get("validation", {}).get("passed")) and float(env.get("confidence", 0)) >= MIN_EXTRACT_CONFIDENCE
        if env.get("status") == "refused":
            return float(env.get("confidence", 0)) >= MIN_REFUSAL_CONFIDENCE
        return False

    def get(self, pf: PreflightResult) -> dict[str, Any] | None:
        if not pf.sha256:
            return None
        path = self._path(pf.sha256)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if stored.get("sha256") != pf.sha256 or not self.cacheable(stored.get("result", {})):
            return None

        env = copy.deepcopy(stored["result"])
        env["doc_id"] = pf.doc_id
        env["provenance"] = {"calls": [], "cost_usd": 0.0, "stages": ["cache:exact_validated_hit"]}
        return env

    def put(self, pf: PreflightResult, env: dict[str, Any]) -> bool:
        if not pf.sha256 or not self.cacheable(env):
            return False
        path = self._path(pf.sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sha256": pf.sha256, "fingerprint": self.fingerprint, "result": env}
        fd, tmp = tempfile.mkstemp(prefix=".optera-cache-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True
