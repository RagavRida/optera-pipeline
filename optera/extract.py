"""Stage 2 - class-specific structured extraction and free validation."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from . import imaging, jsonio, prompts, schemas, validate
from .config import CLASS_POLICY, DEGRADED_RES_BOOST, MAX_USEFUL_DIM, ClassPolicy
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision

def _policy_for(doc_class: str) -> ClassPolicy:
    return CLASS_POLICY[doc_class]


def _resolution_for(pol: ClassPolicy, flags: list[str]) -> int:
    """Spend more pixels on degraded images, not on all images.

    A blurry odometer is exactly the case where downscaling turns a correct
    reading into a wrong one, so the floor rises for that document alone.
    """
    if any(f in flags for f in ("low_sharpness", "underexposed", "very_low_resolution")):
        return min(int(pol.max_dim * DEGRADED_RES_BOOST), MAX_USEFUL_DIM)
    return min(pol.max_dim, MAX_USEFUL_DIM)


def _one_pass(pf: PreflightResult, doc_class: str, model: str, max_dim: int,
              jpeg_q: int, max_tokens: int, ledger: Ledger, stage: str,
              subtype: str | None = None) -> tuple[dict | None, str, float]:
    media, b64, nbytes = imaging.encode(pf.image, max_dim=max_dim, quality=jpeg_q)

    resp = call_vision(
        model=model,
        system=prompts.RULEBOOK,
        text=prompts.extraction_prompt(doc_class, pf.warnings, subtype=subtype),
        images=[(media, b64)],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    ledger.record(pf.doc_id, stage, model, resp.usage,
                  image_px=str(max_dim), image_kb=nbytes / 1024, note=doc_class)
    obj, note = jsonio.parse(resp.text)
    return obj, note, resp.usage.output_tokens


def extract(pf: PreflightResult, doc_class: str, ledger: Ledger,
            subtype: str | None = None, targeted_reread: bool = False) -> dict[str, Any]:
    env = schemas.empty_envelope(pf.doc_id)
    env["quality_flags"] = list(pf.warnings)

    if doc_class == "not_a_document":
        return schemas.refusal(
            pf.doc_id, "not_a_document",
            "Router classified this image as an object/scene photograph carrying no structured record.")

    pol = _policy_for(doc_class)
    max_dim = _resolution_for(pol, pf.warnings)

    obj, note, _ = _one_pass(pf, doc_class, pol.model, max_dim, pol.jpeg_q,
                             pol.max_tokens, ledger, "extract", subtype=subtype)

    if obj is None:
        env["status"] = "error"
        env["doc_class"] = doc_class
        env["validation"] = {"passed": False, "issues": [f"parse_failed:{note}"], "severity": 1.0}
        env["provenance"]["stages"].append(f"extract:{pol.model}@{max_dim}px:parse_failed")
        return env

    # The extractor is allowed to overrule the router - it sees more pixels.
    returned_class = str(obj.get("doc_class") or doc_class)
    if returned_class == "not_a_document" or (obj.get("refusal") and obj.get("data") is None):
        ref = obj.get("refusal") or {}
        out = schemas.refusal(
            pf.doc_id, ref.get("reason", "not_a_document"),
            ref.get("observed", ""), doc_class=returned_class,
            confidence=float(obj.get("confidence", 0.8) or 0.8))
        out["quality_flags"] = list(pf.warnings)
        out["provenance"]["stages"].append(f"extract:{pol.model}@{max_dim}px:refused_at_extract")
        return out

    data = obj.get("data")
    if isinstance(data, dict):
        for rowfield in schemas.COMPACT_COLUMNS:
            data, _n = schemas.expand_compact(data, rowfield)
    rep = validate.validate(doc_class, data if isinstance(data, dict) else None)
    model_conf = float(obj.get("confidence", 0.5) or 0.5)
    conf = validate.effective_confidence(model_conf, rep)

    env.update({
        "doc_class": doc_class, "status": "extracted", "confidence": conf,
        "data": data, "validation": rep.to_json(),
    })
    env["provenance"]["stages"].append(f"extract:{pol.model}@{max_dim}px")
    if note:
        env["provenance"]["stages"].append(f"json:{note}")

    if targeted_reread:
        from .reread import reread_if_needed
        data, rep, accepted = reread_if_needed(pf, doc_class, data, rep, ledger)
        if accepted:
            env["data"] = data
            env["validation"] = rep.to_json()
            env["confidence"] = validate.effective_confidence(model_conf, rep)
            env["provenance"]["stages"].append("reread_crop:accepted_validation_improvement")

    return env
