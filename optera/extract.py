"""Stage 2 - per-class extraction, plus validator-driven escalation.

The policy table in config.CLASS_POLICY decides model and resolution per class.
Escalation is the safety valve that makes an aggressive default defensible: we
start cheap, verify deterministically, and only buy a stronger model for the
documents that actually failed verification.
"""
from __future__ import annotations

from typing import Any

from . import imaging, jsonio, prompts, schemas, validate
from .config import (CLASS_POLICY, DEGRADED_RES_BOOST, ESCALATION_THRESHOLD,
                     MAX_USEFUL_DIM, ClassPolicy)
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision

# Anthropic cache floors in tokens. Below these, cache_control is accepted and
# silently ignored — the request succeeds but nothing is cached.
# Haiku: 2048 tokens. Sonnet/Opus: 1024 tokens.
# RULEBOOK_CORE (~400 tokens) is below both floors — caching was silently doing
# nothing on extraction calls before rich_rulebook() was introduced.
# rich_rulebook() is ~2100 tokens, above both floors.
_HAIKU_CACHE_FLOOR_TOKENS = 2048
_SONNET_OPUS_CACHE_FLOOR_TOKENS = 1024
_APPROX_TOKENS_PER_CHAR = 4  # conservative; actual is closer to 3.8


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


def _cache_floor_tokens(model: str) -> int:
    """Return the minimum cached-prefix length in tokens for this model."""
    if "haiku" in model.lower():
        return _HAIKU_CACHE_FLOOR_TOKENS
    return _SONNET_OPUS_CACHE_FLOOR_TOKENS


def _one_pass(pf: PreflightResult, doc_class: str, model: str, max_dim: int,
              jpeg_q: int, max_tokens: int, ledger: Ledger, stage: str,
              subtype: str | None = None) -> tuple[dict | None, str, float]:
    media, b64, nbytes = imaging.encode(pf.image, max_dim=max_dim, quality=jpeg_q)

    # Use the rich rulebook (core + all few-shot examples). At ~2100 tokens it
    # exceeds the cache floor for every model tier, so prompt caching fires from
    # the second call onwards. The examples also directly improve accuracy on the
    # three failure modes they demonstrate.
    system = prompts.rich_rulebook()
    floor = _cache_floor_tokens(model)
    system_tokens_approx = len(system) // _APPROX_TOKENS_PER_CHAR
    should_cache = system_tokens_approx >= floor

    resp = call_vision(
        model=model,
        system=system,
        text=prompts.extraction_prompt(doc_class, pf.warnings, subtype=subtype),
        images=[(media, b64)],
        max_tokens=max_tokens,
        cache_system=should_cache,
        temperature=0.0,
    )
    ledger.record(pf.doc_id, stage, model, resp.usage,
                  image_px=str(max_dim), image_kb=nbytes / 1024, note=doc_class)
    obj, note = jsonio.parse(resp.text)
    return obj, note, resp.usage.output_tokens


def extract(pf: PreflightResult, doc_class: str, ledger: Ledger,
            allow_escalation: bool = True, subtype: str | None = None) -> dict[str, Any]:
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
        # An unparseable response is a strong escalation signal in its own right.
        if allow_escalation and pol.escalate_to:
            return _escalate(pf, doc_class, pol, ledger, env, reason="parse_failed", subtype=subtype)
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

    if allow_escalation and pol.escalate_to and (conf < ESCALATION_THRESHOLD or not rep.passed):
        return _escalate(pf, doc_class, pol, ledger, env,
                         reason=f"conf={conf}|{';'.join(rep.issues[:3])}", subtype=subtype)
    return env


def _escalate(pf: PreflightResult, doc_class: str, pol: ClassPolicy, ledger: Ledger,
              prev_env: dict, reason: str, subtype: str | None = None) -> dict[str, Any]:
    """Second pass on a stronger model at full resolution.

    Deliberately changes two variables at once (model and pixels). When the
    cheap pass has already failed verification the goal is to be right, not to
    run a clean ablation.
    """
    model = pol.escalate_to
    max_dim = MAX_USEFUL_DIM
    obj, note, _ = _one_pass(pf, doc_class, model, max_dim, 88,
                             int(pol.max_tokens * 1.3), ledger, "escalate", subtype=subtype)

    if obj is None:
        prev_env["provenance"]["stages"].append(f"escalate:{model}:parse_failed")
        prev_env["provenance"]["escalated"] = True
        prev_env["provenance"]["escalation_reason"] = reason
        return prev_env

    if str(obj.get("doc_class")) == "not_a_document" or (obj.get("refusal") and obj.get("data") is None):
        ref = obj.get("refusal") or {}
        out = schemas.refusal(pf.doc_id, ref.get("reason", "not_a_document"),
                              ref.get("observed", ""),
                              doc_class=str(obj.get("doc_class") or "not_a_document"),
                              confidence=float(obj.get("confidence", 0.8) or 0.8))
        out["quality_flags"] = list(pf.warnings)
        out["provenance"]["stages"] = prev_env["provenance"]["stages"] + [f"escalate:{model}:refused"]
        out["provenance"]["escalated"] = True
        return out

    data = obj.get("data")
    if isinstance(data, dict):
        for rowfield in schemas.COMPACT_COLUMNS:
            data, _n = schemas.expand_compact(data, rowfield)
    rep = validate.validate(doc_class, data if isinstance(data, dict) else None)
    conf = validate.effective_confidence(float(obj.get("confidence", 0.6) or 0.6), rep)

    env = schemas.empty_envelope(pf.doc_id)
    env.update({
        "doc_class": doc_class, "status": "extracted", "confidence": conf,
        "data": data, "validation": rep.to_json(),
        "quality_flags": list(pf.warnings),
    })
    env["provenance"]["stages"] = prev_env["provenance"]["stages"] + [f"escalate:{model}@{max_dim}px"]
    env["provenance"]["escalated"] = True
    env["provenance"]["escalation_reason"] = reason
    return env
