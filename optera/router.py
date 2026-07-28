"""Stage 1 - cheap classification and early refusal.

The single most valuable idea in this pipeline: deciding *what* an image is is a
far easier problem than reading it, so it should be bought at a far lower price.
A 384px thumbnail on the cheapest vision model costs roughly a quarter of a cent
per thousand images and lets us throw away the object photos before they ever
reach an extraction prompt.

Refusing here is also strictly safer than refusing later. A model that has been
handed an invoice schema is under pressure to fill it in; a model asked only
"what is this?" has no such pull.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import imaging, jsonio, prompts
from .config import ROUTER_JPEG_Q, ROUTER_MAX_DIM, ROUTER_MODEL, VALID_CLASSES
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision


@dataclass
class RouteResult:
    doc_id: str
    doc_class: str
    confidence: float
    subtype: str | None = None
    note: str = ""


def classify(pf: PreflightResult, ledger: Ledger, model: str = ROUTER_MODEL,
             max_dim: int = ROUTER_MAX_DIM) -> RouteResult:
    media, b64, nbytes = imaging.encode(pf.image, max_dim=max_dim, quality=ROUTER_JPEG_Q)

    resp = call_vision(
        model=model,
        system="You are a fast document triage classifier. Reply with JSON only.",
        text=prompts.classification_prompt(),
        images=[(media, b64)],
        max_tokens=80,          # one short object; a bigger cap invites rambling
        temperature=0.0,
    )
    ledger.record(pf.doc_id, "route", model, resp.usage,
                  image_px=f"{max_dim}", image_kb=nbytes / 1024)

    obj, note = jsonio.parse(resp.text)
    if not obj or "class" not in obj:
        # Fail *open* to the most expensive-but-safe interpretation rather than
        # silently dropping a real document: send it to the generalist path.
        return RouteResult(pf.doc_id, "work_report", 0.0,
                           note=f"router_unparsed:{note}; defaulted")

    cls = str(obj.get("class", "")).strip().lower()
    if cls not in VALID_CLASSES:
        cls = "work_report"
        note = f"router_returned_unknown_class:{obj.get('class')!r}"

    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5

    sub = obj.get("subtype")
    sub = str(sub).strip().lower() if isinstance(sub, str) and sub.strip() else None
    if cls != "meter_reading":
        sub = None
    elif sub not in ("odometer", "dispenser"):
        # Unknown subtype: fall back to the full field list rather than
        # silently dropping fields the document might legitimately need.
        sub = None

    return RouteResult(pf.doc_id, cls, max(0.0, min(1.0, conf)), subtype=sub, note=note)
