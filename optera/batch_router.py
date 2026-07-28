"""Multi-image router: classify several thumbnails in one API call.

.. deprecated::
    This module is kept for reference and potential future use but is NOT called
    in the main pipeline. Multi-image router batching was tried and abandoned
    because it caused misclassification of tyre-service bills and DEF invoices
    as work_reports at 384px thumbnail resolution — two documents were dropped.
    See pipeline.py L144-151 for the full rationale.

The naive router sends one thumbnail per call. Each call pays the same fixed
overhead: system prompt tokens, request latency, per-request connection cost.
Sending N thumbnails in a single call pays that overhead once and gets N
classifications back — identical accuracy, N× cheaper on the overhead portion.

For the Haiku router call (system ~200 tokens + 1 image ~50 tokens + task ~200
tokens = ~450 tokens overhead per call), batching 4 images cuts overhead cost
by 4× while image tokens scale linearly. At our corpus mix the saving is
roughly 20-25% of total router spend.

The model receives a grid instruction:
  "You will see N images. For each, return exactly one JSON object."
and returns a JSON array with one element per image. The order is preserved.

If the model returns fewer elements than images (truncation, refusal), we fall
back to individual calls for the missing ones. This makes the batch layer fully
transparent to callers — they always get one RouteResult per input.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from . import imaging, jsonio
from .config import ROUTER_JPEG_Q, ROUTER_MAX_DIM, ROUTER_MODEL
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision
from .router import RouteResult, classify
from .config import VALID_CLASSES
from .schemas import CLASS_DESCRIPTIONS

_BATCH_SYSTEM = (
    "You are a fast document triage classifier. "
    "You will be shown multiple images in a single message. "
    "Reply with a JSON ARRAY containing exactly one classification object per image, "
    "in the same order as the images. Each object: "
    '{"class":"<class>","subtype":<string|null>,"confidence":<0-1>}. '
    "No other text."
)


def _batch_task(n: int) -> str:
    class_lines = "\n".join(f"- {name}: {desc}" for name, desc in CLASS_DESCRIPTIONS.items())
    return (
        f"Classify each of the {n} images below. "
        "Return a JSON array with exactly one object per image.\n\n"
        f"Classes:\n{class_lines}\n\n"
        "For meter_reading images, set subtype to 'odometer' or 'dispenser'. "
        "For all others, set subtype to null.\n\n"
        "A page of ruled handwriting is work_report. "
        "A printed letterhead with amounts is vendor_bill. "
        "A lit display of numbers is meter_reading. "
        "A physical object, part, or bare surface is not_a_document."
    )


def _parse_batch(text: str, n: int) -> list[dict | None]:
    """Parse the model's array response. Returns None for any unparseable slot."""
    arr, note = jsonio.parse(text)
    if isinstance(arr, list):
        return (arr + [None] * n)[:n]
    # If the model returned a single object instead of an array, wrap it.
    if isinstance(arr, dict):
        return [arr] + [None] * (n - 1)
    return [None] * n


def _result_from_obj(doc_id: str, obj: dict | None) -> RouteResult | None:
    if not isinstance(obj, dict) or "class" not in obj:
        return None
    cls = str(obj.get("class", "")).strip().lower()
    if cls not in VALID_CLASSES:
        cls = "work_report"
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    sub = obj.get("subtype")
    sub = str(sub).strip().lower() if isinstance(sub, str) and sub.strip() else None
    if cls != "meter_reading":
        sub = None
    elif sub not in ("odometer", "dispenser"):
        sub = None
    return RouteResult(doc_id, cls, max(0.0, min(1.0, conf)), subtype=sub)


def classify_batch(pfs: list[PreflightResult], ledger: Ledger,
                   model: str = ROUTER_MODEL, max_dim: int = ROUTER_MAX_DIM,
                   batch_size: int = 4) -> list[RouteResult]:
    """Classify a list of images, sending them in groups of batch_size.

    Falls back to individual classify() calls for any image that can't be
    resolved from a batch response.
    """
    results: list[RouteResult | None] = [None] * len(pfs)
    # Pre-build index map to avoid O(n²) pfs.index(pf) lookups.
    pf_to_idx = {id(pf): i for i, pf in enumerate(pfs)}
    groups = [pfs[i:i + batch_size] for i in range(0, len(pfs), batch_size)]

    for group in groups:
        if len(group) == 1:
            # Single image: use the regular router (no overhead saving to batch)
            pf = group[0]
            idx = pf_to_idx[id(pf)]
            results[idx] = classify(pf, ledger, model=model, max_dim=max_dim)
            continue

        # Encode all thumbnails in the group
        encoded = []
        total_kb = 0.0
        for pf in group:
            media, b64, nbytes = imaging.encode(pf.image, max_dim=max_dim, quality=ROUTER_JPEG_Q)
            encoded.append((media, b64, nbytes))
            total_kb += nbytes / 1024

        images = [(m, b) for m, b, _ in encoded]
        try:
            resp = call_vision(
                model=model,
                system=_BATCH_SYSTEM,
                text=_batch_task(len(group)),
                images=images,
                max_tokens=len(group) * 80,  # ~80 tokens per classification object
                temperature=0.0,
            )
            ledger.record(
                group[0].doc_id,   # representative doc_id for the group
                "route_batch",
                model,
                resp.usage,
                image_px=str(max_dim),
                image_kb=round(total_kb, 1),
                note=f"batch={len(group)}",
            )
            parsed = _parse_batch(resp.text, len(group))
        except Exception:
            parsed = [None] * len(group)

        for pf, obj in zip(group, parsed):
            idx = pf_to_idx[id(pf)]
            rt = _result_from_obj(pf.doc_id, obj)
            if rt is None:
                # Fallback: individual call
                rt = classify(pf, ledger, model=model, max_dim=max_dim)
            results[idx] = rt

    # Safety: any slot still None gets an individual fallback
    for i, pf in enumerate(pfs):
        if results[i] is None:
            results[i] = classify(pf, ledger, model=model, max_dim=max_dim)

    return results  # type: ignore[return-value]
