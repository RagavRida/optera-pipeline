"""Multi-image extraction: extract several same-class documents in one call.

Works on the same principle as batch_router: shared prompt overhead paid once.
For extraction the saving is smaller relative to image tokens (extraction images
are larger than router thumbnails), but the system prompt is also larger
(~2100 tokens for rich_rulebook), making the per-call overhead more significant.

Grouping constraints:
- Same doc_class and same subtype only. Different schemas cannot be mixed.
- Max 3 images per call to stay within context limits and keep output parseable.
- Only for classes whose output is structurally simple and bounded:
    meter_reading  (≤ 10 output fields, ~120 tokens)
    vendor_bill    (bounded line items, ~400 tokens)
  Work reports are NOT batched here — 30-40 row arrays per document, very long
  outputs that frequently exceed max_tokens when multiple are stacked.

Output: the model returns a JSON array, one object per image. Falls back to
individual extract() calls for any slot that fails to parse.
"""
from __future__ import annotations

from typing import Any

from . import imaging, jsonio, prompts, schemas, validate
from .config import CLASS_POLICY, MAX_USEFUL_DIM, ESCALATION_THRESHOLD
from .extract import _cache_floor_tokens, _APPROX_TOKENS_PER_CHAR, _escalate
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision

# Classes where multi-image batching is safe (bounded output, stable schema).
BATCHABLE_CLASSES = {"meter_reading", "vendor_bill"}
MAX_BATCH = 3


def _batch_extraction_prompt(doc_class: str, n: int,
                              subtype: str | None = None,
                              quality_flags_list: list[list[str]] | None = None) -> str:
    spec = schemas.prompt_spec(doc_class, subtype=subtype)
    hints = prompts.CLASS_HINTS.get(doc_class, "")
    flags_text = ""
    if quality_flags_list:
        readable = {
            "low_sharpness": "out of focus",
            "underexposed": "underexposed",
            "truncated_file_recovered": "truncated file",
        }
        per_image = []
        for i, flags in enumerate(quality_flags_list):
            notes = [readable[f] for f in flags if f in readable]
            if notes:
                per_image.append(f"Image {i+1}: {', '.join(notes)}")
        if per_image:
            flags_text = "IMAGE QUALITY NOTES:\n" + "\n".join(per_image) + "\n\n"

    return f"""TASK: extract {n} images. All are {doc_class} documents.
{flags_text}{hints}

SCHEMA (apply to each image)
{spec}

Return a JSON ARRAY with exactly {n} objects, one per image in order:
[
  {{"doc_class":"{doc_class}","confidence":<0-1>,"data":{{...}}}},
  ... ({n} total)
]

If any image is NOT a {doc_class}, set its data to null and add:
  "refusal":{{"reason":"misrouted","observed":"<what it shows>"}}"""


def extract_batch(pfs: list[PreflightResult], doc_class: str, ledger: Ledger,
                  subtype: str | None = None,
                  allow_escalation: bool = True) -> list[dict[str, Any]]:
    """Extract a batch of same-class images in one call. Returns one envelope per pf."""
    from . import extract as extract_mod  # avoid circular at module load

    pol = CLASS_POLICY[doc_class]
    system = prompts.rich_rulebook()
    floor = _cache_floor_tokens(pol.model)
    should_cache = (len(system) // _APPROX_TOKENS_PER_CHAR) >= floor

    # Encode all images
    encoded = []
    for pf in pfs:
        max_dim = min(pol.max_dim, MAX_USEFUL_DIM)
        if any(f in pf.warnings for f in ("low_sharpness", "underexposed")):
            max_dim = min(int(max_dim * 1.35), MAX_USEFUL_DIM)
        media, b64, nbytes = imaging.encode(pf.image, max_dim=max_dim, quality=pol.jpeg_q)
        encoded.append((media, b64, nbytes, max_dim))

    images = [(m, b) for m, b, _, _ in encoded]
    total_kb = sum(n / 1024 for _, _, n, _ in encoded)
    task = _batch_extraction_prompt(
        doc_class, len(pfs), subtype=subtype,
        quality_flags_list=[pf.warnings for pf in pfs])

    try:
        resp = call_vision(
            model=pol.model,
            system=system,
            text=task,
            images=images,
            max_tokens=pol.max_tokens * len(pfs),
            cache_system=should_cache,
            temperature=0.0,
        )
        ledger.record(
            pfs[0].doc_id, "extract_batch", pol.model, resp.usage,
            image_px=str(encoded[0][3]), image_kb=round(total_kb, 1),
            note=f"{doc_class}×{len(pfs)}",
        )
        arr, _ = jsonio.parse(resp.text)
        objs: list[dict | None] = (arr if isinstance(arr, list) else [arr]) + [None] * len(pfs)
        objs = objs[:len(pfs)]
    except Exception:
        objs = [None] * len(pfs)

    results = []
    for i, (pf, obj) in enumerate(zip(pfs, objs)):
        env = schemas.empty_envelope(pf.doc_id)
        env["quality_flags"] = list(pf.warnings)

        if not isinstance(obj, dict):
            # Batch slot failed — fall back to individual call
            results.append(extract_mod.extract(pf, doc_class, ledger,
                                               allow_escalation=allow_escalation,
                                               subtype=subtype))
            continue

        returned_class = str(obj.get("doc_class") or doc_class)
        if returned_class == "not_a_document" or (obj.get("refusal") and obj.get("data") is None):
            ref = obj.get("refusal") or {}
            out = schemas.refusal(pf.doc_id, ref.get("reason", "not_a_document"),
                                  ref.get("observed", ""), doc_class=returned_class,
                                  confidence=float(obj.get("confidence", 0.85) or 0.85))
            out["quality_flags"] = list(pf.warnings)
            out["provenance"]["stages"].append(f"extract_batch:{pol.model}:refused")
            results.append(out)
            continue

        data = obj.get("data")
        if isinstance(data, dict):
            for rf in schemas.COMPACT_COLUMNS:
                data, _ = schemas.expand_compact(data, rf)
        rep = validate.validate(doc_class, data if isinstance(data, dict) else None)
        model_conf = float(obj.get("confidence", 0.5) or 0.5)
        conf = validate.effective_confidence(model_conf, rep)

        env.update({
            "doc_class": doc_class, "status": "extracted",
            "confidence": conf, "data": data, "validation": rep.to_json(),
        })
        env["provenance"]["stages"].append(
            f"extract_batch:{pol.model}@{encoded[i][3]}px")

        if allow_escalation and pol.escalate_to and (conf < ESCALATION_THRESHOLD or not rep.passed):
            env = _escalate(pf, doc_class, pol, ledger, env,
                            reason=f"conf={conf}|{';'.join(rep.issues[:3])}",
                            subtype=subtype)

        results.append(env)

    return results
