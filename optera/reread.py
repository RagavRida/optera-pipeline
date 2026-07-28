"""Opt-in, validation-gated crop re-reads for high-value ambiguous fields."""
from __future__ import annotations

from typing import Any

from . import imaging, jsonio, prompts, validate
from .config import CLASS_POLICY
from .ledger import Ledger
from .preflight import PreflightResult
from .providers import call_vision


def _crop_box(doc_class: str, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Conservative layout-independent crops; none is better than a guessed crop."""
    width, height = size
    if doc_class == "vendor_bill":
        # Letterhead and invoice identity fields overwhelmingly live above the
        # item table; this avoids paying to re-read the full page for a GSTIN.
        return (0, 0, width, max(1, int(height * 0.42)))
    if doc_class == "meter_reading":
        # Remove phone chrome/ground while preserving the instrument display.
        return (0, int(height * 0.10), width, int(height * 0.88))
    return None


def _fields_for(doc_class: str, issues: list[str]) -> list[str]:
    text = " ".join(issues)
    if doc_class == "vendor_bill" and "malformed_gstin" in text:
        return ["vendor_gstin"]
    if doc_class == "meter_reading" and (
            "suspected_dropped_decimal" in text or "dispenser_math_mismatch" in text):
        return ["odometer_km", "trip_km", "amount_rs", "litres", "rate_per_litre"]
    return []


def _accept_if_improved(doc_class: str, original: dict, patch: dict) -> tuple[dict, bool]:
    """Merge only requested non-null fields and only if validation improves."""
    before = validate.validate(doc_class, original)
    candidate = dict(original)
    for key, value in patch.items():
        if value is not None:
            candidate[key] = value
    after = validate.validate(doc_class, candidate)
    return (candidate, True) if after.severity < before.severity else (original, False)


def reread_if_needed(pf: PreflightResult, doc_class: str, data: dict | None,
                     report: validate.ValidationReport, ledger: Ledger) -> tuple[dict | None, validate.ValidationReport, bool]:
    """Run one narrow re-read when it can fix a known, checkable failure.

    This never runs for a clean extraction and never trusts a crop merely because
    the model says it is confident; deterministic validation must improve.
    """
    if not data or report.passed:
        return data, report, False
    fields = _fields_for(doc_class, report.issues)
    box = _crop_box(doc_class, pf.image.size)
    if not fields or box is None:
        return data, report, False

    crop = pf.image.crop(box)
    pol = CLASS_POLICY[doc_class]
    media, b64, nbytes = imaging.encode(crop, max_dim=pol.max_dim, quality=pol.jpeg_q)
    task = (
        f"Re-read only these fields from this cropped {doc_class}: {', '.join(fields)}. "
        "Return exactly {\"data\":{...},\"confidence\":<0-1>}. "
        "Use null when the crop does not show a field; never infer missing digits."
    )
    response = call_vision(pol.model, prompts.RULEBOOK, task, [(media, b64)], max_tokens=180, temperature=0.0)
    ledger.record(pf.doc_id, "reread_crop", pol.model, response.usage,
                  image_px=f"{crop.width}x{crop.height}", image_kb=nbytes / 1024,
                  note=f"{doc_class}:{','.join(fields)}")
    obj, _ = jsonio.parse(response.text)
    patch = (obj or {}).get("data")
    if not isinstance(patch, dict):
        return data, report, False
    merged, accepted = _accept_if_improved(doc_class, data, patch)
    return merged, validate.validate(doc_class, merged), accepted
