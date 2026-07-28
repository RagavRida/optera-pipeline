"""Canonical output schemas.

One envelope for every image regardless of what it turns out to be, so a
consumer never has to branch on success before it can branch on type. The
per-class field specs below are the single source of truth: the extraction
prompt is *generated* from them (see prompt_spec), which means a schema change
cannot silently leave a stale prompt behind.

Field naming follows the vocabulary the documents themselves use (bus_no, GSTIN,
DEF) rather than a generic invoice ontology, because the operators reading this
output speak that vocabulary.
"""
from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------
# Field spec format: name -> (type, required, description)
# type is a hint for the model and a check for the validator.
# --------------------------------------------------------------------------

WORK_REPORT_FIELDS: dict[str, tuple[str, bool, str]] = {
    "depot": ("string|null", False,
              "Depot/branch printed in the form header, e.g. VADAJ, PALDI, SARANGPUR. null if the page is a plain notebook sheet with no header."),
    "report_date": ("date|null", True,
                    "Date of the log in strict YYYY-MM-DD. Indian forms are day-first: 19-06-25 means 2025-06-19. null if genuinely absent."),
    "entries": ("array<work_entry>", True,
                "One object per work line. Include every line, in page order. Do NOT merge or summarise lines."),
    "page_label": ("string|null", False, "Any page marker such as 'Page 1'."),
}

WORK_ENTRY_FIELDS: dict[str, tuple[str, bool, str]] = {
    "sr_no": ("string|null", False, "Serial/SR column value as written."),
    "mechanic": ("string|null", False, "Mechanic name or code from the MECH column."),
    "bus_no": ("string|null", False,
               "Vehicle code from the BUS NO column, e.g. TCM35, MMM43, MAC-1. Preserve exactly as written including letter case and hyphens."),
    "work_done": ("string", True,
                  "The work description verbatim. Keep the original language and script (English/Hindi/Gujarati) - transliterate nothing."),
    "material": ("string|null", False, "MATERIAL column content if present."),
    "struck_through": ("boolean", False,
                       "true if the line is crossed out. Cancelled work still gets a row, flagged - deleting it loses an audit trail."),
}

VENDOR_BILL_FIELDS: dict[str, tuple[str, bool, str]] = {
    "vendor_name": ("string|null", True, "Trading name in the printed letterhead."),
    "vendor_gstin": ("string|null", False, "15-character GSTIN if printed."),
    "vendor_phone": ("string|null", False, "Contact number if printed."),
    "invoice_no": ("string|null", True, "Bill/invoice number, often handwritten."),
    "invoice_date": ("date|null", True, "YYYY-MM-DD, day-first source format."),
    "buyer_name": ("string|null", False, "Customer the bill is made out to, e.g. DTC."),
    "vehicle_no": ("string|null", False,
                   "Registration number, e.g. GJ-06-AV-4045. Preserve the operator's spacing/hyphenation."),
    "line_items": ("array<line_item>", True, "Every billed line. Empty array if none are legible."),
    "subtotal": ("number|null", False, "Pre-tax total if separately printed."),
    "tax_amount": ("number|null", False, "Total GST/CGST/SGST if shown."),
    "total_amount": ("number|null", True,
                     "Final payable amount. This is the highest-value field on the document - if it is illegible say null, never guess."),
    "amount_in_words": ("string|null", False, "Written-out amount if present, useful as a cross-check on total_amount."),
    "currency": ("string", False, "ISO code; INR unless the document says otherwise."),
}

LINE_ITEM_FIELDS: dict[str, tuple[str, bool, str]] = {
    "description": ("string", True, "Item text as written."),
    "hsn": ("string|null", False, "HSN/SAC code if printed."),
    "qty": ("number|null", False, "Quantity."),
    "rate": ("number|null", False, "Unit rate."),
    "amount": ("number|null", False, "Line amount."),
}

METER_READING_FIELDS: dict[str, tuple[str, bool, str]] = {
    "reading_type": ("enum<odometer,def_dispenser,fuel_dispenser,other>", True,
                     "odometer for an instrument cluster; def_dispenser for an AdBlue/DEF pump display."),
    "odometer_km": ("number|null", False,
                    "Total distance shown against ODO. Transcribe every digit including any decimal - 32065.4 is not 320654."),
    "trip_km": ("number|null", False, "TRIP A/B value if shown."),
    "fuel_level_pct": ("number|null", False, "Only if shown numerically."),
    "amount_rs": ("number|null", False, "Rupee total on a dispenser display."),
    "litres": ("number|null", False, "Litres dispensed."),
    "rate_per_litre": ("number|null", False, "Rs per litre."),
    "urea_concentration_pct": ("number|null", False, "%% urea on a DEF pump."),
    "vehicle_no": ("string|null", False, "Only if visibly present in the frame."),
    "captured_at": ("datetime|null", False, "Only if the photo carries a visible timestamp overlay."),
}

CLASS_FIELDS = {
    "work_report": WORK_REPORT_FIELDS,
    "vendor_bill": VENDOR_BILL_FIELDS,
    "meter_reading": METER_READING_FIELDS,
}

# Subtype -> the only fields that can legitimately appear on that device.
#
# This exists because of a measured failure, not a theory. Asked for all ten
# meter fields at once, the model put the dashboard's "AFE 3.4 km/l" fuel
# efficiency into rate_per_litre and fabricated a date from the clock. Unused
# schema fields are not free: they are distractors that invite contamination.
# Narrowing the field list cuts prompt tokens AND removes the failure mode.
SUBTYPE_FIELDS: dict[str, list[str]] = {
    "odometer": ["reading_type", "odometer_km", "trip_km", "fuel_level_pct",
                 "vehicle_no", "captured_at"],
    "dispenser": ["reading_type", "amount_rs", "litres", "rate_per_litre",
                  "urea_concentration_pct", "vehicle_no", "captured_at"],
}

# Vehicle codes are a closed-ish vocabulary: a three-letter fleet prefix plus a
# number. Supplying the SHAPE (not a whitelist of values) fixed a systematic
# confusion between visually similar prefixes without overfitting to the 47
# starter images - an unseen prefix still parses.
BUS_CODE_HINT = (
    "Vehicle codes in these logs are a 3-letter fleet prefix followed by a "
    "1-2 digit number, e.g. TAM17, MAM43, TCM06, MAC31, TPM13; a few carry an "
    "'-Ex' suffix such as TCM-Ex1. The prefix letters are commonly confused: "
    "read the first letter carefully (T vs M) and the middle letter carefully "
    "(A vs C vs M vs P). Transcribe the code exactly as written; if a letter is "
    "genuinely ambiguous, still give your best single reading rather than null."
)
NESTED_FIELDS = {
    "work_entry": WORK_ENTRY_FIELDS,
    "line_item": LINE_ITEM_FIELDS,
}

CLASS_DESCRIPTIONS = {
    "work_report": "A mechanic's daily work log: ruled/printed form or plain notebook page listing repairs per vehicle.",
    "vendor_bill": "A supplier invoice, cash memo or bill of supply from a parts/tyre/greasing/service shop.",
    "meter_reading": "A photograph of a display showing readings: vehicle instrument cluster or a DEF/fuel dispenser.",
    "not_a_document": "A photograph of an object or scene carrying no structured record: a battery, a tyre, a part, a number plate, a person, a blank page.",
}


def prompt_spec(doc_class: str, subtype: str | None = None, compact: bool = True) -> str:
    """Render the field spec as compact prompt text.

    Compact on purpose: this text is resent on every extraction call, so its
    length is a recurring cost. Verbose JSON Schema would roughly triple it for
    no measured accuracy gain.

    When subtype is given, only the fields valid for that device are emitted.
    """
    lines: list[str] = []
    fields = CLASS_FIELDS[doc_class]
    keep = SUBTYPE_FIELDS.get(subtype or "")
    if keep:
        fields = {k: v for k, v in fields.items() if k in keep}

    def render(name: str, spec: dict[str, tuple[str, bool, str]], indent: str = "") -> None:
        lines.append(f"{indent}{name}:")
        for fname, (ftype, required, desc) in spec.items():
            flag = "REQUIRED" if required else "optional"
            lines.append(f"{indent}  {fname} <{ftype}> [{flag}] {desc}")

    render(doc_class, fields)
    if compact:
        for rowfield in COMPACT_COLUMNS:
            if rowfield in fields:
                lines.append("")
                lines.append(compact_spec(rowfield))
    else:
        # Verbose object rows: what you write before you start counting tokens.
        for nested, nspec in NESTED_FIELDS.items():
            if any(f"<{nested}>" in t for t, _, _ in fields.values()):
                render(nested, nspec, indent="  ")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Compact wire format for repeating rows.
#
# Repeating a JSON key on every row is pure output cost. A 40-row work report
# spends roughly 30 tokens per row restating "sr_no"/"mechanic"/"bus_no"/...
# before a single character of content. Emitting positional arrays and
# re-inflating them locally is free on our side and cuts output tokens on the
# largest and most expensive class in the corpus.
#
# The canonical schema the caller receives is unchanged - only the wire format
# the model produces is compressed.
# --------------------------------------------------------------------------
COMPACT_COLUMNS: dict[str, list[str]] = {
    "entries": ["sr_no", "mechanic", "bus_no", "work_done", "material", "struck_through"],
    "line_items": ["description", "hsn", "qty", "rate", "amount"],
}


def compact_spec(field: str) -> str:
    cols = COMPACT_COLUMNS[field]
    return (f'"{field}" is an array of ARRAYS, not of objects. Each inner array has '
            f'exactly {len(cols)} positions in this fixed order:\n  '
            + " | ".join(f"[{i}] {c}" for i, c in enumerate(cols))
            + "\nUse null for any position you cannot read. Emit every row; do not "
              "collapse or summarise rows. Example of one row:\n  "
            + json.dumps(_EXAMPLE_ROW[field], ensure_ascii=False))


_EXAMPLE_ROW = {
    "entries": ["03", None, "TAM17", "brake pedal valve change", None, 0],
    "line_items": ["Puncture Fitting", None, 1, 150, 150],
}


def expand_compact(data: dict, field: str) -> tuple[dict, int]:
    """Inflate positional rows back into the canonical object form.

    Tolerant by design: short rows pad with null, long rows truncate, and rows
    that already arrived as objects pass through untouched. A model that
    ignores the compact instruction must not cost us the document.
    """
    rows = data.get(field)
    if not isinstance(rows, list):
        return data, 0
    cols = COMPACT_COLUMNS[field]
    out, converted = [], 0
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
            continue
        if isinstance(row, list):
            vals = (list(row) + [None] * len(cols))[: len(cols)]
            obj = dict(zip(cols, vals))
            if "struck_through" in obj:
                obj["struck_through"] = bool(obj["struck_through"])
            out.append(obj)
            converted += 1
    data[field] = out
    return data, converted


def empty_envelope(doc_id: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_class": None,
        "status": "error",
        "confidence": 0.0,
        "data": None,
        "refusal": None,
        "quality_flags": [],
        "validation": {"passed": True, "issues": []},
        "provenance": {"calls": [], "cost_usd": 0.0, "stages": []},
    }


def refusal(doc_id: str, reason: str, observed: str = "", doc_class: str = "not_a_document",
            confidence: float = 1.0) -> dict[str, Any]:
    """A refusal is a first-class result, not an error.

    Structured refusal is what stops a battery photo becoming an invoice row.
    """
    env = empty_envelope(doc_id)
    env.update({
        "doc_class": doc_class, "status": "refused", "confidence": confidence,
        "data": None, "refusal": {"reason": reason, "observed": observed},
    })
    return env
