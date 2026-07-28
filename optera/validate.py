"""Stage 3 - deterministic validation of model output at zero token cost."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from .jsonio import coerce_number

# Values that mean "I could not read it" but are formatted as if they were data.
PLACEHOLDERS = {
    "n/a", "na", "none", "null", "unknown", "not visible", "not legible",
    "illegible", "xxx", "xxxx", "-", "--", "?", "??", "tbd", "abc", "test",
    "string", "example", "sample", "0000", "123456",
}

GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]{3}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ValidationReport:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    severity: float = 0.0     # 0 = clean, 1 = certainly wrong

    def add(self, issue: str, weight: float) -> None:
        self.issues.append(issue)
        self.severity = min(1.0, self.severity + weight)
        if weight >= 0.3:
            self.passed = False

    def to_json(self) -> dict:
        return {"passed": self.passed, "issues": self.issues,
                "severity": round(self.severity, 3)}


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in PLACEHOLDERS


def _valid_date(s: Any) -> bool:
    if not isinstance(s, str) or not DATE_RE.match(s):
        return False
    try:
        d = dt.date.fromisoformat(s)
    except ValueError:
        return False
    # Fleet paperwork is recent; a 1970 or 2049 date is a parsing artefact.
    return dt.date(2015, 1, 1) <= d <= dt.date.today() + dt.timedelta(days=2)


def validate(doc_class: str, data: dict | None) -> ValidationReport:
    rep = ValidationReport()
    if data is None:
        rep.add("no_data_object", 0.5)
        return rep

    for key, val in data.items():
        if _is_placeholder(val):
            rep.add(f"placeholder_value:{key}={val!r}", 0.25)

    if doc_class == "vendor_bill":
        _validate_bill(data, rep)
    elif doc_class == "work_report":
        _validate_work_report(data, rep)
    elif doc_class == "meter_reading":
        _validate_meter(data, rep)
    return rep


def _validate_bill(d: dict, rep: ValidationReport) -> None:
    total = coerce_number(d.get("total_amount"))
    items = d.get("line_items") or []

    if not isinstance(items, list):
        rep.add("line_items_not_a_list", 0.4)
        items = []
    if total is None:
        rep.add("missing_total_amount", 0.3)
    elif total <= 0:
        rep.add(f"non_positive_total:{total}", 0.3)
    elif total > 5_000_000:
        rep.add(f"implausible_total:{total}", 0.25)

    # Arithmetic cross-check: the strongest free signal that a digit was misread.
    line_sum = 0.0
    counted = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        amt = coerce_number(it.get("amount"))
        if amt is not None:
            line_sum += amt
            counted += 1
        qty, rate = coerce_number(it.get("qty")), coerce_number(it.get("rate"))
        if qty is not None and rate is not None and amt is not None:
            if abs(qty * rate - amt) > max(1.0, 0.02 * abs(amt)):
                rep.add(f"line_math_mismatch:{qty}x{rate}!={amt}", 0.15)

    if total is not None and counted and line_sum > 0:
        tax = coerce_number(d.get("tax_amount")) or 0.0
        # Accept either a tax-exclusive or tax-inclusive reading of the total.
        if min(abs(line_sum - total), abs(line_sum + tax - total)) > max(2.0, 0.03 * total):
            rep.add(f"total_mismatch:lines={round(line_sum, 2)} tax={tax} total={total}", 0.35)

    gstin = d.get("vendor_gstin")
    if isinstance(gstin, str) and gstin.strip() and not GSTIN_RE.match(gstin.strip().upper()):
        rep.add(f"malformed_gstin:{gstin!r}", 0.1)

    date = d.get("invoice_date")
    if date is not None and not _valid_date(date):
        rep.add(f"bad_invoice_date:{date!r}", 0.2)
    vendor = d.get("vendor_name")
    if not isinstance(vendor, str) or not vendor.strip():
        rep.add("missing_vendor_name", 0.15)


def _validate_work_report(d: dict, rep: ValidationReport) -> None:
    entries = d.get("entries")
    if not isinstance(entries, list):
        rep.add("entries_not_a_list", 0.5)
        return
    if not entries:
        # A near-blank ruled form can be valid, so this is a warning rather than
        # a hard failure.
        rep.add("zero_entries_on_work_report", 0.2)
        return

    empty = 0
    for e in entries:
        if not isinstance(e, dict):
            rep.add("entry_not_an_object", 0.2)
            continue
        work = e.get("work_done")
        if not isinstance(work, str) or not work.strip():
            empty += 1
    if empty:
        # Proportional and quiet below a quarter of the page: a few blank cells
        # should not overwhelm an otherwise useful record.
        frac = empty / max(len(entries), 1)
        if frac >= 0.5:
            rep.add(f"entries_missing_work_done:{empty}/{len(entries)}", 0.3)
        elif frac >= 0.25:
            rep.add(f"entries_missing_work_done:{empty}/{len(entries)}", 0.15)

    # Rows duplicated verbatim usually means the model lost its place in a
    # ruled table and repeated a line rather than reading the next one.
    texts = [str(e.get("work_done", "")).strip().lower()
             for e in entries if isinstance(e, dict) and e.get("work_done")]
    if len(texts) >= 4:
        dupes = len(texts) - len(set(texts))
        # Short repeated entries are genuinely common on these pages ("air fill",
        # "brake set"), so only a majority of duplicates indicates the model lost
        # its place rather than the page simply repeating itself.
        if dupes >= max(3, int(len(texts) * 0.6)):
            rep.add(f"repeated_rows:{dupes}/{len(texts)}", 0.3)

    date = d.get("report_date")
    if date is not None and not _valid_date(date):
        rep.add(f"bad_report_date:{date!r}", 0.2)


def _validate_meter(d: dict, rep: ValidationReport) -> None:
    rtype = d.get("reading_type")
    if rtype not in ("odometer", "def_dispenser", "fuel_dispenser", "other", None):
        rep.add(f"unknown_reading_type:{rtype!r}", 0.2)

    odo = coerce_number(d.get("odometer_km"))
    if odo is not None and not (0 <= odo <= 3_000_000):
        rep.add(f"implausible_odometer:{odo}", 0.35)

    amt = coerce_number(d.get("amount_rs"))
    lit = coerce_number(d.get("litres"))
    rate = coerce_number(d.get("rate_per_litre"))
    # A dispenser prints all three; they must agree or a digit was misread.
    if amt is not None and lit is not None and rate is not None and lit > 0:
        if abs(lit * rate - amt) > max(2.0, 0.03 * amt):
            rep.add(f"dispenser_math_mismatch:{lit}x{rate}!={amt}", 0.35)

    urea = coerce_number(d.get("urea_concentration_pct"))
    if urea is not None and not (0 <= urea <= 100):
        rep.add(f"impossible_urea_pct:{urea}", 0.3)

    # --- cross-field contamination -------------------------------------
    # Caught a real failure: on a dashboard photo the model put "AFE 3.4 km/l"
    # (fuel efficiency) into rate_per_litre. Dispenser-only fields appearing
    # without any dispenser context means a number was taken from the wrong
    # part of the image, which is exactly the class of error self-reported
    # confidence never notices.
    dispenser_ctx = amt is not None or lit is not None
    for fname in ("rate_per_litre", "urea_concentration_pct"):
        if coerce_number(d.get(fname)) is not None and not dispenser_ctx:
            rep.add(f"dispenser_field_without_dispenser_context:{fname}", 0.35)

    if odo is not None and (amt is not None or lit is not None):
        rep.add("odometer_and_dispenser_fields_on_one_reading", 0.3)

    # A cluster renders ODO and TRIP in the same font and precision. Trip
    # showing a decimal while a 6+ digit odometer shows none is the signature
    # of a dropped decimal point, not of two differently-formatted displays.
    trip = coerce_number(d.get("trip_km"))
    if odo is not None and trip is not None:
        if odo == int(odo) and trip != int(trip) and odo >= 100000:
            rep.add(f"suspected_dropped_decimal:odo={odo} trip={trip}", 0.3)

    # Dashboard clocks get mistaken for dates. A captured_at whose date part is
    # implausible or whose time equals the odometer-panel clock is not evidence.
    cap = d.get("captured_at")
    if isinstance(cap, str) and cap.strip():
        datepart = cap.strip()[:10]
        if not _valid_date(datepart):
            rep.add(f"implausible_captured_at:{cap!r}", 0.25)

    if all(coerce_number(d.get(k)) is None
           for k in ("odometer_km", "trip_km", "amount_rs", "litres",
                     "rate_per_litre", "urea_concentration_pct")):
        rep.add("meter_reading_with_no_readings", 0.4)


def effective_confidence(model_confidence: float, rep: ValidationReport) -> float:
    """Lower model confidence when a free validation check finds an issue."""
    return round(max(0.0, min(1.0, model_confidence) - min(rep.severity, 0.6)), 3)
