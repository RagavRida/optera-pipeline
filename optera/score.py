"""Accuracy scoring against hand-written ground truth.

Three separate numbers, deliberately not averaged into one:

  routing_accuracy      - did we work out what the image is
  refusal_accuracy      - did we decline everything that carries no record
  field_accuracy        - on the gold subset, are the values right

They are kept apart because they fail differently. A pipeline can hit 95% field
accuracy while inventing an invoice from a battery photo, and that pipeline is
not 95% good - it is dangerous. Hallucination rate is therefore reported as its
own hard gate rather than folded into an average.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jsonio import coerce_number

GT_DIR = Path(__file__).resolve().parent.parent / "groundtruth"


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _norm_code(s: Any) -> str:
    """Vehicle/bus codes: compare alphanumerics only.

    Operators write GJ-06-AV-4045, GJ 06 AV 4045 and GJ06AV4045 for the same
    vehicle. Punishing punctuation would measure formatting, not reading.
    """
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _num_match(a: Any, b: Any, rel: float = 0.005) -> bool:
    x, y = coerce_number(a), coerce_number(b)
    if x is None or y is None:
        return False
    if y == 0:
        return abs(x) < 1e-9
    return abs(x - y) <= max(abs(y) * rel, 0.01)


def _text_match(a: Any, b: Any) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Vendor names get abbreviated ("Anupam" for "Anupam Enterprise"); accept
    # containment in either direction on names long enough to be unambiguous.
    return (na in nb or nb in na) and min(len(na), len(nb)) >= 5


@dataclass
class Score:
    routing_total: int = 0
    routing_correct: int = 0
    refusal_total: int = 0
    refusal_correct: int = 0
    hallucinated: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)         # real doc wrongly refused
    misrouted: list[tuple[str, str, str]] = field(default_factory=list)
    field_total: int = 0
    field_correct: int = 0
    per_doc: dict[str, dict] = field(default_factory=dict)
    dedup_found: list[str] = field(default_factory=list)
    dedup_missed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "routing_accuracy": round(self.routing_correct / self.routing_total, 4) if self.routing_total else None,
            "routing_correct": self.routing_correct, "routing_total": self.routing_total,
            "refusal_accuracy": round(self.refusal_correct / self.refusal_total, 4) if self.refusal_total else None,
            "refusal_correct": self.refusal_correct, "refusal_total": self.refusal_total,
            "hallucination_count": len(self.hallucinated),
            "hallucinated_docs": self.hallucinated,
            "dropped_real_documents": self.dropped,
            "misrouted": [{"doc": d, "expected": e, "got": g} for d, e, g in self.misrouted],
            "field_accuracy": round(self.field_correct / self.field_total, 4) if self.field_total else None,
            "field_correct": self.field_correct, "field_total": self.field_total,
            "dedup_detected": self.dedup_found, "dedup_missed": self.dedup_missed,
            "per_doc": self.per_doc,
        }


def load_gt() -> tuple[dict, dict]:
    with open(GT_DIR / "routing.json", encoding="utf-8") as fh:
        routing = json.load(fh)["labels"]
    with open(GT_DIR / "fields.json", encoding="utf-8") as fh:
        fields = json.load(fh)
    return routing, fields


def _score_meter(got: dict, gold: dict) -> tuple[int, int, list[str]]:
    ok = tot = 0
    misses = []
    for key, want in gold.items():
        if key.startswith("_"):
            continue
        tot += 1
        have = got.get(key)
        good = _text_match(have, want) if key == "reading_type" else _num_match(have, want)
        ok += int(good)
        if not good:
            misses.append(f"{key}: got {have!r} want {want!r}")
    return ok, tot, misses


def _score_bill(got: dict, gold: dict) -> tuple[int, int, list[str]]:
    ok = tot = 0
    misses = []
    for key, want in gold.items():
        if key.startswith("_") or key == "line_items":
            continue
        tot += 1
        if key == "line_item_count":
            items = got.get("line_items") or []
            good = isinstance(items, list) and len(items) == want
            have: Any = len(items) if isinstance(items, list) else None
        elif key == "vehicle_no":
            have = got.get(key)
            good = _norm_code(have) == _norm_code(want)
        elif key in ("total_amount", "tax_amount"):
            have = got.get(key)
            good = _num_match(have, want)
        elif key == "invoice_date":
            have = got.get(key)
            good = str(have or "").strip() == want
        else:
            have = got.get(key)
            good = _text_match(have, want)
        ok += int(good)
        if not good:
            misses.append(f"{key}: got {have!r} want {want!r}")
    return ok, tot, misses


def _score_work_report(got: dict, gold: dict) -> tuple[int, int, list[str]]:
    """Structural scoring only - see the scope note in groundtruth/fields.json."""
    ok = tot = 0
    misses = []

    tot += 1
    if _text_match(got.get("depot"), gold["depot"]):
        ok += 1
    else:
        misses.append(f"depot: got {got.get('depot')!r} want {gold['depot']!r}")

    tot += 1
    if str(got.get("report_date") or "").strip() == gold["report_date"]:
        ok += 1
    else:
        misses.append(f"report_date: got {got.get('report_date')!r} want {gold['report_date']!r}")

    entries = got.get("entries") or []
    n = len(entries) if isinstance(entries, list) else 0
    tot += 1
    # Allow +/-1 row: an operator drawing a stray rule line is a genuine
    # ambiguity about what counts as a row, not a reading error.
    if abs(n - gold["entry_count"]) <= 1:
        ok += 1
    else:
        misses.append(f"entry_count: got {n} want {gold['entry_count']}")

    want_codes = {_norm_code(c) for c in gold.get("bus_codes", []) if c}
    if want_codes:
        got_codes = {_norm_code(e.get("bus_no")) for e in entries
                     if isinstance(e, dict) and e.get("bus_no")}
        got_codes.discard("")
        hit = len(want_codes & got_codes)
        tot += len(want_codes)
        ok += hit
        missing = want_codes - got_codes
        if missing:
            misses.append(f"bus_codes missing: {sorted(missing)}")
    return ok, tot, misses


def score(results: list[dict]) -> Score:
    routing_gt, field_gt = load_gt()
    sc = Score()
    by_id = {r["doc_id"]: r for r in results}

    for doc_id, gt in routing_gt.items():
        res = by_id.get(doc_id)
        entry: dict[str, Any] = {"expected_class": gt["class"], "expect": gt["expect"]}
        if res is None:
            entry["error"] = "no_result"
            sc.per_doc[doc_id] = entry
            continue

        refused = res.get("status") == "refused"
        got_class = res.get("doc_class")
        entry["got_class"] = got_class
        entry["status"] = res.get("status")
        entry["cost_usd"] = res.get("provenance", {}).get("cost_usd", 0.0)

        if gt["expect"] == "refuse":
            sc.refusal_total += 1
            if refused:
                sc.refusal_correct += 1
                entry["verdict"] = "correctly_refused"
            else:
                # Only counts as a hallucination if it actually produced content.
                has_data = bool(res.get("data"))
                entry["verdict"] = "HALLUCINATED" if has_data else "not_refused_but_empty"
                if has_data:
                    sc.hallucinated.append(doc_id)
            if gt.get("duplicate_of"):
                (sc.dedup_found if refused else sc.dedup_missed).append(doc_id)
        else:
            sc.routing_total += 1
            if refused:
                sc.dropped.append(doc_id)
                entry["verdict"] = "DROPPED_REAL_DOCUMENT"
            elif got_class == gt["class"]:
                sc.routing_correct += 1
                entry["verdict"] = "routed_ok"
            else:
                sc.misrouted.append((doc_id, gt["class"], str(got_class)))
                entry["verdict"] = "MISROUTED"

        # ------------------------------------------------ field-level gold --
        # Non-documents and duplicates carry no fields to get right; their
        # correctness is entirely captured by refusal_accuracy. Counting them
        # here would have double-penalised a correct refusal.
        cls = gt["class"]
        if gt["expect"] == "refuse":
            sc.per_doc[doc_id] = entry
            continue
        gold = (field_gt.get(cls) or {}).get(doc_id)
        if gold and isinstance(res.get("data"), dict) and got_class == cls:
            data = res["data"]
            if cls == "meter_reading":
                ok, tot, misses = _score_meter(data, gold)
            elif cls == "vendor_bill":
                ok, tot, misses = _score_bill(data, gold)
            elif cls == "work_report":
                ok, tot, misses = _score_work_report(data, gold)
            else:
                ok = tot = 0
                misses = []
            sc.field_correct += ok
            sc.field_total += tot
            entry["fields"] = {"correct": ok, "total": tot, "misses": misses}
        elif gold:
            n = len([k for k in gold if not k.startswith("_")])
            sc.field_total += n
            entry["fields"] = {"correct": 0, "total": n,
                               "misses": ["no usable data extracted"]}

        sc.per_doc[doc_id] = entry

    return sc
