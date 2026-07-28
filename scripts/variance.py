#!/usr/bin/env python3
"""How much of an accuracy 'difference' is just noise?

Runs the production per-class policy over the gold subset several times with
identical settings and reports the spread. Temperature is 0, but these models
are not deterministic, and the gold set is small - so before believing that a
config change moved accuracy by N points, you need to know how far accuracy
moves when nothing changes at all.

This exists because two apparent per-class regressions in the headline
comparison turned out to be the same model at the same resolution scoring
differently on consecutive runs.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from optera import imaging, jsonio, preflight, prompts, schemas, score as scoring  # noqa: E402
from optera.config import CLASS_POLICY, cost_usd  # noqa: E402
from optera.providers import call_vision  # noqa: E402

REPEATS = int(os.getenv("OPTERA_VARIANCE_REPEATS", "3"))
SCORERS = {
    "meter_reading": scoring._score_meter,
    "vendor_bill": scoring._score_bill,
    "work_report": scoring._score_work_report,
}


def main() -> int:
    routing_gt, field_gt = scoring.load_gt()
    pfs = {p.doc_id: p for p in preflight.run(sorted((ROOT / "images").glob("*.jpg")), dedupe=False)}

    jobs = []
    for cls, docs in field_gt.items():
        if cls not in SCORERS:
            continue
        for doc_id, gold in docs.items():
            if doc_id.startswith("_") or not isinstance(gold, dict):
                continue
            if doc_id in pfs:
                jobs.append((cls, doc_id, gold))

    print(f"gold documents: {len(jobs)}  |  repeats: {REPEATS}\n")

    def run_one(job):
        cls, doc_id, gold = job
        pol = CLASS_POLICY[cls]
        pf = pfs[doc_id]
        subtype = None
        if cls == "meter_reading":
            subtype = "dispenser" if gold.get("reading_type", "").endswith("dispenser") else "odometer"
        media, b64, _ = imaging.encode(pf.image, pol.max_dim, pol.jpeg_q)
        try:
            r = call_vision(model=pol.model, system=prompts.RULEBOOK,
                            text=prompts.extraction_prompt(cls, pf.warnings, subtype=subtype),
                            images=[(media, b64)], max_tokens=pol.max_tokens, temperature=0.0)
        except Exception:
            return cls, doc_id, 0, len([k for k in gold if not k.startswith("_")]), 0.0
        obj, _n = jsonio.parse(r.text)
        data = (obj or {}).get("data")
        if isinstance(data, dict):
            for rf in schemas.COMPACT_COLUMNS:
                data, _c = schemas.expand_compact(data, rf)
            ok, tot, _m = SCORERS[cls](data, gold)
        else:
            ok, tot = 0, len([k for k in gold if not k.startswith("_")])
        return cls, doc_id, ok, tot, cost_usd(pol.model, r.usage.input_tokens, r.usage.output_tokens)

    overall, per_class_runs = [], {}
    for rep in range(REPEATS):
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(run_one, jobs))
        agg = {}
        for cls, _d, ok, tot, _u in rows:
            a = agg.setdefault(cls, [0, 0])
            a[0] += ok
            a[1] += tot
        tot_ok = sum(r[2] for r in rows)
        tot_all = sum(r[3] for r in rows)
        acc = tot_ok / tot_all if tot_all else 0
        overall.append(acc)
        line = "  ".join(f"{c}={agg[c][0]}/{agg[c][1]} ({agg[c][0]/agg[c][1]:.0%})" for c in sorted(agg))
        for c, v in agg.items():
            per_class_runs.setdefault(c, []).append(v[0] / v[1] if v[1] else 0)
        print(f"  run {rep + 1}: overall {tot_ok}/{tot_all} = {acc:.1%}   {line}")

    print(f"\n{'=' * 74}\nSPREAD WITH NOTHING CHANGED\n{'=' * 74}")
    span = (max(overall) - min(overall)) * 100
    print(f"  overall field accuracy: {min(overall):.1%} - {max(overall):.1%}  "
          f"(spread {span:.1f} pts"
          + (f", sd {statistics.stdev(overall) * 100:.1f} pts)" if len(overall) > 1 else ")"))
    for c, vals in sorted(per_class_runs.items()):
        print(f"  {c:16} {min(vals):.0%} - {max(vals):.0%}   (spread {(max(vals) - min(vals)) * 100:.1f} pts)")
    print(f"\n  => Treat any accuracy difference smaller than ~{max(span, 1):.0f} points as "
          f"unresolved by this gold set.")

    dest = ROOT / "out" / "variance.json"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "w") as fh:
        json.dump({"repeats": REPEATS, "overall": overall,
                   "per_class": per_class_runs,
                   "overall_spread_pts": round(span, 2)}, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
