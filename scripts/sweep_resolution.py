#!/usr/bin/env python3
"""Measure the model x resolution grid per document class.

The per-class policy in config.py is only defensible if the floors in it were
measured rather than guessed. This script extracts the gold-labelled documents
at every combination of model and resolution, scores each against ground truth,
and prints accuracy alongside the cost of getting it.

The output is the evidence behind CLASS_POLICY. Run it again whenever prices,
models or the gold set change.
"""
from __future__ import annotations

import json
import os
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

from optera import imaging, jsonio, preflight, prompts, score as scoring  # noqa: E402
from optera.config import cost_usd  # noqa: E402
from optera.providers import call_vision  # noqa: E402

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-5-20250929"
OPUS = "claude-opus-4-5-20251101"

GRID = {
    "meter_reading": {
        "docs": ["optera_doc_38", "optera_doc_39", "optera_doc_42", "optera_doc_44", "optera_doc_37"],
        "models": [HAIKU, SONNET],
        "dims": [768, 1024, 1568],
        "max_tokens": 400,
        "subtype": "odometer",
    },
    "vendor_bill": {
        "docs": ["optera_doc_26", "optera_doc_27", "optera_doc_28",
                 "optera_doc_32", "optera_doc_35", "optera_doc_43"],
        "models": [HAIKU, SONNET],
        "dims": [1024, 1568],
        "max_tokens": 1600,
        "subtype": None,
    },
    "work_report": {
        "docs": ["optera_doc_03", "optera_doc_14", "optera_doc_20"],
        "models": [HAIKU, SONNET],
        "dims": [1024, 1568],
        "max_tokens": 3000,
        "subtype": None,
    },
}

SCORERS = {
    "meter_reading": scoring._score_meter,
    "vendor_bill": scoring._score_bill,
    "work_report": scoring._score_work_report,
}


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    _, field_gt = scoring.load_gt()
    img_dir = ROOT / "images"

    pfs = {p.doc_id: p for p in preflight.run(sorted(img_dir.glob("*.jpg")), dedupe=False)}
    out_rows = []

    for cls, cfg in GRID.items():
        if only and only != cls:
            continue
        print(f"\n{'=' * 78}\n{cls.upper()}\n{'=' * 78}")
        print(f"{'model':<34}{'dim':>6}{'accuracy':>12}{'cost/doc':>12}{'out tok':>9}")
        print("-" * 78)

        for model in cfg["models"]:
            for dim in cfg["dims"]:
                def one(doc_id: str):
                    pf = pfs.get(doc_id)
                    gold = (field_gt.get(cls) or {}).get(doc_id)
                    if pf is None or not gold:
                        return None
                    media, b64, _ = imaging.encode(pf.image, dim, 85)
                    try:
                        r = call_vision(
                            model=model, system=prompts.RULEBOOK,
                            text=prompts.extraction_prompt(cls, pf.warnings, subtype=cfg["subtype"]),
                            images=[(media, b64)], max_tokens=cfg["max_tokens"], temperature=0.0)
                    except Exception as exc:
                        return {"err": str(exc)[:60], "ok": 0, "tot": 0, "usd": 0.0, "out": 0}
                    obj, _note = jsonio.parse(r.text)
                    data = (obj or {}).get("data")
                    if isinstance(data, dict):
                        ok, tot, _m = SCORERS[cls](data, gold)
                    else:
                        ok, tot = 0, len([k for k in gold if not k.startswith("_")])
                    return {"ok": ok, "tot": tot, "out": r.usage.output_tokens,
                            "usd": cost_usd(model, r.usage.input_tokens, r.usage.output_tokens)}

                with ThreadPoolExecutor(max_workers=5) as pool:
                    res = [x for x in pool.map(one, cfg["docs"]) if x]

                ok = sum(r["ok"] for r in res)
                tot = sum(r["tot"] for r in res)
                usd = sum(r["usd"] for r in res) / max(len(res), 1)
                out = sum(r["out"] for r in res) // max(len(res), 1)
                acc = ok / tot if tot else 0.0
                print(f"{model:<34}{dim:>6}{acc:>11.1%}{usd:>12.5f}{out:>9}")
                out_rows.append({"class": cls, "model": model, "dim": dim,
                                 "accuracy": round(acc, 4), "correct": ok, "total": tot,
                                 "cost_per_doc": round(usd, 6), "out_tokens": out})

    dest = ROOT / "out" / "sweep.json"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out_rows, fh, indent=2)
    print(f"\nwrote {dest}")

    print(f"\n{'=' * 78}\nCHEAPEST CONFIGURATION WITHIN 2 POINTS OF THE BEST, PER CLASS\n{'=' * 78}")
    for cls in {r["class"] for r in out_rows}:
        rows = [r for r in out_rows if r["class"] == cls]
        best = max(r["accuracy"] for r in rows)
        viable = [r for r in rows if r["accuracy"] >= best - 0.02]
        pick = min(viable, key=lambda r: r["cost_per_doc"])
        print(f"  {cls:15} {pick['model']} @ {pick['dim']}px  "
              f"acc={pick['accuracy']:.1%} (best {best:.1%})  ${pick['cost_per_doc']:.5f}/doc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
