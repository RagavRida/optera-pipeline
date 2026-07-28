#!/usr/bin/env python3
"""Recompute the published comparison from committed artifacts, without an API key.

This intentionally verifies recorded model outputs; it does not pretend to
extract new images offline. Live extraction still requires a model credential.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optera import score  # noqa: E402
from optera.ledger import load_rows  # noqa: E402


RUNS = {
    "baseline": ("extractions_baseline_opus1x.json", "ledger_baseline_opus1x.jsonl"),
    "optimized": ("extractions_optimized_accurate.json", "ledger_optimized_accurate.jsonl"),
}


def _summarise(results_path: Path, ledger_path: Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    ledger = load_rows(ledger_path)
    accuracy = score.score(results).as_dict()
    paid = [row for row in ledger if row.get("model") not in (None, "-")]
    total = sum(float(row.get("cost_usd", 0.0)) for row in ledger)
    return {
        "documents": len(results),
        "api_calls": len(paid),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in ledger),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in ledger),
        "total_cost_usd": round(total, 6),
        "cost_per_doc_usd": round(total / len(results), 6) if results else 0.0,
        "accuracy": accuracy,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify committed Optera cost/accuracy evidence")
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="directory containing committed extraction JSON and ledgers")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    for label, (extractions, ledger) in RUNS.items():
        result_path = args.results / extractions
        ledger_path = args.results / ledger
        missing = [str(path) for path in (result_path, ledger_path) if not path.exists()]
        if missing:
            ap.error(f"missing evidence artifact(s): {', '.join(missing)}")
        report[label] = _summarise(result_path, ledger_path)

    baseline, optimized = report["baseline"], report["optimized"]
    report["comparison"] = {
        "reduction_x": round(baseline["cost_per_doc_usd"] / optimized["cost_per_doc_usd"], 2),
        "saved_pct": round(
            (1 - optimized["cost_per_doc_usd"] / baseline["cost_per_doc_usd"]) * 100, 2),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("OPTERA COMMITTED-EVIDENCE VERIFICATION (no API calls)")
    for label in ("baseline", "optimized"):
        item, acc = report[label], report[label]["accuracy"]
        print(
            f"{label:9} ${item['cost_per_doc_usd']:.6f}/doc  "
            f"{item['api_calls']} calls  {item['input_tokens']:,} input / "
            f"{item['output_tokens']:,} output tokens  "
            f"field {acc['field_correct']}/{acc['field_total']}  "
            f"refusal {acc['refusal_correct']}/{acc['refusal_total']}  "
            f"hallucinations {acc['hallucination_count']}"
        )
    print(
        f"optimized is {report['comparison']['reduction_x']:.2f}x cheaper "
        f"({report['comparison']['saved_pct']:.2f}% saved)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
