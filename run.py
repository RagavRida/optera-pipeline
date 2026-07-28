#!/usr/bin/env python3
"""Optera document pipeline - single entrypoint.

    python run.py                 # optimised pipeline + accuracy report
    python run.py --mode both     # baseline vs optimised, the headline comparison
    python run.py --mode baseline # the naive 1x only

Everything it prints is derived from out/ledger_*.jsonl, which is written call
by call as the run proceeds.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the repo has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from optera import pipeline, score as scoring  # noqa: E402
from optera.cache import ResultCache  # noqa: E402
from optera.config import BASELINE_MODEL, CLASS_POLICY, ROUTER_MODEL  # noqa: E402
from optera.ledger import Ledger  # noqa: E402

BAR = "=" * 78
SUB = "-" * 78


def _fmt_usd(x: float) -> str:
    return f"${x:,.6f}" if x < 0.01 else f"${x:,.4f}"


def _print_summary(title: str, summ: dict) -> None:
    print(f"\n{SUB}\n{title}\n{SUB}")
    print(f"  documents            {summ['documents']}")
    print(f"  API calls            {summ['api_calls']}  ({summ['calls_per_doc']}/doc)")
    print(f"  input tokens         {summ['input_tokens']:,}")
    print(f"  output tokens        {summ['output_tokens']:,}")
    if summ["cache_read_tokens"]:
        print(f"  cached prompt tokens {summ['cache_read_tokens']:,}")
    print(f"  TOTAL COST           {_fmt_usd(summ['total_cost_usd'])}")
    print(f"  COST PER DOC         {_fmt_usd(summ['cost_per_doc_usd'])}")
    print(f"  cost per 1,000 docs  ${summ['cost_per_1000_docs_usd']:,.2f}")
    if summ["by_model"]:
        print("  by model:")
        for m, v in sorted(summ["by_model"].items(), key=lambda kv: -kv[1]["cost"]):
            print(f"    {m:34} {v['calls']:3} calls  {_fmt_usd(v['cost'])}")
    if summ["by_stage"]:
        print("  by stage:")
        for s, v in sorted(summ["by_stage"].items(), key=lambda kv: -kv[1]["cost"]):
            print(f"    {s:34} {v['calls']:3} calls  {_fmt_usd(v['cost'])}")


def _print_accuracy(title: str, sc: dict) -> None:
    print(f"\n{SUB}\n{title}\n{SUB}")
    ra, fa, rf = sc["routing_accuracy"], sc["field_accuracy"], sc["refusal_accuracy"]
    print(f"  routing accuracy     {ra:.1%}  ({sc['routing_correct']}/{sc['routing_total']} real documents classified correctly)"
          if ra is not None else "  routing accuracy     n/a")
    print(f"  refusal accuracy     {rf:.1%}  ({sc['refusal_correct']}/{sc['refusal_total']} non-records correctly declined)"
          if rf is not None else "  refusal accuracy     n/a")
    print(f"  field accuracy       {fa:.1%}  ({sc['field_correct']}/{sc['field_total']} gold fields on the labelled subset)"
          if fa is not None else "  field accuracy       n/a")
    halluc = sc["hallucination_count"]
    flag = "PASS" if halluc == 0 else "FAIL"
    print(f"  hallucinations       {halluc}  [{flag}]  <- structured output invented from a non-document")
    if sc["hallucinated_docs"]:
        print(f"     {', '.join(sc['hallucinated_docs'])}")
    if sc["dropped_real_documents"]:
        print(f"  dropped real docs    {len(sc['dropped_real_documents'])}: {', '.join(sc['dropped_real_documents'])}")
    if sc["misrouted"]:
        print("  misrouted:")
        for m in sc["misrouted"]:
            print(f"     {m['doc']}: expected {m['expected']}, got {m['got']}")
    if sc["dedup_detected"] or sc["dedup_missed"]:
        print(f"  duplicates caught    {len(sc['dedup_detected'])} "
              f"(missed {len(sc['dedup_missed'])})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Optera document extraction pipeline")
    ap.add_argument("--input", default="images", help="directory of images")
    ap.add_argument("--mode", choices=["optimized", "baseline", "both"], default="optimized")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--cache-dir", default=".optera-cache",
                    help="persistent exact-image result cache for optimized runs")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable result-cache reads and writes (best for fresh benchmarks)")
    ap.add_argument("--targeted-reread", action="store_true",
                    help="opt in to validation-gated crop re-reads; evaluate before production")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="logging verbosity (default: WARNING)")
    ap.add_argument("--log-format", default="text", choices=["text", "json"],
                    help="log output format (default: text)")
    args = ap.parse_args()

    # Configure logging before anything else runs.
    level = getattr(logging, args.log_level)
    if args.log_format == "json":
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "ts": self.formatTime(record), "level": record.levelname,
                    "logger": record.name, "msg": record.getMessage(),
                }, ensure_ascii=False)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s  %(message)s",
            datefmt="%H:%M:%S"))
    logging.basicConfig(level=level, handlers=[handler])

    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"error: input directory {in_dir} does not exist", file=sys.stderr)
        return 2

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic"}
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"error: no images found in {in_dir}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress = None if args.quiet else (lambda s: print(s, flush=True))

    print(BAR)
    print(f"OPTERA PIPELINE  |  {len(paths)} files  |  mode={args.mode}  |  {stamp}")
    print(BAR)
    print(f"  baseline model : {BASELINE_MODEL}")
    print(f"  router model   : {ROUTER_MODEL}")
    for cls, pol in CLASS_POLICY.items():
        print(f"  {cls:15}: {pol.model} @ {pol.max_dim}px")

    report: dict = {"run": stamp, "files": len(paths), "mode": args.mode}
    cache = None if args.no_cache else ResultCache(Path(args.cache_dir))
    if cache and args.mode in ("optimized", "both"):
        print(f"  result cache   : {cache.root} (fingerprint {cache.fingerprint})")

    # ------------------------------------------------------------ baseline --
    if args.mode in ("baseline", "both"):
        print(f"\n>>> BASELINE: one full-resolution {BASELINE_MODEL} call per image")
        led = Ledger(out / f"ledger_baseline_{stamp}.jsonl", f"baseline_{stamp}")
        t0 = time.time()
        res = pipeline.run_baseline(paths, led, workers=args.workers, progress=progress)
        led.close()
        pipeline.write_results(res, out / f"results_baseline_{stamp}.json")
        summ = led.summary(n_docs=len(paths))
        summ["wall_clock_s"] = round(time.time() - t0, 1)
        report["baseline"] = {"cost": summ}
        _print_summary("BASELINE (1x)", summ)
        if not args.no_score:
            sc = scoring.score(res).as_dict()
            report["baseline"]["accuracy"] = sc
            _print_accuracy("BASELINE ACCURACY", sc)

    # ----------------------------------------------------------- optimised --
    if args.mode in ("optimized", "both"):
        print("\n>>> OPTIMISED: preflight -> route -> class-specific extraction -> validate")
        led = Ledger(out / f"ledger_optimized_{stamp}.jsonl", f"optimized_{stamp}")
        t0 = time.time()
        res = pipeline.run_optimized(
            paths, led, workers=args.workers, dedupe=not args.no_dedupe,
            cache=cache,
            targeted_reread=args.targeted_reread,
            progress=progress)
        led.close()
        pipeline.write_results(res, out / f"results_optimized_{stamp}.json")
        summ = led.summary(n_docs=len(paths))
        summ["wall_clock_s"] = round(time.time() - t0, 1)
        report["optimized"] = {"cost": summ}
        _print_summary("OPTIMISED", summ)
        if not args.no_score:
            sc = scoring.score(res).as_dict()
            report["optimized"]["accuracy"] = sc
            _print_accuracy("OPTIMISED ACCURACY", sc)

    # ---------------------------------------------------------- comparison --
    if args.mode == "both":
        b, o = report["baseline"], report["optimized"]
        bc, oc = b["cost"]["cost_per_doc_usd"], o["cost"]["cost_per_doc_usd"]
        print(f"\n{BAR}\nHEADLINE\n{BAR}")
        print(f"  baseline    {_fmt_usd(bc)} / doc   (${b['cost']['cost_per_1000_docs_usd']:,.2f} per 1,000)")
        print(f"  optimised   {_fmt_usd(oc)} / doc   (${o['cost']['cost_per_1000_docs_usd']:,.2f} per 1,000)")
        if oc > 0:
            print(f"  reduction   {bc / oc:.1f}x cheaper  ({(1 - oc / bc) * 100:.1f}% saved)")
        if not args.no_score:
            ba, oa = b.get("accuracy", {}), o.get("accuracy", {})
            for label, key in (("routing", "routing_accuracy"),
                               ("refusal", "refusal_accuracy"),
                               ("field  ", "field_accuracy")):
                bv, ov = ba.get(key), oa.get(key)
                if bv is not None and ov is not None:
                    delta = (ov - bv) * 100
                    sign = "+" if delta >= 0 else ""
                    print(f"  {label} accuracy   baseline {bv:.1%}  ->  optimised {ov:.1%}   ({sign}{delta:.1f} pts)")
            print(f"  hallucinations   baseline {ba.get('hallucination_count')}  ->  optimised {oa.get('hallucination_count')}")
        report["comparison"] = {
            "baseline_cost_per_doc": bc, "optimized_cost_per_doc": oc,
            "reduction_x": round(bc / oc, 2) if oc else None,
            "saved_pct": round((1 - oc / bc) * 100, 2) if bc else None,
        }

    path = out / f"report_{stamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    print(f"ledgers and per-document JSON in {out}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
