"""Orchestration for both the naive baseline and the optimised pipeline.

Both runners consume the same file list and emit the same envelope so a single
scorer can grade them. The only differences are the ones under test: how many
calls get made, to which models, at what resolution.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from . import extract as extract_mod
from . import imaging, jsonio, preflight, prompts, router, schemas, validate
from .config import (BASELINE_JPEG_Q, BASELINE_MAX_DIM, BASELINE_MODEL,
                     ROUTER_MODEL)
from .ledger import Ledger
from .preflight import PreflightResult


def _finalise(env: dict, pf: PreflightResult | None, ledger: Ledger) -> dict:
    """Attach the per-document cost actually recorded in the ledger."""
    rows = [r for r in ledger.rows if r.doc_id == env["doc_id"]]
    env["provenance"]["cost_usd"] = round(sum(r.cost_usd for r in rows), 8)
    env["provenance"]["calls"] = [
        {"stage": r.stage, "model": r.model, "in": r.input_tokens,
         "out": r.output_tokens, "cache_read": r.cache_read_tokens,
         "usd": round(r.cost_usd, 8)} for r in rows
    ]
    if pf is not None:
        env["preflight"] = pf.to_json()
    return env


# --------------------------------------------------------------- baseline --
def run_baseline(paths: list[Path], ledger: Ledger, workers: int = 4,
                 progress: Callable[[str], None] | None = None) -> list[dict]:
    """1x: one full-resolution call per image to the big model. No routing,
    no dedup, no preflight - it pays for every file that arrives, including the
    HTML error page and the duplicates.
    """
    system = prompts.RULEBOOK
    task = prompts.baseline_prompt()

    def one(path: Path) -> dict:
        doc_id = path.stem
        env = schemas.empty_envelope(doc_id)
        try:
            li = imaging.load(path)
        except Exception as exc:
            # Even the naive path cannot send bytes it cannot decode; this is a
            # floor on how naive the comparison is, not a favour to the baseline.
            env["status"] = "error"
            env["validation"] = {"passed": False,
                                 "issues": [f"undecodable:{type(exc).__name__}"], "severity": 1.0}
            if progress:
                progress(f"  baseline {doc_id}: undecodable")
            return _finalise(env, None, ledger)

        media, b64, nbytes = imaging.encode(li.image, BASELINE_MAX_DIM, BASELINE_JPEG_Q)
        from .providers import call_vision
        resp = call_vision(model=BASELINE_MODEL, system=system, text=task,
                           images=[(media, b64)], max_tokens=4000, temperature=0.0)
        ledger.record(doc_id, "baseline", BASELINE_MODEL, resp.usage,
                      image_px=str(BASELINE_MAX_DIM), image_kb=nbytes / 1024)

        obj, note = jsonio.parse(resp.text)
        if obj is None:
            env["status"] = "error"
            env["validation"] = {"passed": False, "issues": [f"parse_failed:{note}"], "severity": 1.0}
            if progress:
                progress(f"  baseline {doc_id}: parse_failed")
            return _finalise(env, None, ledger)

        cls = str(obj.get("doc_class") or "").strip() or "work_report"
        if cls == "not_a_document" or (obj.get("refusal") and obj.get("data") is None):
            ref = obj.get("refusal") or {}
            out = schemas.refusal(doc_id, ref.get("reason", "not_a_document"),
                                  ref.get("observed", ""), doc_class=cls,
                                  confidence=float(obj.get("confidence", 0.8) or 0.8))
            if progress:
                progress(f"  baseline {doc_id}: refused")
            return _finalise(out, None, ledger)

        data = obj.get("data")
        if isinstance(data, dict):
            for rowfield in schemas.COMPACT_COLUMNS:
                data, _n = schemas.expand_compact(data, rowfield)
        rep = validate.validate(cls, data if isinstance(data, dict) else None) \
            if cls in schemas.CLASS_FIELDS else validate.ValidationReport()
        env.update({"doc_class": cls, "status": "extracted",
                    "confidence": float(obj.get("confidence", 0.5) or 0.5),
                    "data": data, "validation": rep.to_json()})
        env["provenance"]["stages"].append(f"baseline:{BASELINE_MODEL}@{BASELINE_MAX_DIM}px")
        if progress:
            progress(f"  baseline {doc_id}: {cls}")
        return _finalise(env, None, ledger)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, sorted(paths)))


# -------------------------------------------------------------- optimised --
def run_optimized(paths: list[Path], ledger: Ledger, workers: int = 4,
                  dedupe: bool = True, allow_escalation: bool = True,
                  progress: Callable[[str], None] | None = None) -> list[dict]:
    """Stage 0 free gate -> stage 1 cheap router -> stage 2 sized extraction
    -> stage 3 validation and targeted escalation.
    """
    pfs = preflight.run(sorted(paths), dedupe=dedupe)
    ledger.record("-", "preflight", "-", None,
                  note=f"screened={len(pfs)} passed={sum(1 for p in pfs if p.ok)}")

    results: list[dict] = []
    live: list[PreflightResult] = []

    for pf in pfs:
        if pf.ok:
            live.append(pf)
            continue
        # Rejected for free. Still emits a full envelope so downstream
        # consumers see one record per input file.
        env = schemas.refusal(pf.doc_id, pf.reason,
                              f"Rejected at preflight without any model call. "
                              f"{'Duplicate of ' + pf.duplicate_of if pf.duplicate_of else ''}".strip(),
                              doc_class="not_a_document" if pf.reason.startswith("not_an_image") else "duplicate")
        env["status"] = "refused"
        env["provenance"]["stages"].append("preflight:free_reject")
        results.append(_finalise(env, pf, ledger))
        if progress:
            progress(f"  preflight {pf.doc_id}: {pf.reason} ($0)")

    def one(pf: PreflightResult) -> dict:
        rt = router.classify(pf, ledger, model=ROUTER_MODEL)
        if rt.doc_class == "not_a_document":
            env = schemas.refusal(
                pf.doc_id, "not_a_document",
                "Classified as an object/scene photograph; no structured record present.",
                confidence=rt.confidence)
            env["quality_flags"] = list(pf.warnings)
            env["provenance"]["stages"].append(f"route:{ROUTER_MODEL}:refused")
            if progress:
                progress(f"  {pf.doc_id}: refused at router")
            return _finalise(env, pf, ledger)

        env = extract_mod.extract(pf, rt.doc_class, ledger,
                                  allow_escalation=allow_escalation, subtype=rt.subtype)
        tag = f"{rt.doc_class}/{rt.subtype}" if rt.subtype else rt.doc_class
        env["provenance"]["stages"].insert(0, f"route:{ROUTER_MODEL}->{tag}")
        if rt.note:
            env["provenance"]["stages"].append(f"router_note:{rt.note}")
        if progress:
            esc = " (escalated)" if env["provenance"].get("escalated") else ""
            progress(f"  {pf.doc_id}: {env['doc_class']} conf={env['confidence']}{esc}")
        return _finalise(env, pf, ledger)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results.extend(pool.map(one, live))

    results.sort(key=lambda e: e["doc_id"])
    return results


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
