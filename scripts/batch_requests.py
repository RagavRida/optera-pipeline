#!/usr/bin/env python3
"""Build, submit and collect independent OpenAI Batch API document requests.

This deliberately batches transport, not document context: every JSONL line is
one image and one Chat Completions request. Optimized extraction is a two-phase
workflow: submit routers, collect them, then submit extraction requests for the
routed real documents.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optera import batch_api, imaging, jsonio, preflight, prompts  # noqa: E402
from optera.config import (BASELINE_JPEG_Q, BASELINE_MAX_DIM, BASELINE_MODEL,
                           CLASS_POLICY, ROUTER_JPEG_Q, ROUTER_MAX_DIM, ROUTER_MODEL)  # noqa: E402


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    import os
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _payload(model: str, system: str, task: str, image: tuple[str, str], max_tokens: int) -> dict:
    mime, b64 = image
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": task},
            ]},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def build(stage: str, paths: list[Path], output: Path, routes: Path | None = None) -> int:
    route_map = {}
    if stage == "extract":
        if not routes:
            raise SystemExit("--routes is required for extract")
        route_map = {row["doc_id"]: row for row in json.loads(routes.read_text())}

    records = []
    if stage == "router":
        usable = [pf for pf in preflight.run(paths, dedupe=True) if pf.ok]
        for pf in usable:
            media, b64, _ = imaging.encode(pf.image, ROUTER_MAX_DIM, ROUTER_JPEG_Q)
            records.append({"custom_id": pf.doc_id, "method": "POST", "url": "/v1/chat/completions",
                            "body": _payload(ROUTER_MODEL, "You are a fast document triage classifier. Reply with JSON only.",
                                             prompts.classification_prompt(), (media, b64), 80)})
    else:
        for path in paths:
            if stage == "baseline":
                try:
                    loaded = imaging.load(path)
                except Exception:
                    # Mirrors the live naive baseline: an undecodable file is
                    # reported as an error, never uploaded as a fake image.
                    continue
                media, b64, _ = imaging.encode(loaded.image, BASELINE_MAX_DIM, BASELINE_JPEG_Q)
                model, task, max_tokens = BASELINE_MODEL, prompts.baseline_prompt(), 4000
            else:
                pf = next((x for x in preflight.run([path], dedupe=False) if x.ok), None)
                route = route_map.get(path.stem)
                if pf is None or not route or route.get("doc_class") == "not_a_document":
                    continue
                pol = CLASS_POLICY[route["doc_class"]]
                media, b64, _ = imaging.encode(pf.image, pol.max_dim, pol.jpeg_q)
                model, task, max_tokens = pol.model, prompts.extraction_prompt(
                    route["doc_class"], pf.warnings, route.get("subtype")), pol.max_tokens
            records.append({"custom_id": path.stem, "method": "POST", "url": "/v1/chat/completions",
                            "body": _payload(model, prompts.RULEBOOK, task, (media, b64), max_tokens)})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"wrote {len(records)} independent {stage} requests to {output}")
    return 0


def routes(router_output: Path, output: Path) -> int:
    """Turn completed router Batch output into extraction-stage route inputs."""
    parsed = []
    for line in router_output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        body = (row.get("response") or {}).get("body") or {}
        message = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        obj, note = jsonio.parse(message)
        cls = str((obj or {}).get("class", "")).strip().lower()
        if cls not in CLASS_POLICY and cls != "not_a_document":
            cls = "work_report"  # same safe fail-open policy as the live router
        subtype = (obj or {}).get("subtype")
        subtype = subtype if cls == "meter_reading" and subtype in {"odometer", "dispenser"} else None
        parsed.append({"doc_id": row.get("custom_id"), "doc_class": cls,
                       "subtype": subtype, "router_note": note})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"wrote {len(parsed)} routes to {output}")
    return 0


def submit(requests: Path, state: Path, label: str) -> int:
    _load_env()
    submitted = batch_api.submit_jsonl(requests, {"project": "optera", "stage": label})
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"label": label, "requests": str(requests), **submitted}, indent=2))
    print(json.dumps(submitted, indent=2))
    return 0


def collect(state: Path, output: Path) -> int:
    _load_env()
    saved = json.loads(state.read_text())
    result = batch_api.collect_if_complete(saved["batch_id"], output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Optera OpenAI Batch API helper")
    sub = ap.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--stage", choices=["baseline", "router", "extract"], required=True)
    p_build.add_argument("--input", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--routes", type=Path)
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--requests", type=Path, required=True)
    p_submit.add_argument("--state", type=Path, required=True)
    p_submit.add_argument("--label", required=True)
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--state", type=Path, required=True)
    p_collect.add_argument("--out", type=Path, required=True)
    p_routes = sub.add_parser("routes")
    p_routes.add_argument("--router-output", type=Path, required=True)
    p_routes.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.command == "build":
        paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
        return build(args.stage, paths, args.out, args.routes)
    if args.command == "submit":
        return submit(args.requests, args.state, args.label)
    if args.command == "collect":
        return collect(args.state, args.out)
    return routes(args.router_output, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
