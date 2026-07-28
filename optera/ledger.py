"""Token and cost ledger.

Append-only JSONL, one row per API call, written as calls complete. Every cost
figure this project reports is a sum over these rows - there is no second,
prettier accounting path. If a number appears in the README it can be traced to
lines in out/ledger_*.jsonl.

Zero-token events (free rejects, cache hits on our own disk cache) are recorded
too, with cost 0.0, so "calls avoided" is visible rather than merely implied.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import cost_usd
from .providers import Usage


@dataclass
class LedgerRow:
    run: str
    doc_id: str
    stage: str            # preflight | route | extract | escalate | validate
    model: str            # "-" for zero-token events
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    image_px: str = ""
    image_kb: float = 0.0
    note: str = ""
    # A multi-image request is still represented by exactly one ledger row so
    # API-call counts remain truthful. The participating document ids allow
    # result envelopes to receive an explicit equal-share attribution.
    batch_doc_ids: list[str] = field(default_factory=list)


class Ledger:
    def __init__(self, path: Path, run: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.rows: list[LedgerRow] = []
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def record(self, doc_id: str, stage: str, model: str, usage: Usage | None = None,
               image_px: str = "", image_kb: float = 0.0, note: str = "",
               batch_doc_ids: list[str] | None = None) -> LedgerRow:
        u = usage or Usage()
        row = LedgerRow(
            run=self.run, doc_id=doc_id, stage=stage, model=model,
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_write_tokens=u.cache_write_tokens, cache_read_tokens=u.cache_read_tokens,
            cost_usd=(cost_usd(model, u.input_tokens, u.output_tokens,
                               u.cache_write_tokens, u.cache_read_tokens)
                      if model and model != "-" else 0.0),
            latency_s=round(u.latency_s, 3), image_px=image_px,
            image_kb=round(image_kb, 1), note=note,
            batch_doc_ids=list(batch_doc_ids or []),
        )
        with self._lock:
            self.rows.append(row)
            self._fh.write(json.dumps(asdict(row)) + "\n")
            self._fh.flush()
        return row

    def record_batch(self, doc_ids: list[str], stage: str, model: str,
                     usage: Usage, image_px: str = "", image_kb: float = 0.0,
                     note: str = "") -> LedgerRow:
        """Record one API call shared by multiple input documents.

        The underlying token counts are the provider's actual request-level
        counts. Per-document output provenance uses a transparent equal share;
        we do not fabricate image-token estimates for individual positions.
        """
        if not doc_ids:
            raise ValueError("record_batch requires at least one document id")
        return self.record(
            doc_id=f"batch:{doc_ids[0]}", stage=stage, model=model, usage=usage,
            image_px=image_px, image_kb=image_kb, note=note,
            batch_doc_ids=doc_ids,
        )

    def rows_for_doc(self, doc_id: str) -> list[LedgerRow]:
        """Rows directly or jointly attributable to ``doc_id``."""
        with self._lock:
            return [row for row in self.rows
                    if row.doc_id == doc_id or doc_id in row.batch_doc_ids]

    @staticmethod
    def share_for_doc(row: LedgerRow, doc_id: str) -> float:
        """Cost attribution for one document, preserving the ledger total."""
        if doc_id in row.batch_doc_ids:
            return row.cost_usd / len(row.batch_doc_ids)
        return row.cost_usd

    @staticmethod
    def usage_share_for_doc(row: LedgerRow, doc_id: str) -> dict[str, float | int]:
        """Request usage annotated with an equal allocation for batch members."""
        n = len(row.batch_doc_ids) if doc_id in row.batch_doc_ids else 1
        return {
            "in": row.input_tokens / n,
            "out": row.output_tokens / n,
            "cache_read": row.cache_read_tokens / n,
            "shared_by": n,
        }

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ reporting --
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.rows)

    def docs(self) -> set[str]:
        docs: set[str] = set()
        for row in self.rows:
            docs.update(row.batch_doc_ids or [row.doc_id])
        return docs

    def summary(self, n_docs: int | None = None) -> dict[str, Any]:
        n = n_docs if n_docs is not None else len(self.docs())
        by_stage: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "in": 0, "out": 0, "cache_r": 0, "cost": 0.0})
        by_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        for r in self.rows:
            s = by_stage[r.stage]
            s["calls"] += 1
            s["in"] += r.input_tokens
            s["out"] += r.output_tokens
            s["cache_r"] += r.cache_read_tokens
            s["cost"] += r.cost_usd
            if r.model and r.model != "-":
                m = by_model[r.model]
                m["calls"] += 1
                m["in"] += r.input_tokens
                m["out"] += r.output_tokens
                m["cost"] += r.cost_usd
        total = self.total_cost()
        paid = [r for r in self.rows if r.model and r.model != "-"]
        return {
            "run": self.run,
            "documents": n,
            "api_calls": len(paid),
            "calls_per_doc": round(len(paid) / n, 3) if n else 0.0,
            "input_tokens": sum(r.input_tokens for r in self.rows),
            "output_tokens": sum(r.output_tokens for r in self.rows),
            "cache_read_tokens": sum(r.cache_read_tokens for r in self.rows),
            "cache_write_tokens": sum(r.cache_write_tokens for r in self.rows),
            "total_cost_usd": round(total, 6),
            "cost_per_doc_usd": round(total / n, 6) if n else 0.0,
            "cost_per_1000_docs_usd": round(total / n * 1000, 2) if n else 0.0,
            "wall_latency_s": round(sum(r.latency_s for r in self.rows), 1),
            "by_stage": {k: {kk: (round(vv, 6) if kk == "cost" else int(vv))
                             for kk, vv in v.items()} for k, v in by_stage.items()},
            "by_model": {k: {kk: (round(vv, 6) if kk == "cost" else int(vv))
                             for kk, vv in v.items()} for k, v in by_model.items()},
        }


def load_rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]
