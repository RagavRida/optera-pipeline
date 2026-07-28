"""Tests for accurate accounting of shared multi-image requests."""
from __future__ import annotations

import pytest

from optera.config import cost_usd
from optera.ledger import Ledger
from optera.providers import Usage


def test_batch_call_is_one_api_call_with_explicit_document_shares(tmp_path):
    usage = Usage(input_tokens=1_000, output_tokens=100)
    ledger = Ledger(tmp_path / "ledger.jsonl", "test")
    row = ledger.record_batch(
        ["doc_a", "doc_b", "doc_c"], "extract_batch",
        "claude-haiku-4-5-20251001", usage, note="meter_reading×3",
    )

    assert row.doc_id == "batch:doc_a"
    assert row.batch_doc_ids == ["doc_a", "doc_b", "doc_c"]
    assert ledger.summary()["documents"] == 3
    assert ledger.summary(n_docs=3)["api_calls"] == 1

    shares = [ledger.share_for_doc(row, doc_id)
              for doc_id in ("doc_a", "doc_b", "doc_c")]
    assert sum(shares) == pytest.approx(cost_usd(row.model, 1_000, 100))
    assert all(share == pytest.approx(shares[0]) for share in shares)
    assert ledger.rows_for_doc("doc_b") == [row]

    allocated = ledger.usage_share_for_doc(row, "doc_c")
    assert allocated == {"in": pytest.approx(1000 / 3),
                         "out": pytest.approx(100 / 3),
                         "cache_read": 0.0, "shared_by": 3}
    ledger.close()
