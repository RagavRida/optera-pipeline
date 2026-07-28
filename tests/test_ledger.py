"""Tests for append-only, API-usage based cost accounting."""
from __future__ import annotations

import pytest

from optera.config import cost_usd
from optera.ledger import Ledger
from optera.providers import Usage


def test_ledger_records_one_api_call_per_document(tmp_path):
    usage = Usage(input_tokens=1_000, output_tokens=100)
    ledger = Ledger(tmp_path / "ledger.jsonl", "test")
    row = ledger.record("doc_a", "extract", "gpt-4o-mini", usage,
                        note="meter_reading")

    assert row.doc_id == "doc_a"
    assert ledger.summary()["documents"] == 1
    assert ledger.summary()["api_calls"] == 1
    assert row.cost_usd == pytest.approx(cost_usd(row.model, 1_000, 100))
    assert ledger.rows_for_doc("doc_a") == [row]
    ledger.close()
