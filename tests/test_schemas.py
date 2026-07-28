"""Tests for optera.schemas — envelope creation, compact encoding, prompt generation."""
from __future__ import annotations

import pytest
from optera.schemas import (
    expand_compact, prompt_spec, empty_envelope, refusal,
    COMPACT_COLUMNS, SUBTYPE_FIELDS, CLASS_FIELDS,
)


class TestExpandCompact:
    """Positional array -> object inflation."""

    def test_basic_entries(self):
        data = {
            "entries": [
                ["03", None, "TAM17", "brake pedal valve change", None, 0],
            ]
        }
        data, n = expand_compact(data, "entries")
        assert n == 1
        row = data["entries"][0]
        assert row["sr_no"] == "03"
        assert row["bus_no"] == "TAM17"
        assert row["work_done"] == "brake pedal valve change"
        assert row["struck_through"] is False  # bool coercion from 0

    def test_struck_through_true(self):
        data = {"entries": [["01", None, "X", "cancelled", None, 1]]}
        data, _ = expand_compact(data, "entries")
        assert data["entries"][0]["struck_through"] is True

    def test_short_row_pads_with_null(self):
        data = {"entries": [["01", "Raju"]]}  # only 2 of 6 fields
        data, n = expand_compact(data, "entries")
        assert n == 1
        row = data["entries"][0]
        assert row["sr_no"] == "01"
        assert row["mechanic"] == "Raju"
        assert row["bus_no"] is None
        assert row["work_done"] is None

    def test_long_row_truncates(self):
        data = {"entries": [["01", None, "X", "work", None, False, "extra", "more"]]}
        data, n = expand_compact(data, "entries")
        assert n == 1
        assert len(data["entries"][0]) == 6  # 6 fields, extra discarded

    def test_object_rows_pass_through(self):
        obj = {"sr_no": "01", "bus_no": "TAM17", "work_done": "brake set"}
        data = {"entries": [obj]}
        data, n = expand_compact(data, "entries")
        assert n == 0  # no conversion
        assert data["entries"][0] is obj

    def test_mixed_arrays_and_objects(self):
        data = {"entries": [
            {"sr_no": "01", "bus_no": "X", "work_done": "task1"},
            ["02", None, "Y", "task2", None, 0],
        ]}
        data, n = expand_compact(data, "entries")
        assert n == 1
        assert data["entries"][0]["sr_no"] == "01"  # object passed through
        assert data["entries"][1]["sr_no"] == "02"   # array converted

    def test_line_items(self):
        data = {"line_items": [["Puncture Fitting", None, 1, 150, 150]]}
        data, n = expand_compact(data, "line_items")
        assert n == 1
        row = data["line_items"][0]
        assert row["description"] == "Puncture Fitting"
        assert row["amount"] == 150

    def test_no_field_present(self):
        data = {"something_else": 42}
        data, n = expand_compact(data, "entries")
        assert n == 0
        assert "entries" not in data or data.get("entries") is None

    def test_field_not_a_list(self):
        data = {"entries": "not a list"}
        data, n = expand_compact(data, "entries")
        assert n == 0


class TestPromptSpec:
    """Schema-to-prompt generation."""

    def test_generates_for_all_classes(self):
        for cls in CLASS_FIELDS:
            spec = prompt_spec(cls)
            assert cls in spec
            assert len(spec) > 50  # non-trivial output

    def test_subtype_narrows_fields(self):
        full = prompt_spec("meter_reading")
        odo = prompt_spec("meter_reading", subtype="odometer")
        disp = prompt_spec("meter_reading", subtype="dispenser")
        # Subtype specs should be shorter than the full spec
        assert len(odo) < len(full)
        assert len(disp) < len(full)
        # Odometer spec should NOT mention amount_rs
        assert "amount_rs" not in odo
        # Dispenser spec should NOT mention odometer_km
        assert "odometer_km" not in disp

    def test_unknown_subtype_returns_full(self):
        full = prompt_spec("meter_reading")
        unknown = prompt_spec("meter_reading", subtype="unknown_device")
        assert full == unknown

    def test_compact_includes_positional_spec(self):
        spec = prompt_spec("work_report", compact=True)
        assert "positional" in spec.lower() or "array" in spec.lower()

    def test_verbose_excludes_compact_spec(self):
        spec = prompt_spec("work_report", compact=False)
        assert "positional" not in spec.lower()


class TestEmptyEnvelope:

    def test_shape(self):
        env = empty_envelope("doc_99")
        assert env["doc_id"] == "doc_99"
        assert env["status"] == "error"
        assert env["data"] is None
        assert env["refusal"] is None
        assert isinstance(env["quality_flags"], list)
        assert isinstance(env["provenance"]["calls"], list)
        assert isinstance(env["provenance"]["stages"], list)
        assert env["provenance"]["cost_usd"] == 0.0

    def test_independent_instances(self):
        a = empty_envelope("a")
        b = empty_envelope("b")
        a["provenance"]["stages"].append("test")
        assert "test" not in b["provenance"]["stages"]


class TestRefusal:

    def test_basic_refusal(self):
        ref = refusal("doc_40", "not_a_document", "DEF filler neck")
        assert ref["doc_id"] == "doc_40"
        assert ref["status"] == "refused"
        assert ref["data"] is None
        assert ref["refusal"]["reason"] == "not_a_document"
        assert ref["refusal"]["observed"] == "DEF filler neck"
        assert ref["confidence"] == 1.0

    def test_custom_confidence(self):
        ref = refusal("doc_40", "not_a_document", "", confidence=0.85)
        assert ref["confidence"] == 0.85

    def test_custom_doc_class(self):
        ref = refusal("doc_05", "duplicate", "re-sent copy", doc_class="duplicate")
        assert ref["doc_class"] == "duplicate"
