"""Tests for optera.jsonio — tolerant JSON parsing and number coercion."""
from __future__ import annotations

import pytest
from optera.jsonio import parse, coerce_number


class TestParse:
    """JSON recovery from model output."""

    def test_clean_json(self):
        obj, note = parse('{"key": "value", "n": 42}')
        assert obj == {"key": "value", "n": 42}
        assert note == ""

    def test_empty_string(self):
        obj, note = parse("")
        assert obj is None
        assert note == "empty_response"

    def test_whitespace_only(self):
        obj, note = parse("   \n  ")
        assert obj is None
        assert note == "empty_response"

    def test_markdown_fenced(self):
        text = 'Here is the result:\n```json\n{"doc_class": "vendor_bill"}\n```'
        obj, note = parse(text)
        assert obj == {"doc_class": "vendor_bill"}
        assert note == "unfenced"

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"x": 1}\n```'
        obj, note = parse(text)
        assert obj == {"x": 1}
        assert note == "unfenced"

    def test_prose_before_json(self):
        text = 'Based on my analysis, the result is:\n{"doc_class": "meter_reading", "confidence": 0.9}'
        obj, note = parse(text)
        assert obj is not None
        assert obj["doc_class"] == "meter_reading"

    def test_prose_after_json(self):
        text = '{"key": "val"}\nHere is what I found...'
        obj, note = parse(text)
        assert obj == {"key": "val"}

    def test_truncated_json_balanced(self):
        text = '{"data": {"entries": [{"bus_no": "TAM17", "work_done": "brake'
        obj, note = parse(text)
        assert obj is not None
        assert note == "repaired_truncated"
        assert "bus_no" in obj.get("data", {}).get("entries", [{}])[0]

    def test_truncated_array_balanced(self):
        text = '{"items": [1, 2, 3'
        obj, note = parse(text)
        assert obj is not None
        assert obj["items"] == [1, 2, 3]

    def test_truncated_top_level_array_balanced(self):
        text = '[{"doc_class":"meter_reading","data":{"odometer_km":32065.4'
        obj, note = parse(text)
        assert note == "repaired_truncated"
        assert isinstance(obj, list)
        assert obj[0]["data"]["odometer_km"] == 32065.4

    def test_no_json_at_all(self):
        obj, note = parse("This is just plain text with no JSON.")
        assert obj is None
        assert "no_json_object" in note

    def test_nested_objects(self):
        text = '{"a": {"b": {"c": 1}}, "d": [1, 2]}'
        obj, note = parse(text)
        assert obj["a"]["b"]["c"] == 1
        assert obj["d"] == [1, 2]
        assert note == ""

    def test_trailing_comma_in_truncated(self):
        """Trailing commas from truncation should be stripped before balancing."""
        text = '{"a": 1, "b": 2,'
        obj, note = parse(text)
        assert obj is not None
        assert obj["a"] == 1


class TestCoerceNumber:
    """Indian-format number parsing."""

    def test_int(self):
        assert coerce_number(42) == 42.0

    def test_float(self):
        assert coerce_number(3.14) == 3.14

    def test_string_simple(self):
        assert coerce_number("150") == 150.0

    def test_string_with_commas(self):
        assert coerce_number("1,50,000") == 150000.0

    def test_string_with_currency(self):
        assert coerce_number("Rs 200") == 200.0

    def test_string_with_rupee_symbol(self):
        assert coerce_number("₹1,500.50") == 1500.50

    def test_negative(self):
        assert coerce_number("-42") == -42.0

    def test_none(self):
        assert coerce_number(None) is None

    def test_bool_false(self):
        assert coerce_number(False) is None

    def test_bool_true(self):
        assert coerce_number(True) is None

    def test_empty_string(self):
        assert coerce_number("") is None

    def test_whitespace(self):
        assert coerce_number("   ") is None

    def test_dash_only(self):
        assert coerce_number("-") is None

    def test_dot_only(self):
        assert coerce_number(".") is None

    def test_zero(self):
        assert coerce_number(0) == 0.0

    def test_string_zero(self):
        assert coerce_number("0") == 0.0

    def test_decimal(self):
        assert coerce_number("3199.76") == 3199.76
