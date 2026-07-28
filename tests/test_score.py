"""Tests for optera.score — accuracy scoring helpers."""
from __future__ import annotations

import pytest
from optera.score import _norm_code, _norm_text, _num_match, _text_match


class TestNormCode:
    """Vehicle code normalisation — punctuation invariance."""

    def test_hyphenated(self):
        assert _norm_code("GJ-06-AV-4045") == "gj06av4045"

    def test_spaced(self):
        assert _norm_code("GJ 06 AV 4045") == "gj06av4045"

    def test_no_punctuation(self):
        assert _norm_code("GJ06AV4045") == "gj06av4045"

    def test_all_forms_equal(self):
        a = _norm_code("GJ-06-AV-4045")
        b = _norm_code("GJ 06 AV 4045")
        c = _norm_code("GJ06AV4045")
        assert a == b == c

    def test_none(self):
        assert _norm_code(None) == ""

    def test_bus_code(self):
        assert _norm_code("TAM17") == "tam17"

    def test_with_suffix(self):
        assert _norm_code("TCM-Ex1") == "tcmex1"


class TestNormText:
    """Text normalisation for comparison."""

    def test_basic(self):
        assert _norm_text("Hello World!") == "hello world"

    def test_special_chars(self):
        assert _norm_text("TIWARI AUTO PARTS") == "tiwari auto parts"

    def test_none(self):
        assert _norm_text(None) == ""

    def test_numeric_string(self):
        assert _norm_text("312") == "312"


class TestNumMatch:
    """Numeric comparison with tolerance."""

    def test_exact(self):
        assert _num_match(150.0, 150.0) is True

    def test_within_tolerance(self):
        assert _num_match(150.5, 150.0) is True  # 0.5/150 < 0.5%

    def test_outside_tolerance(self):
        assert _num_match(200, 150) is False

    def test_zero_gold(self):
        assert _num_match(0, 0) is True
        assert _num_match(0.001, 0) is False

    def test_none_values(self):
        assert _num_match(None, 150) is False
        assert _num_match(150, None) is False
        assert _num_match(None, None) is False

    def test_string_numbers(self):
        assert _num_match("150.0", 150) is True

    def test_indian_format(self):
        assert _num_match("1,50,000", 150000) is True


class TestTextMatch:
    """Text comparison with containment for long strings."""

    def test_exact(self):
        assert _text_match("TIWARI AUTO PARTS", "TIWARI AUTO PARTS") is True

    def test_case_insensitive(self):
        assert _text_match("tiwari auto parts", "TIWARI AUTO PARTS") is True

    def test_containment_long(self):
        assert _text_match("Anupam", "Anupam Enterprise") is True
        assert _text_match("Anupam Enterprise", "Anupam") is True

    def test_containment_too_short(self):
        # Short strings should not match via containment
        assert _text_match("Auto", "Auto Parts") is False  # "Auto" < 5 chars

    def test_none_values(self):
        assert _text_match(None, "anything") is False
        assert _text_match("anything", None) is False

    def test_empty_strings(self):
        assert _text_match("", "") is False
        assert _text_match("", "something") is False
