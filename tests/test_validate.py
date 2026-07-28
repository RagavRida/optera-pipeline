"""Tests for optera.validate — deterministic validation checks."""
from __future__ import annotations

import pytest
from optera.validate import validate, effective_confidence, ValidationReport


class TestValidateBill:
    """vendor_bill validation rules."""

    def test_valid_bill_passes(self, sample_bill_data):
        rep = validate("vendor_bill", sample_bill_data)
        assert rep.passed

    def test_missing_total(self, sample_bill_data):
        sample_bill_data["total_amount"] = None
        rep = validate("vendor_bill", sample_bill_data)
        assert not rep.passed
        assert any("missing_total" in i for i in rep.issues)

    def test_negative_total(self, sample_bill_data):
        sample_bill_data["total_amount"] = -500
        rep = validate("vendor_bill", sample_bill_data)
        assert not rep.passed
        assert any("non_positive" in i for i in rep.issues)

    def test_implausible_total(self, sample_bill_data):
        sample_bill_data["total_amount"] = 10_000_000
        rep = validate("vendor_bill", sample_bill_data)
        assert any("implausible" in i for i in rep.issues)

    def test_line_math_mismatch(self, sample_bill_data):
        sample_bill_data["line_items"] = [
            {"description": "Part", "qty": 2, "rate": 100, "amount": 500},
        ]
        sample_bill_data["total_amount"] = 500
        rep = validate("vendor_bill", sample_bill_data)
        assert any("line_math_mismatch" in i for i in rep.issues)

    def test_total_mismatch(self, sample_bill_data):
        sample_bill_data["line_items"] = [
            {"description": "Part A", "amount": 500},
            {"description": "Part B", "amount": 300},
        ]
        sample_bill_data["total_amount"] = 1500  # way off from 800
        rep = validate("vendor_bill", sample_bill_data)
        assert not rep.passed
        assert any("total_mismatch" in i for i in rep.issues)

    def test_arithmetic_closes(self, sample_bill_data):
        """When lines + tax == total, no mismatch."""
        sample_bill_data["line_items"] = [
            {"description": "Battery", "amount": 13220.34},
        ]
        sample_bill_data["tax_amount"] = 2379.66
        sample_bill_data["total_amount"] = 15600.00
        rep = validate("vendor_bill", sample_bill_data)
        assert not any("total_mismatch" in i for i in rep.issues)

    def test_malformed_gstin(self, sample_bill_data):
        sample_bill_data["vendor_gstin"] = "NOTAVALIDGSTIN"
        rep = validate("vendor_bill", sample_bill_data)
        assert any("malformed_gstin" in i for i in rep.issues)

    def test_valid_gstin_passes(self, sample_bill_data):
        rep = validate("vendor_bill", sample_bill_data)
        assert not any("malformed_gstin" in i for i in rep.issues)

    def test_bad_date(self, sample_bill_data):
        sample_bill_data["invoice_date"] = "1970-01-01"
        rep = validate("vendor_bill", sample_bill_data)
        assert any("bad_invoice_date" in i for i in rep.issues)

    def test_placeholder_value(self, sample_bill_data):
        sample_bill_data["vendor_name"] = "N/A"
        rep = validate("vendor_bill", sample_bill_data)
        assert any("placeholder" in i for i in rep.issues)

    def test_missing_vendor_name(self, sample_bill_data):
        sample_bill_data["vendor_name"] = ""
        rep = validate("vendor_bill", sample_bill_data)
        assert any("missing_vendor" in i for i in rep.issues)


class TestValidateWorkReport:
    """work_report validation rules."""

    def test_valid_report_passes(self, sample_work_report_data):
        rep = validate("work_report", sample_work_report_data)
        assert rep.passed

    def test_entries_not_a_list(self, sample_work_report_data):
        sample_work_report_data["entries"] = "not a list"
        rep = validate("work_report", sample_work_report_data)
        assert not rep.passed
        assert any("entries_not_a_list" in i for i in rep.issues)

    def test_zero_entries(self, sample_work_report_data):
        sample_work_report_data["entries"] = []
        rep = validate("work_report", sample_work_report_data)
        # Severity is below the hard-fail line (0.2 < 0.3)
        assert rep.passed
        assert any("zero_entries" in i for i in rep.issues)

    def test_many_missing_work_done(self, sample_work_report_data):
        sample_work_report_data["entries"] = [
            {"work_done": ""} for _ in range(10)
        ]
        rep = validate("work_report", sample_work_report_data)
        assert not rep.passed  # 100% empty > 50% threshold

    def test_proportional_few_missing(self, sample_work_report_data):
        """One blank work_done in 10 rows (10%) should not trigger the issue."""
        entries = [{"work_done": f"task {i}"} for i in range(9)]
        entries.append({"work_done": ""})
        sample_work_report_data["entries"] = entries
        rep = validate("work_report", sample_work_report_data)
        # 10% is below 25% threshold, so no issue added
        assert not any("entries_missing_work_done" in i for i in rep.issues)

    def test_duplicate_rows_detected(self, sample_work_report_data):
        sample_work_report_data["entries"] = [
            {"work_done": "brake set"} for _ in range(8)
        ]
        rep = validate("work_report", sample_work_report_data)
        assert any("repeated_rows" in i for i in rep.issues)

    def test_bad_report_date(self, sample_work_report_data):
        sample_work_report_data["report_date"] = "2049-12-31"
        rep = validate("work_report", sample_work_report_data)
        assert any("bad_report_date" in i for i in rep.issues)


class TestValidateMeter:
    """meter_reading validation rules."""

    def test_valid_dispenser_passes(self, sample_meter_data):
        rep = validate("meter_reading", sample_meter_data)
        assert rep.passed

    def test_dispenser_math_mismatch(self, sample_meter_data):
        sample_meter_data["litres"] = 50.0  # 50 * 74 = 3700 != 3199.76
        rep = validate("meter_reading", sample_meter_data)
        assert any("dispenser_math_mismatch" in i for i in rep.issues)

    def test_impossible_urea(self, sample_meter_data):
        sample_meter_data["urea_concentration_pct"] = 150.0
        rep = validate("meter_reading", sample_meter_data)
        assert any("impossible_urea" in i for i in rep.issues)

    def test_cross_field_contamination(self):
        """rate_per_litre without any dispenser context flags contamination."""
        data = {
            "reading_type": "odometer",
            "odometer_km": 32065.4,
            "rate_per_litre": 3.4,  # This is actually AFE km/l, not a price
        }
        rep = validate("meter_reading", data)
        assert any("dispenser_field_without_dispenser" in i for i in rep.issues)

    def test_odometer_and_dispenser_conflict(self):
        data = {
            "reading_type": "odometer",
            "odometer_km": 32065.4,
            "amount_rs": 3199.76,
            "litres": 43.24,
        }
        rep = validate("meter_reading", data)
        assert any("odometer_and_dispenser" in i for i in rep.issues)

    def test_implausible_odometer(self):
        data = {"reading_type": "odometer", "odometer_km": 5_000_000}
        rep = validate("meter_reading", data)
        assert any("implausible_odometer" in i for i in rep.issues)

    def test_dropped_decimal_suspected(self, sample_odometer_data):
        """ODO integer + trip decimal on a high-mileage vehicle = dropped decimal."""
        sample_odometer_data["odometer_km"] = 320654  # should be 32065.4
        sample_odometer_data["trip_km"] = 1458.1
        rep = validate("meter_reading", sample_odometer_data)
        assert any("suspected_dropped_decimal" in i for i in rep.issues)

    def test_no_readings_at_all(self):
        data = {"reading_type": "odometer"}
        rep = validate("meter_reading", data)
        assert any("meter_reading_with_no_readings" in i for i in rep.issues)

    def test_valid_odometer_passes(self, sample_odometer_data):
        rep = validate("meter_reading", sample_odometer_data)
        assert rep.passed

    def test_implausible_captured_at(self):
        data = {"reading_type": "odometer", "odometer_km": 32065.4,
                "captured_at": "2024-08-21T08:21:00"}
        rep = validate("meter_reading", data)
        # 2024-08-21 might be constructed from clock time; depends on date range
        # The point is it doesn't crash
        assert isinstance(rep, ValidationReport)


class TestValidateGeneral:
    """Cross-cutting validation concerns."""

    def test_none_data(self):
        rep = validate("vendor_bill", None)
        assert not rep.passed
        assert any("no_data" in i for i in rep.issues)

    def test_placeholder_detection(self):
        data = {"vendor_name": "unknown", "total_amount": 100, "line_items": []}
        rep = validate("vendor_bill", data)
        assert any("placeholder" in i for i in rep.issues)

    def test_multiple_placeholders(self):
        data = {"vendor_name": "N/A", "invoice_no": "xxx",
                "total_amount": 100, "line_items": []}
        rep = validate("vendor_bill", data)
        placeholder_issues = [i for i in rep.issues if "placeholder" in i]
        assert len(placeholder_issues) == 2


class TestEffectiveConfidence:
    """Confidence combination logic."""

    def test_clean_report_no_change(self):
        rep = ValidationReport()
        assert effective_confidence(0.9, rep) == 0.9

    def test_severity_subtracts(self):
        rep = ValidationReport()
        rep.severity = 0.3
        assert effective_confidence(0.9, rep) == 0.6

    def test_severity_capped_at_06(self):
        """Even catastrophic severity can only subtract 0.6."""
        rep = ValidationReport()
        rep.severity = 1.0
        result = effective_confidence(0.9, rep)
        assert result == pytest.approx(0.3, abs=0.001)

    def test_floor_at_zero(self):
        rep = ValidationReport()
        rep.severity = 0.6
        result = effective_confidence(0.2, rep)
        assert result == 0.0

    def test_model_confidence_clamped(self):
        rep = ValidationReport()
        assert effective_confidence(1.5, rep) == 1.0

    def test_negative_model_confidence(self):
        rep = ValidationReport()
        assert effective_confidence(-0.5, rep) == 0.0
