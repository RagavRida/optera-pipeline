"""Shared test fixtures."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_bill_data():
    """A valid vendor bill extraction result."""
    return {
        "vendor_name": "TIWARI AUTO PARTS",
        "vendor_gstin": "24ACWPP0885C1ZB",
        "vendor_phone": "87806 93146",
        "invoice_no": "312",
        "invoice_date": "2026-07-08",
        "buyer_name": "DTC",
        "vehicle_no": "GJ16AV 9065",
        "line_items": [
            {"description": "DEF TATA", "hsn": None, "qty": 1, "rate": 1750, "amount": 1750},
        ],
        "subtotal": None,
        "tax_amount": None,
        "total_amount": 1750,
        "amount_in_words": None,
        "currency": "INR",
    }


@pytest.fixture
def sample_meter_data():
    """A valid meter reading extraction result (dispenser)."""
    return {
        "reading_type": "def_dispenser",
        "amount_rs": 3199.76,
        "litres": 43.24,
        "rate_per_litre": 74.00,
        "urea_concentration_pct": 32.50,
        "vehicle_no": None,
        "captured_at": None,
    }


@pytest.fixture
def sample_odometer_data():
    """A valid odometer reading."""
    return {
        "reading_type": "odometer",
        "odometer_km": 32065.4,
        "trip_km": 1458.1,
        "fuel_level_pct": None,
        "vehicle_no": None,
        "captured_at": None,
    }


@pytest.fixture
def sample_work_report_data():
    """A valid work report with entries."""
    return {
        "depot": "Paldi",
        "report_date": "2026-06-16",
        "page_label": "Page 1",
        "entries": [
            {"sr_no": "12", "mechanic": None, "bus_no": "TAM17",
             "work_done": "All wheel brake set", "material": None, "struck_through": False},
            {"sr_no": "13", "mechanic": None, "bus_no": "MAM28",
             "work_done": "Front Right Side tyre puncture repair", "material": None, "struck_through": False},
        ],
    }


@pytest.fixture
def sample_envelope():
    """An empty pipeline envelope."""
    from optera.schemas import empty_envelope
    return empty_envelope("test_doc_01")


@pytest.fixture
def tmp_image(tmp_path):
    """Create a minimal valid JPEG file for testing."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def tmp_png(tmp_path):
    """Create a minimal valid PNG file for testing."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="blue")
    path = tmp_path / "test.png"
    img.save(path, "PNG")
    return path


@pytest.fixture
def tmp_html_as_jpg(tmp_path):
    """An HTML file masquerading as a .jpg — the kind of thing WhatsApp delivers."""
    path = tmp_path / "error.jpg"
    path.write_text("<!doctype html><html><body>404 Not Found</body></html>")
    return path
