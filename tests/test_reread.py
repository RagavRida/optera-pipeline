from optera.reread import _accept_if_improved, _crop_box, _fields_for


def test_vendor_crop_is_top_of_page_and_only_targets_invalid_gstin():
    assert _crop_box("vendor_bill", (1000, 2000)) == (0, 0, 1000, 840)
    assert _fields_for("vendor_bill", ["malformed_gstin:'not-a-gstin'"]) == ["vendor_gstin"]
    assert _fields_for("vendor_bill", ["missing_total_amount"]) == []


def test_reread_merge_requires_validation_improvement():
    original = {"vendor_name": "A", "vendor_gstin": "not-a-gstin", "total_amount": 100, "line_items": []}
    merged, accepted = _accept_if_improved("vendor_bill", original, {"vendor_gstin": "24ABCDE1234F1Z5"})
    assert accepted
    assert merged["vendor_gstin"] == "24ABCDE1234F1Z5"

    unchanged, accepted = _accept_if_improved("vendor_bill", original, {"vendor_gstin": None})
    assert not accepted
    assert unchanged == original
