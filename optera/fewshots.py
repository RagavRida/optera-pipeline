"""Real few-shot examples drawn from hand-verified gold extractions.

These live in a separate module rather than inline in prompts.py so they can be
updated independently as the gold set grows. Every value here was read from the
image by a human and cross-checked against the documents themselves.

Purpose: padding the cached system prefix past the 2048-token Haiku floor so
that prompt caching actually fires. An empty RULEBOOK at ~400 tokens means
every extraction call has paid full price for identical bytes — the cache
directive was accepted and silently ignored.

An above-floor RULEBOOK with real examples also directly improves accuracy by
showing the model what correct output looks like on this exact document type,
rather than relying on general instruction-following.
"""
from __future__ import annotations

# -----------------------------------------------------------------
# METER READING — TATA DEF dispenser (optera_doc_41)
# Shows: dispenser subtype, arithmetic-valid values, no ODO fields.
# -----------------------------------------------------------------
METER_DISPENSER_EXAMPLE = """\
EXAMPLE — meter_reading / dispenser
Input: photograph of a TATA Motors DEF dispenser display showing four readings.

Correct output:
{
  "doc_class": "meter_reading",
  "confidence": 0.95,
  "data": {
    "reading_type": "def_dispenser",
    "amount_rs": 3199.76,
    "litres": 43.24,
    "rate_per_litre": 74.00,
    "urea_concentration_pct": 32.50,
    "vehicle_no": null,
    "captured_at": null
  }
}

Why each choice was made:
- reading_type "def_dispenser": the display shows Rs / Litres / Rs-per-Litre / % Urea — a DEF pump, not an ODO cluster.
- 43.24 × 74.00 = 3199.76 exactly — the arithmetic closes, confidence is high.
- captured_at null: "Shot on OnePlus / Amit Singh Rathore / 15 July 2026 at 12:12" is a phone watermark, not a visible overlay on the device display itself.
- No odometer_km, trip_km, fuel_level_pct: those are ODO-cluster fields. Dispenser subtypes never emit them.
"""

# -----------------------------------------------------------------
# METER READING — instrument cluster (optera_doc_38)
# Shows: odometer subtype, decimal transcription, null for non-visible fields.
# -----------------------------------------------------------------
METER_ODOMETER_EXAMPLE = """\
EXAMPLE — meter_reading / odometer
Input: photograph of a vehicle instrument cluster.

Correct output:
{
  "doc_class": "meter_reading",
  "confidence": 0.92,
  "data": {
    "reading_type": "odometer",
    "odometer_km": 32065.4,
    "trip_km": 1458.1,
    "fuel_level_pct": null,
    "vehicle_no": null,
    "captured_at": null
  }
}

Why each choice was made:
- odometer_km 32065.4 not 320654: the display shows a decimal point between the 5 and the 4. Transcribe what is shown, not a rounded integer.
- trip_km 1458.1: the TRIP A counter reads 1458.1, distinct from the ODO.
- AFE (average fuel efficiency shown as km/l on the cluster) was NOT placed in rate_per_litre. AFE is not a price. rate_per_litre is a dispenser field and does not exist on an odometer subtype.
- fuel_level_pct null: no numeric percentage is shown — only a gauge needle whose level cannot be precisely read.
- captured_at null: the cluster shows a time of day (08:21 PM) but no calendar date overlay.
"""

# -----------------------------------------------------------------
# VENDOR BILL — Tiwari Auto Parts (optera_doc_28)
# Shows: simple handwritten bill, null for blank fields, vehicle code.
# -----------------------------------------------------------------
VENDOR_BILL_EXAMPLE = """\
EXAMPLE — vendor_bill
Input: photograph of a printed Tiwari Auto Parts cash bill with one handwritten line item.

Correct output:
{
  "doc_class": "vendor_bill",
  "confidence": 0.91,
  "data": {
    "vendor_name": "TIWARI AUTO PARTS",
    "vendor_gstin": null,
    "vendor_phone": "87806 93146",
    "invoice_no": "312",
    "invoice_date": "2026-07-08",
    "buyer_name": "DTC",
    "vehicle_no": "GJ16AV 9065",
    "line_items": [
      {"description": "DEF TATA", "hsn": null, "qty": null, "rate": null, "amount": 1750}
    ],
    "subtotal": null,
    "tax_amount": null,
    "total_amount": 1750,
    "amount_in_words": null,
    "currency": "INR"
  }
}

Why each choice was made:
- vendor_gstin null: the printed form has a GSTIN field but it was left blank on this bill.
- invoice_date "2026-07-08": the handwritten date reads "08/07/2026". Indian forms are day-first; day=08, month=07, year=2026.
- vehicle_no "GJ16AV 9065": transcribed exactly as written, preserving the operator's spacing.
- qty and rate null: the bill shows only the amount column; individual qty and rate were not filled in.
- amount_in_words null: the "Rupees:" line was left blank by the cashier.
"""

# -----------------------------------------------------------------
# VENDOR BILL — Sajid Tyre Service (optera_doc_35)
# Shows: checklist form, only priced rows, arithmetic cross-check.
# -----------------------------------------------------------------
VENDOR_BILL_CHECKLIST_EXAMPLE = """\
EXAMPLE — vendor_bill (printed checklist form)
Input: photograph of a Sajid Tyre Service cash memo with 15 printed service rows, only 3 of which have amounts filled in.

Correct output:
{
  "doc_class": "vendor_bill",
  "confidence": 0.88,
  "data": {
    "vendor_name": "SAJID TYRE SERVICE",
    "vendor_gstin": null,
    "vendor_phone": null,
    "invoice_no": "196",
    "invoice_date": "2026-07-07",
    "buyer_name": null,
    "vehicle_no": "GJ16 AV 2305",
    "line_items": [
      {"description": "Puncture Fitting", "hsn": null, "qty": 1, "rate": null, "amount": 150},
      {"description": "Machine Fitting",  "hsn": null, "qty": 1, "rate": null, "amount": 50},
      {"description": "Omni Patch",       "hsn": null, "qty": null, "rate": null, "amount": 1400}
    ],
    "subtotal": null,
    "tax_amount": null,
    "total_amount": 1600,
    "amount_in_words": null,
    "currency": "INR"
  }
}

Why each choice was made:
- Only 3 of the 15 printed rows have amounts. Rows 2 (Extra Puncture), 4 (Langot), 5 (Geater), 13 (Air Check, ticked) etc. have tick marks but NO amount — they are free services. Do not invent amounts.
- 150 + 50 + 1400 = 1600 — arithmetic confirms the total.
- date "2026-07-07": handwritten as "07/07/26", day-first Indian format, year 2026.
"""

# -----------------------------------------------------------------
# WORK REPORT — Paldi form (optera_doc_20)
# Shows: structured rows, bus code format, no MECH column on this form.
# -----------------------------------------------------------------
WORK_REPORT_EXAMPLE = """\
EXAMPLE — work_report (Paldi ruled form, page 2 of a day)
Input: photograph of a PALDI WORK REPORT form, SR numbers 12–16 (this is page 2 of the day).

Correct output:
{
  "doc_class": "work_report",
  "confidence": 0.82,
  "data": {
    "depot": "Paldi",
    "report_date": "2026-06-16",
    "page_label": "Page 1",
    "entries": [
      [12, null, "TAM17",  "All wheel brake set",                              null, false],
      [13, null, "MAM28",  "Front Right Side tyre puncture repair complete",   null, false],
      [14, null, "MAM20",  "Rear Right outer tyre puncture made and fixed in MAM28, Rear Right outer side", null, false],
      [15, null, "TAM13",  "Rear Left outer tyre puncture made and fix it",    null, false],
      [16, null, null,     "20 buses air fill complete",                       null, false]
    ]
  }
}

Why each choice was made:
- depot "Paldi": printed in the form header as "Paldi WORK REPORT".
- report_date "2026-06-16": handwritten as "16-06-2026", day-first.
- entries as positional arrays [sr_no, mechanic, bus_no, work_done, material, struck_through].
- SR 16 has no bus_no: "20 buses air fill complete" refers to all buses, not one.
- MECH column is blank on every row — null, not invented.
- Bus codes: TAM17 not TCM17, MAM28 not MAM23. Read the first letter (T vs M) and second letter (A vs C) carefully. These codes are a 3-letter prefix + number. Common confusions: T/M, A/C/M/P.
"""

# -----------------------------------------------------------------
# NOT A DOCUMENT — DEF filler neck (optera_doc_40)
# Shows: the refusal that lexical similarity traps trip up.
# -----------------------------------------------------------------
NOT_A_DOCUMENT_EXAMPLE = """\
EXAMPLE — not_a_document (object carrying a text label)
Input: photograph of a DEF filler neck on a Cummins engine, showing a blue cap labelled "DEF ONLY".

Correct output:
{
  "doc_class": "not_a_document",
  "confidence": 0.97,
  "data": null,
  "refusal": {
    "reason": "not_a_document",
    "observed": "close-up photograph of a DEF filler neck on a vehicle engine; the blue cap reads DEF ONLY but there are no readings, amounts, or structured records present"
  }
}

Why: The text "DEF ONLY" is a moulded label on a mechanical part, not a dispenser reading. A dispenser shows numeric values: Rs, Litres, Rs/Litre. This image shows none of those. Do not attempt to extract meter_reading fields from an object photo just because it carries the word DEF.
"""


# -----------------------------------------------------------------
# VENDOR BILL — Anupam/Exide computer-generated tax invoice (doc_26)
# Shows: GST breakdown, two-digit arithmetic, GSTIN field.
# -----------------------------------------------------------------
VENDOR_BILL_GST_EXAMPLE = """\
EXAMPLE — vendor_bill (computer-generated GST invoice with CGST+SGST)
Input: photograph of an Anupam Enterprise / EXIDE Authorised Dealer computer-printed tax invoice.

Correct output:
{
  "doc_class": "vendor_bill",
  "confidence": 0.94,
  "data": {
    "vendor_name": "ANUPAM ENTERPRISE",
    "vendor_gstin": "24ACWPP0885C1ZB",
    "vendor_phone": "9408008181",
    "invoice_no": "AE/268/26-27",
    "invoice_date": "2026-06-20",
    "buyer_name": "Avtarsingh Mahendrasingh Ramgadia",
    "vehicle_no": "GJ 16 AU 9885",
    "line_items": [
      {"description": "XP1000 (battery)", "hsn": "85071000", "qty": 2, "rate": 6610.17, "amount": 13220.34}
    ],
    "subtotal": 13220.34,
    "tax_amount": 2379.66,
    "total_amount": 15600.00,
    "amount_in_words": "INR Fifteen Thousand Six Hundred Only",
    "currency": "INR"
  }
}

Why each choice was made:
- vendor_gstin "24ACWPP0885C1ZB" is the VENDOR's GSTIN in the letterhead. The BUYER's GSTIN
  "24ABNPR5802P1Z2" appears in the Bill-To section. Do not mix them up.
- tax_amount 2379.66 = CGST 1189.83 + SGST 1189.83. Report the TOTAL GST, not one component.
- total_amount 15600.00: subtotal 13220.34 + tax 2379.66 = 15600.00. The arithmetic closes.
- invoice_no "AE/268/26-27": three digits "268", not two. Read each digit of the number
  separately — "8" and "6" look similar in some typefaces; the context "26-27" (financial year)
  confirms the sequence.
"""

# -----------------------------------------------------------------
# VENDOR BILL — Puran Car Seat (doc_32)
# Shows: two line items summing to total, seat/upholstery shop.
# -----------------------------------------------------------------
VENDOR_BILL_SEAT_EXAMPLE = """\
EXAMPLE — vendor_bill (seat upholstery shop, two items)
Input: photograph of a Puran Car Seat bill with two handwritten service line items.

Correct output:
{
  "doc_class": "vendor_bill",
  "confidence": 0.88,
  "data": {
    "vendor_name": "PURAN CAR SEAT",
    "vendor_gstin": "24BOMPM8191K1ZA",
    "vendor_phone": "8980234090",
    "invoice_no": "691",
    "invoice_date": "2026-06-04",
    "buyer_name": "GSRTC DTC",
    "vehicle_no": "GJ 16 AV 6545",
    "line_items": [
      {"description": "Driver Seat / Driver Back", "hsn": null, "qty": 2, "rate": null, "amount": 700},
      {"description": "Conductor Seat / Conductor Back", "hsn": null, "qty": 2, "rate": null, "amount": 700}
    ],
    "subtotal": null,
    "tax_amount": null,
    "total_amount": 1400,
    "amount_in_words": null,
    "currency": "INR"
  }
}

Why: 700 + 700 = 1400. Two distinct seat types at the same price. The rate-per-unit is not
printed — only the line amount — so rate is null on both rows.
"""

# -----------------------------------------------------------------
# WORK REPORT — dense Paldi form (doc_14)
# Shows: Gujarati/Devanagari first two entries, English rest.
# -----------------------------------------------------------------
WORK_REPORT_GUJARATI_EXAMPLE = """\
EXAMPLE — work_report (mixed Gujarati/English, Paldi depot, 10 entries)
Input: photograph of a Paldi WORK REPORT form. The first two entries are written in
Gujarati/Devanagari script; the remaining entries are in English.

Correct output (condensed):
{
  "doc_class": "work_report",
  "confidence": 0.78,
  "data": {
    "depot": "Paldi",
    "report_date": "2026-06-14",
    "page_label": "Page 1",
    "entries": [
      [null, null, "TCM06",   "ઢਁચ સਰ੍ਵਿਸ ਕਰੀਵੀ",   null, false],
      [null, null, "TCMEX1",  "નਵਾਂ ਕਾਚਾ ਫੋਰ ਸਬਮਿਸ਼ਨ",   null, false],
      [null, null, "MAM43",   "al pca battery was faulty",   null, false],
      [null, null, "TCM47",   "cell, altenator, battery was checked",   null, false],
      [null, null, "MAM19",   "driver cabin fan was faulty",   null, false],
      [null, null, "MAM21",   "drive cabin fan wiring repair",   null, false],
      [null, null, "TAM26",   "al pca spark plug change, al pca head light bulb change",   null, false],
      [null, null, "TAM24",   "throttle body change with",   null, false],
      [null, null, "MAM18",   "starter motor repair, alpsa starter solenoid change",   null, false],
      [null, null, "TAM12",   "fan whisekus wiring repair, al pca shocket new",   null, false]
    ]
  }
}

KEY RULES demonstrated here:
- Gujarati/Devanagari text in work_done is KEPT AS-IS. Do not transliterate to Latin.
- SR column was blank on this page — null for all rows.
- MECH column was blank — null for all rows.
- Bus codes here: TCM06, TCMEX1, MAM43, TCM47, MAM19, MAM21, TAM26, TAM24, MAM18, TAM12.
  Note TCMEX1 (has Ex suffix). Note TAM vs TCM vs MAM — read the middle letter (A vs C vs M) carefully.
"""


# -----------------------------------------------------------------
# KNOWN FAILURE MODE REFERENCE — summarises the errors caught in evaluation
# -----------------------------------------------------------------
FAILURE_MODE_REFERENCE = """\
KNOWN FAILURE MODES — read before every extraction

1. DECIMAL ODOMETERS
   "32065.4" is NOT "320654". The decimal separates the last digit. If you see a point
   on the display, transcribe it. A six-digit odometer without a decimal is also valid
   (459794.3 on a higher-mileage vehicle). Never round or drop the decimal.

2. AFE ≠ RATE_PER_LITRE
   Vehicle instrument clusters show AFE (Average Fuel Efficiency) in km/l. This is engine
   performance data, not a price. It must never go into rate_per_litre. The rate_per_litre
   field only exists for dispenser subtypes; it does not appear on odometer schemas at all.

3. CLOCK ≠ DATE
   "08:21 PM" on a dashboard display is the time of day shown by the cluster's internal
   clock. It is NOT a date. Do not construct captured_at from it. Only set captured_at
   if a visible date-and-time overlay (e.g. a camera watermark) is present in the image.

4. VENDOR vs BUYER GSTIN
   Tax invoices show both parties' GSTINs. vendor_gstin is in the letterhead block
   (top of the bill). The buyer's GSTIN is in the "Bill To" or "Buyer" section.
   They are 15-character strings with the same format — the only way to tell them apart
   is position, not format.

5. BLANK CHECKLIST ROWS
   Printed service forms list 10–15 possible services. Rows with no amount filled in —
   even if ticked — are FREE services. Do not invent an amount. A tick mark with no
   rupee figure means the service was performed at no charge.

6. BUS CODE PREFIXES
   The codes are 3-letter prefix + number. Common confusions:
   - First letter: T vs M  (look at the stroke — T has a horizontal top, M has two peaks)
   - Second letter: A vs C vs M vs P
   - Suffix: some codes carry "-Ex" (e.g. TCMEX1, TCMEX2) — preserve the suffix exactly.
   Transcribe what is written. If the letter is genuinely ambiguous, give your best single
   reading rather than null. A wrong code is recoverable by fuzzy-matching against the fleet
   database; a null is a lost record.

7. OBJECT PHOTOS WITH TEXT LABELS
   A DEF filler neck labelled "DEF ONLY", a battery sticker, a tyre sidewall marking —
   these carry text but are not documents. The test is: does the image show readings,
   amounts, dates, or structured records? If no, refuse it. The presence of text alone
   is not sufficient reason to extract.
"""


def all_examples() -> str:
    """Return all few-shot examples as one block for inclusion in the system prompt."""
    return "\n\n".join([
        "=" * 72,
        "FEW-SHOT EXAMPLES — real documents from the Optera fleet-operator corpus.",
        "Study the examples and the failure-mode reference before extracting.",
        "=" * 72,
        FAILURE_MODE_REFERENCE,
        METER_DISPENSER_EXAMPLE,
        METER_ODOMETER_EXAMPLE,
        VENDOR_BILL_EXAMPLE,
        VENDOR_BILL_CHECKLIST_EXAMPLE,
        VENDOR_BILL_GST_EXAMPLE,
        VENDOR_BILL_SEAT_EXAMPLE,
        WORK_REPORT_EXAMPLE,
        WORK_REPORT_GUJARATI_EXAMPLE,
        NOT_A_DOCUMENT_EXAMPLE,
    ])


def examples_for(doc_class: str, subtype: str | None = None) -> str:
    """Return only the examples most relevant to this extraction call."""
    if doc_class == "meter_reading":
        if subtype == "dispenser":
            return METER_DISPENSER_EXAMPLE
        return METER_ODOMETER_EXAMPLE + "\n\n" + METER_DISPENSER_EXAMPLE
    if doc_class == "vendor_bill":
        return VENDOR_BILL_EXAMPLE + "\n\n" + VENDOR_BILL_CHECKLIST_EXAMPLE
    if doc_class == "work_report":
        return WORK_REPORT_EXAMPLE
    return NOT_A_DOCUMENT_EXAMPLE
